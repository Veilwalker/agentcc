"""Topology-aware collective constructors used as measured search candidates.

These constructors infer locality from the supplied edges. They do not use
topology family names, saved schedules, or undeclared physical connections.
"""

from __future__ import annotations

from collections import defaultdict
import math
import time


def locality_order(graph) -> list[str]:
    """Build a balanced rank hierarchy from actual two-way link affinities.

    Direct rank links and two-hop shared-switch bandwidth imply affinity.
    A switch connected to many ranks contributes less than a local switch, so
    a global fabric does not erase strong local neighborhoods. Greedy matching
    at successive levels gives contiguous groups of 2, 4, 8, ... ranks.
    """
    ranks = list(graph.ranks)
    n = len(ranks)
    if n < 2 or n & (n - 1):
        return ranks
    index = {rank: i for i, rank in enumerate(ranks)}
    affinity = [[0.0] * n for _ in range(n)]
    switch_members = defaultdict(dict)
    for (src, dst), capacity in graph.edges.items():
        if src in index and dst in index:
            backward = graph.edges.get((dst, src), 0.0)
            affinity[index[src]][index[dst]] += min(capacity, backward)
        elif src in index and dst in graph.switches:
            backward = graph.edges.get((dst, src), 0.0)
            if backward:
                switch_members[dst][index[src]] = min(capacity, backward)
    for members in switch_members.values():
        entries = list(members.items())
        divisor = max(1, len(entries) - 1)
        for offset, (left, left_capacity) in enumerate(entries):
            for right, right_capacity in entries[offset + 1:]:
                value = left_capacity * right_capacity / (left_capacity + right_capacity) / divisor
                affinity[left][right] += value
                affinity[right][left] += value
    groups = [[rank] for rank in ranks]
    while len(groups) > 1:
        size = len(groups)
        choices = sorted((-affinity[left][right], left, right)
                         for left in range(size) for right in range(left + 1, size))
        matched, pairs = set(), []
        for _, left, right in choices:
            if left not in matched and right not in matched:
                pairs.append((left, right))
                matched.update((left, right))
        pairs.sort(key=lambda pair: min(index[rank] for i in pair for rank in groups[i]))
        next_groups = [groups[left] + groups[right] for left, right in pairs]
        next_affinity = [[0.0] * len(pairs) for _ in pairs]
        for first, (a, b) in enumerate(pairs):
            for second in range(first + 1, len(pairs)):
                c, d = pairs[second]
                value = affinity[a][c] + affinity[a][d] + affinity[b][c] + affinity[b][d]
                next_affinity[first][second] = value
                next_affinity[second][first] = value
        groups, affinity = next_groups, next_affinity
    return groups[0]


def _initial(task: dict, ranks: list[str], split: int) -> dict:
    owned = {rank: [f"{rank}_c{slot}" for slot in range(split)] for rank in ranks}
    chunks = [chunk for rank in ranks for chunk in owned[rank]]
    return {"operator": task["operator"], "split": split, "chunks": chunks,
            "initial_possession": owned,
            "final_chunks": chunks.copy() if task["operator"] == "all-gather" else
            [f"c{slot}_sum" for slot in range(split)], "round_count": 0, "rounds": []}


def _round(solution: dict, movements: dict, graph, chunk_bytes: int,
           resource_weight: float, deadline: float, packets: int = 1) -> None:
    transfers = []
    for (action, src, dst), chunks in movements.items():
        if time.monotonic() >= deadline:
            raise TimeoutError("collective construction budget exhausted")
        path = graph.path(src, dst, size_bytes=chunk_bytes, resource_weight=resource_weight)
        packet_size = max(1, (len(chunks) + packets - 1) // packets)
        for first in range(0, len(chunks), packet_size):
            transfers.append({"action": action, "src": src, "dst": dst,
                              "chunk_ranges": chunks[first:first + packet_size], "path": list(path)})
    if transfers:
        solution["rounds"].append({"round": len(solution["rounds"]) + 1, "transfers": transfers})
        solution["round_count"] = len(solution["rounds"])


def build(task: dict, graph, split: int = 1, *, kind: str = "reverse_doubling",
          order: list[str] | None = None, resource_weight: float = 1.0,
          deadline: float = math.inf, packets: int = 1,
          routing_bytes: float | None = None) -> dict:
    """Construct a candidate. The caller must independently replay its result.

    Reverse doubling sends each original contribution across broad locality
    cuts before replicating it along narrower, faster local links. Ordinary
    low-to-high doubling performs those two operations in the opposite order.
    """
    ranks = list(graph.ranks)
    n = len(ranks)
    if not n or n & (n - 1):
        raise ValueError("this collective requires a power-of-two rank count")
    if (task["operator"], kind) not in {("all-gather", "reverse_doubling"), ("all-reduce", "xor_tree")}:
        raise ValueError("unsupported collective constructor")
    if type(split) is not int or split <= 0:
        raise ValueError("split must be a positive integer")
    if type(packets) is not int or not 1 <= packets <= 16:
        raise ValueError("packets must be an integer from 1 to 16")
    data_bytes = task["data_size"]["per_rank_bytes"]
    policy = task.get("split_policy", {})
    if split not in policy.get("candidates", [task.get("split", 1)]) or data_bytes % split:
        raise ValueError("invalid task split")
    if data_bytes // split % policy.get("alignment_bytes", 1):
        raise ValueError("split violates byte alignment")
    representative_bytes = data_bytes // split if routing_bytes is None else routing_bytes
    if (isinstance(representative_bytes, bool) or not isinstance(representative_bytes, (int, float))
            or not math.isfinite(representative_bytes) or representative_bytes <= 0):
        raise ValueError("routing_bytes must be finite and positive")
    ordered = locality_order(graph) if order is None else list(order)
    if len(ordered) != n or set(ordered) != set(ranks):
        raise ValueError("rank order must be a permutation of the input ranks")
    solution = _initial(task, ranks, split)
    if kind == "xor_tree":
        stages = (n - 1).bit_length()
        rounds = [defaultdict(list) for _ in range(stages * 2)]
        # Odd strides permute all power-of-two rank IDs. Unlike a cyclic shift
        # of the rank list, XOR keeps every established locality cut intact.
        root_stride = n // split + 1
        if root_stride % 2 == 0:
            root_stride += 1
        for slot in range(split):
            root = slot * root_stride % n
            slot_order = [ordered[index ^ root] for index in range(n)]
            for stage in range(stages):
                distance = 1 << stage
                for first in range(0, n, distance * 2):
                    src, dst = slot_order[first + distance], slot_order[first]
                    contributions = "+".join(f"{rank}_c{slot}" for rank in
                                             slot_order[first + distance:first + distance * 2])
                    rounds[stage][("reduce", src, dst)].append(contributions)
                    rounds[2 * stages - stage - 1][("copy", dst, src)].append(f"c{slot}_sum")
        for movements in rounds:
            _round(solution, movements, graph, representative_bytes, resource_weight, deadline, packets)
        if solution["round_count"] > task.get("max_nonempty_rounds", 64):
            raise ValueError("collective exceeds task round limit")
        return solution
    held = [[rank] for rank in ordered]
    distance = n // 2
    while distance:
        movements = {}
        for index, source in enumerate(ordered):
            target = ordered[index ^ distance]
            movements[("copy", source, target)] = [
                chunk for owner in held[index] for chunk in solution["initial_possession"][owner]]
        _round(solution, movements, graph, representative_bytes, resource_weight, deadline, packets)
        held = [owners + held[index ^ distance] for index, owners in enumerate(held)]
        distance //= 2
    if solution["round_count"] > task.get("max_nonempty_rounds", 64):
        raise ValueError("collective exceeds task round limit")
    return solution
