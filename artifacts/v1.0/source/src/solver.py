"""Construct and compare legal collective schedules from the current input graph.

The language model supplies small search hints, never executable code or paths.
Every returned candidate has passed the independent local replay implementation.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
import time
from typing import Any

from .replay import replay
from .topology import Topology


class SolverError(ValueError):
    """No locally valid schedule was found within the allowed search budget."""


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("schedule search budget exhausted")


def _splits(task: dict) -> list[int]:
    data = task.get("data_size")
    if not isinstance(data, dict):
        raise SolverError("data_size must be an object")
    data_bytes = data.get("per_rank_bytes")
    policy = task.get("split_policy", {})
    if not isinstance(policy, dict):
        raise SolverError("split_policy must be an object")
    alignment = policy.get("alignment_bytes", 1)
    values = policy.get("candidates", [task.get("split", 1)])
    if type(data_bytes) is not int or data_bytes <= 0:
        raise SolverError("per_rank_bytes must be a positive integer")
    if type(alignment) is not int or alignment <= 0:
        raise SolverError("alignment_bytes must be a positive integer")
    if not isinstance(values, list) or not values or any(type(value) is not int or value <= 0 for value in values):
        raise SolverError("split candidates must be a nonempty array of positive integers")
    suggested = task.get("split", 1)
    if type(suggested) is not int or suggested <= 0:
        raise SolverError("task.split must be a positive integer")
    mode = policy.get("mode", "search_allowed")
    if not isinstance(mode, str):
        raise SolverError("split_policy.mode must be a string")
    if mode != "search_allowed":
        if suggested not in values:
            raise SolverError("fixed split must be in split_policy.candidates")
        values = [suggested]
    result = sorted({value for value in values if data_bytes % (value * alignment) == 0})
    if not result:
        raise SolverError("no allowed split satisfies the byte alignment")
    return result


def validate_task(task: dict, graph: Topology) -> list[int]:
    """Validate task constraints before planning, returning feasible split values.

    This performs no model calls, path search, or mutation. Individual candidate
    splits that fail byte alignment are filtered; at least one must be usable.
    """
    if not isinstance(task, dict):
        raise SolverError("task must be an object")
    ranks = task.get("ranks", task.get("npu"))
    if not isinstance(ranks, list) or not ranks or any(not isinstance(rank, str) for rank in ranks):
        raise SolverError("task ranks must be a nonempty array of strings")
    if ranks != list(graph.ranks):
        raise SolverError("task ranks must match topology ranks in order")
    if "npu" in task and task["npu"] != ranks:
        raise SolverError("task.npu must match task.ranks")
    if not isinstance(task.get("operator"), str) or task["operator"] not in {"all-gather", "all-reduce"}:
        raise SolverError("unsupported operator")
    max_rounds = task.get("max_nonempty_rounds", 64)
    if type(max_rounds) is not int or max_rounds < 0:
        raise SolverError("max_nonempty_rounds must be a nonnegative integer")
    if len(ranks) > 1 and max_rounds == 0:
        raise SolverError("multiple ranks require at least one communication round")
    for flag in ("allow_copy", "allow_detour", "bidirectional"):
        if flag in task and not isinstance(task[flag], bool):
            raise SolverError(f"{flag} must be boolean")
    if not task.get("allow_copy", True) and len(ranks) > 1:
        raise SolverError("this constructor requires allow_copy")
    constraints = task.get("communication_constraints", {})
    if not isinstance(constraints, dict):
        raise SolverError("communication_constraints must be an object")
    for flag in ("receive_unique_chunk_once", "send_only_possessed_chunks",
                 "send_unique_chunk_pair_once", "switches_forward_same_round_without_storage"):
        if flag in constraints and not isinstance(constraints[flag], bool):
            raise SolverError(f"communication_constraints.{flag} must be boolean")
    return _splits(task)


def _solution(task: dict, ranks: list[str], split: int) -> dict:
    possession = {rank: [f"{rank}_c{slot}" for slot in range(split)] for rank in ranks}
    chunks = [chunk for rank in ranks for chunk in possession[rank]]
    return {
        "operator": task["operator"],
        "split": split,
        "chunks": chunks,
        "initial_possession": possession,
        "final_chunks": chunks.copy() if task["operator"] == "all-gather" else
        [f"c{slot}_sum" for slot in range(split)],
        "round_count": 0,
        "rounds": [],
    }


def _topology_order(graph: Topology, ranks: list[str]) -> list[str]:
    """Group equal switch neighborhoods without inferring meaning from node names."""
    neighborhoods: dict[str, set[str]] = {rank: set() for rank in ranks}
    for src, dst in graph.edges:
        if src in neighborhoods and dst in graph.switches:
            neighborhoods[src].add(dst)
        if dst in neighborhoods and src in graph.switches:
            neighborhoods[dst].add(src)
    positions = {rank: index for index, rank in enumerate(ranks)}
    return sorted(ranks, key=lambda rank: (tuple(sorted(neighborhoods[rank])), positions[rank]))


def _append_round(solution: dict, movements: dict, graph: Topology,
                  chunk_bytes: int, resource_weight: float, deadline: float) -> None:
    transfers = []
    for (action, src, dst), expressions in movements.items():
        _check_deadline(deadline)
        # Route with one representative chunk so all rounds reuse a source tree.
        # Replay still charges the full transfer payload, including every slot.
        path = graph.path(src, dst, size_bytes=chunk_bytes, resource_weight=resource_weight)
        transfers.append({"action": action, "src": src, "dst": dst,
                          "chunk_ranges": expressions, "path": list(path)})
    if transfers:
        solution["rounds"].append({"round": len(solution["rounds"]) + 1, "transfers": transfers})
        solution["round_count"] = len(solution["rounds"])


def _allgather_doubling(task: dict, graph: Topology, split: int, order: list[str],
                        weight: float, deadline: float) -> dict:
    n = len(order)
    if n & (n - 1):
        raise SolverError("recursive doubling requires a power-of-two rank count")
    result = _solution(task, graph.ranks, split)
    chunk_bytes = task["data_size"]["per_rank_bytes"] // split
    distance = 1
    while distance < n:
        movements = {}
        for index, src in enumerate(order):
            first = index // distance * distance
            expressions = [f"{rank}_c{slot}" for rank in order[first:first + distance]
                           for slot in range(split)]
            movements[("copy", src, order[index ^ distance])] = expressions
        _append_round(result, movements, graph, chunk_bytes, weight, deadline)
        distance *= 2
    return result


def _tree(task: dict, graph: Topology, split: int, order: list[str],
          weight: float, deadline: float) -> dict:
    """Balanced reduction/gather followed by dissemination, with roots per slot."""
    n = len(order)
    result = _solution(task, graph.ranks, split)
    if n == 1:
        return result
    stages = (n - 1).bit_length()
    rounds: list[dict] = [defaultdict(list) for _ in range(stages * 2)]
    allreduce = task["operator"] == "all-reduce"
    for slot in range(split):
        _check_deadline(deadline)
        # Spread completed slots across ranks instead of overloading one root.
        root_index = slot * n // split
        slot_order = order[root_index:] + order[:root_index]
        for stage in range(stages):
            step = 1 << stage
            for first in range(0, n, step * 2):
                middle = first + step
                if middle >= n:
                    continue
                dst, src = slot_order[first], slot_order[middle]
                sources = slot_order[middle:min(first + step * 2, n)]
                if allreduce:
                    expressions = ["+".join(f"{rank}_c{slot}" for rank in sources)]
                    rounds[stage][("reduce", src, dst)].extend(expressions)
                    rounds[stages * 2 - stage - 1][("copy", dst, src)].append(f"c{slot}_sum")
                else:
                    rounds[stage][("copy", src, dst)].extend(f"{rank}_c{slot}" for rank in sources)
                    # The subtree root retained its gathered data; send only missing data.
                    source_set = set(sources)
                    missing = [rank for rank in order if rank not in source_set]
                    rounds[stages * 2 - stage - 1][("copy", dst, src)].extend(
                        f"{rank}_c{slot}" for rank in missing)
    chunk_bytes = task["data_size"]["per_rank_bytes"] // split
    for movements in rounds:
        _append_round(result, movements, graph, chunk_bytes, weight, deadline)
    return result


def _hierarchical_allgather(task: dict, graph: Topology, split: int, order: list[str],
                           weight: float, deadline: float, group_size: int) -> dict:
    """Gather within groups, exchange between leaders, then distribute locally."""
    n = len(order)
    if n % group_size or group_size >= n or n & (n - 1):
        raise SolverError("hierarchical candidate needs equal power-of-two groups")
    groups = [order[first:first + group_size] for first in range(0, n, group_size)]
    result = _solution(task, graph.ranks, split)
    chunk_bytes = task["data_size"]["per_rank_bytes"] // split
    stages = (group_size - 1).bit_length()
    for stage in range(stages):
        step = 1 << stage
        movements = defaultdict(list)
        for group in groups:
            for first in range(0, group_size, step * 2):
                src, dst = group[first + step], group[first]
                movements[("copy", src, dst)].extend(
                    f"{rank}_c{slot}" for rank in group[first + step:first + 2 * step]
                    for slot in range(split))
        _append_round(result, movements, graph, chunk_bytes, weight, deadline)
    distance = 1
    while distance < len(groups):
        movements = defaultdict(list)
        for index, group in enumerate(groups):
            first = index // distance * distance
            src, dst = group[0], groups[index ^ distance][0]
            movements[("copy", src, dst)].extend(
                f"{rank}_c{slot}" for held_group in groups[first:first + distance]
                for rank in held_group for slot in range(split))
        _append_round(result, movements, graph, chunk_bytes, weight, deadline)
        distance *= 2
    for stage in reversed(range(stages)):
        step = 1 << stage
        movements = defaultdict(list)
        for group in groups:
            for first in range(0, group_size, step * 2):
                src, dst = group[first], group[first + step]
                held = set(group[first + step:first + 2 * step])
                movements[("copy", src, dst)].extend(
                    f"{rank}_c{slot}" for rank in order if rank not in held
                    for slot in range(split))
        _append_round(result, movements, graph, chunk_bytes, weight, deadline)
    return result


def _rank_overlay(graph: Topology, deadline: float) -> dict[str, list[str]]:
    """Reachability for a single transfer; ranks are endpoints, never interior nodes."""
    adjacency: dict[str, list[str]] = defaultdict(list)
    for src, dst in graph.edges:
        adjacency[src].append(dst)
    rank_set = set(graph.ranks)
    result = {}
    for source in graph.ranks:
        _check_deadline(deadline)
        seen, reachable = {source}, set()
        queue = deque([source])
        while queue:
            for node in adjacency[queue.popleft()]:
                if node in seen:
                    continue
                seen.add(node)
                if node in rank_set:
                    reachable.add(node)
                else:
                    queue.append(node)
        result[source] = [rank for rank in graph.ranks if rank in reachable]
    return result


def _bfs_tree(adjacency: dict, root: str) -> tuple[dict, dict]:
    parents, levels = {}, {root: 0}
    queue = deque([root])
    while queue:
        src = queue.popleft()
        for dst in adjacency[src]:
            if dst not in levels:
                parents[dst], levels[dst] = src, levels[src] + 1
                queue.append(dst)
    if len(levels) != len(adjacency):
        raise SolverError("rank transfer graph is not strongly connected")
    return parents, levels


def _routed_fallback(task: dict, graph: Topology, split: int, weight: float,
                     deadline: float) -> dict:
    """Use explicit multi-round rank relays when arbitrary rank pairs lack paths."""
    result = _solution(task, graph.ranks, split)
    if len(graph.ranks) == 1:
        return result
    adjacency = _rank_overlay(graph, deadline)
    max_rounds = task.get("max_nonempty_rounds", 64)
    rounds: list[dict] = []
    if task["operator"] == "all-gather":
        for origin in graph.ranks:
            _check_deadline(deadline)
            parents, levels = _bfs_tree(adjacency, origin)
            depth = max(levels.values())
            if depth > max_rounds:
                raise SolverError("rank relay depth exceeds the round limit")
            while len(rounds) < depth:
                rounds.append(defaultdict(list))
            for dst, src in parents.items():
                rounds[levels[dst] - 1][("copy", src, dst)].extend(
                    f"{origin}_c{slot}" for slot in range(split))
    else:
        reverse = {rank: [] for rank in graph.ranks}
        for src, destinations in adjacency.items():
            for dst in destinations:
                reverse[dst].append(src)
        best_trees = None
        # Compare a few roots to keep fallback construction bounded.
        for root in graph.ranks[:8]:
            _check_deadline(deadline)
            inbound, in_levels = _bfs_tree(reverse, root)
            outbound, out_levels = _bfs_tree(adjacency, root)
            count = max(in_levels.values()) + max(out_levels.values())
            if best_trees is None or count < best_trees[0]:
                best_trees = (count, inbound, in_levels, outbound, out_levels)
        count, inbound, in_levels, outbound, out_levels = best_trees
        if count > max_rounds:
            raise SolverError("reduction and broadcast relay depth exceeds the round limit")
        depth = max(in_levels.values())
        rounds = [defaultdict(list) for _ in range(count)]
        contributors = {rank: [rank] for rank in graph.ranks}
        for src in sorted(inbound, key=in_levels.get, reverse=True):
            dst = inbound[src]
            for slot in range(split):
                expression = "+".join(f"{rank}_c{slot}" for rank in contributors[src])
                rounds[depth - in_levels[src]][("reduce", src, dst)].append(expression)
            contributors[dst].extend(contributors[src])
        for dst, src in outbound.items():
            rounds[depth + out_levels[dst] - 1][("copy", src, dst)].extend(
                f"c{slot}_sum" for slot in range(split))
    chunk_bytes = task["data_size"]["per_rank_bytes"] // split
    for movements in rounds:
        _append_round(result, movements, graph, chunk_bytes, weight, deadline)
    return result


def solve_schedule(task: dict, topology: dict | Topology, strategy: dict | None = None,
                   time_budget_s: float = 10.0) -> dict[str, Any]:
    """Return the lowest-cost locally valid candidate found before the deadline.

    ``strategy`` accepts algorithm (auto/recursive_doubling/tree), split,
    resource_weight (0..4), and rank_order (input/topology). Unknown or invalid
    hints are ignored; task constraints and replay always take precedence.
    """
    started = time.monotonic()
    if not math.isfinite(time_budget_s) or time_budget_s <= 0:
        raise SolverError("time_budget_s must be positive and finite")
    deadline = started + time_budget_s
    graph = topology if isinstance(topology, Topology) else Topology.from_data(topology)
    ranks = list(graph.ranks)
    allowed = validate_task(task, graph)
    hints = strategy if isinstance(strategy, dict) else {}
    weight = hints.get("resource_weight", 1.0)
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 4:
        weight = 1.0
    is_ag = task["operator"] == "all-gather"
    power_of_two = len(ranks) & (len(ranks) - 1) == 0
    default_algorithm = "recursive_doubling" if is_ag and power_of_two else "tree"
    preferred_algorithm = hints.get("algorithm", default_algorithm)
    if preferred_algorithm not in {"tree", "recursive_doubling"}:
        preferred_algorithm = default_algorithm
    if not is_ag or not power_of_two:
        preferred_algorithm = "tree"
    local_order = _topology_order(graph, ranks)
    hinted_order_name = "topology" if hints.get("rank_order") == "topology" else "input"
    hinted_order = local_order if hinted_order_name == "topology" else ranks
    hinted_split = hints.get("split")
    if isinstance(hinted_split, bool) or hinted_split not in allowed:
        hinted_split = allowed[0]
    requested = {key: hints[key] for key in ("algorithm", "split", "resource_weight", "rank_order")
                 if key in hints}
    adjustments = []
    # Bound the symbolic output of an AllGather candidate under the memory cap.
    # Splitting is still a model-controlled input within this implementation cap.
    if is_ag and len(ranks) * max(0, len(ranks) - 1) * hinted_split > 1_000_000:
        safe_splits = [value for value in allowed
                       if len(ranks) * max(0, len(ranks) - 1) * value <= 1_000_000]
        hinted_split = max(safe_splits, default=allowed[0])
        adjustments.append("AllGather split limited to bound symbolic response size")
    adopted = {"algorithm": preferred_algorithm, "split": hinted_split,
               "rank_order": hinted_order_name, "resource_weight": float(weight)}

    # The model proposal is replayed first: it must not be starved by baseline search.
    candidates = []
    if hints:
        candidates.append((preferred_algorithm, hinted_split, hinted_order, float(weight), "model", hinted_order_name))
    candidates.append((default_algorithm, allowed[0], ranks, 1.0, "baseline", "input"))
    if is_ag:
        # A shared inter-domain cut can favor gathering once and broadcasting
        # locally over every rank independently exchanging the remote half.
        candidates.append(("tree", allowed[0], local_order, float(weight), "local_search", "topology"))
        candidates.append(("tree", allowed[0], ranks, 1.0, "local_search", "input"))
        if power_of_two:
            for group_size in [8, 32]:
                if group_size < len(ranks):
                    candidates.append((f"hierarchical_{group_size}", allowed[0], local_order,
                                       float(weight), "local_search", "topology"))
    else:
        candidates.extend(("tree", split, local_order, float(weight), "local_search", "topology")
                          for split in sorted(set([min(allowed, key=lambda s: abs(s - 4)), allowed[-1]])))
    candidates.extend([(default_algorithm, allowed[0], local_order, float(weight), "local_search", "topology"),
                       (default_algorithm, allowed[0], ranks, 0.0, "local_search", "input"),
                       (default_algorithm, allowed[0], local_order, 4.0, "local_search", "topology")])
    best, best_time = None, float("inf")
    seen, errors = set(), []
    checked = 0
    winner = ""
    records = []
    winning_record = None
    for candidate_index, (algorithm, split, order, resource_weight, source, order_name) in enumerate(candidates):
        key = (algorithm, split, tuple(order), resource_weight)
        if key in seen and source != "model_compact":
            continue
        seen.add(key)
        if time.monotonic() >= deadline:
            break
        params = {"algorithm": algorithm, "split": split, "rank_order": order_name,
                  "resource_weight": resource_weight}
        record = {"source": source, "requested": requested if source in {"model", "model_compact"} else params.copy(),
                  "actual": params, "valid": False, "selected": False}
        if source == "model_compact":
            record["reason"] = "retry failed model proposal with the smallest feasible split"
        records.append(record)
        candidate_deadline = min(deadline, started + time_budget_s * 0.5) if source == "model" else deadline

        def queue_model_compact(*, timed_out: bool = False) -> None:
            # Retry the same small proposal only after a timeout, when cached
            # source routes can make progress. A semantic failure needs a change.
            if source == "model" and (split != allowed[0] or timed_out):
                candidates.insert(candidate_index + 1, (
                    algorithm, allowed[0], order, resource_weight, "model_compact", order_name))

        try:
            if algorithm.startswith("hierarchical_"):
                candidate = _hierarchical_allgather(task, graph, split, order, resource_weight,
                                                  candidate_deadline, int(algorithm.split("_")[1]))
            else:
                builder = _allgather_doubling if algorithm == "recursive_doubling" else _tree
                candidate = builder(task, graph, split, order, resource_weight, candidate_deadline)
            report = replay(task, graph, candidate)
            checked += 1
            if not report["valid"]:
                errors.extend(report.get("errors", []))
                record["errors"] = report.get("errors", [])[:3]
                queue_model_compact()
                continue
            record.update(valid=True, total_time_us=report["total_time_us"])
            if report["total_time_us"] < best_time:
                best, best_time = candidate, report["total_time_us"]
                winner = algorithm
                winning_record = record
                best["metrics"] = {"total_time": best_time / 1_000_000,
                                   "round_times": [value / 1_000_000 for value in report["round_times_us"]]}
        except (ValueError, TimeoutError) as exc:
            errors.append(str(exc))
            record["errors"] = [str(exc)]
            queue_model_compact(timed_out=isinstance(exc, TimeoutError))
    if best is None and time.monotonic() < deadline:
        try:
            candidate = _routed_fallback(task, graph, allowed[0], 1.0, deadline)
            report = replay(task, graph, candidate)
            checked += 1
            if report["valid"]:
                best, best_time, winner = candidate, report["total_time_us"], "rank_relay"
                winning_record = {"source": "fallback", "actual": {"algorithm": "rank_relay", "split": allowed[0],
                                  "resource_weight": 1.0}, "valid": True, "selected": False,
                                  "total_time_us": best_time}
                records.append(winning_record)
                best["metrics"] = {"total_time": best_time / 1_000_000,
                                   "round_times": [value / 1_000_000 for value in report["round_times_us"]]}
            else:
                errors.extend(report.get("errors", []))
        except (ValueError, TimeoutError) as exc:
            errors.append(str(exc))
    if best is None:
        detail = "; ".join(str(error) for error in errors[-3:]) or "search budget exhausted"
        raise SolverError("no locally valid schedule found: " + detail)
    winning_record["selected"] = True
    best["strategy"] = {"algorithm": winner, "candidates_replayed": checked,
                        "local_replay_valid": True, "model_requested": requested,
                        "model_adopted": adopted if hints else None,
                        "model_adjustments": adjustments, "candidates": records}
    return best
