"""Independent symbolic replay and communication cost model for AgentCC.

This implements the published rules locally; it does not claim to replace the
official Judge. Times returned here are microseconds. No tensor data is needed.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from .topology import Topology


class ReplayError(ValueError):
    """A task or schedule violates a published replay constraint."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unique_strings(value: Any, field: str) -> list[str]:
    _require(isinstance(value, list), f"{field} must be an array")
    _require(all(isinstance(item, str) for item in value), f"{field} must contain strings")
    _require(len(value) == len(set(value)), f"{field} contains duplicates")
    return value


def _solution_body(task: dict, value: dict) -> dict:
    _require(isinstance(value, dict), "solution must be an object")
    for field in ("task_id", "instance_id"):
        if field in value and field in task:
            _require(value[field] == task[field], f"{field} does not match the task")
    if "output" in value:
        _require(value.get("status") == "success", "response status is not success")
        output = value["output"]
        _require(isinstance(output, dict), "output must be an object")
        value = output.get("answer")
        _require(isinstance(value, dict), "output.answer must be an object")
    if "solution" in value:
        value = value["solution"]
        _require(isinstance(value, dict), "answer.solution must be an object")
    for field in ("task_id", "instance_id"):
        if field in value and field in task:
            _require(value[field] == task[field], f"solution {field} does not match the task")
    return value


def replay(task: dict, topology: dict | Topology, solution: dict) -> dict:
    """Validate and time a solution, returning a JSON-friendly report.

    Invalid schedules return ``valid=False`` and an explanatory ``errors`` list.
    The replay maintains its own round snapshots, separate from the generator.
    The supported path forms are explicit ``path`` and array-valued ``route``.
    """
    try:
        graph = topology if isinstance(topology, Topology) else Topology.from_data(topology)
        return _replay(task, graph, _solution_body(task, solution))
    except (ReplayError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "total_time_us": None,
            "round_times_us": [],
            "nonempty_rounds": 0,
            "split": None,
        }


def _replay(task: dict, graph: Topology, solution: dict) -> dict:
    ranks = _unique_strings(task.get("ranks", task.get("npu")), "task.ranks")
    _require(bool(ranks), "task.ranks cannot be empty")
    _require(ranks == list(graph.ranks), "task.ranks and topology.ranks must agree in order")
    rank_index = {rank: i for i, rank in enumerate(ranks)}
    operator = task.get("operator")
    _require(operator in ("all-gather", "all-reduce"), "unsupported operator")
    _require(solution.get("operator") == operator, "solution operator does not match task")
    chunks = _unique_strings(solution.get("chunks"), "chunks")
    _require(len(chunks) > 0 and len(chunks) % len(ranks) == 0, "incomplete rank × split chunks")
    split = len(chunks) // len(ranks)
    policy = task.get("split_policy", {})
    candidates = policy.get("candidates", [task.get("split", 1)])
    _require(split in candidates, "split is not in split_policy.candidates")
    if policy.get("mode", "search_allowed") != "search_allowed":
        _require(split == task.get("split", 1), "task requires its fixed split")
    if "split" in solution:
        _require(_integer(solution["split"]) and solution["split"] == split, "solution.split disagrees with chunks")
    data_bytes = task.get("data_size", {}).get("per_rank_bytes")
    alignment = policy.get("alignment_bytes", 1)
    _require(_integer(data_bytes) and data_bytes > 0, "per_rank_bytes must be a positive integer")
    _require(_integer(alignment) and alignment > 0, "alignment_bytes must be a positive integer")
    _require(data_bytes % split == 0, "split does not divide per_rank_bytes")
    chunk_bytes = data_bytes // split
    _require(chunk_bytes % alignment == 0, "chunk size violates alignment_bytes")

    chunk_ids = {
        f"{rank}_c{slot}": (rank_id, slot)
        for rank_id, rank in enumerate(ranks)
        for slot in range(split)
    }
    _require(set(chunks) == set(chunk_ids), "chunks must contain exactly every rank and slot")
    initial = solution.get("initial_possession")
    _require(isinstance(initial, dict) and set(initial) == set(ranks), "initial_possession must cover all ranks")
    for rank in ranks:
        own = _unique_strings(initial[rank], f"initial_possession[{rank}]")
        _require(set(own) == {f"{rank}_c{slot}" for slot in range(split)}, "initial_possession must contain only the rank's own slots")

    full_contributors = (1 << len(ranks)) - 1
    expression_cache: dict[str, tuple[int, int]] = {}

    def expression(token: str) -> tuple[int, int]:
        """Return (slot, contributor bits), rejecting duplicate contributions."""
        if token in expression_cache:
            return expression_cache[token]
        total = re.fullmatch(r"c(\d+)_sum", token)
        if total:
            slot = int(total.group(1))
            _require(slot < split, "complete result refers to an unknown slot")
            result = (slot, full_contributors)
        else:
            slot = None
            contributors = 0
            for part in token.split("+"):
                _require(part in chunk_ids, f"unknown contribution: {part}")
                source, part_slot = chunk_ids[part]
                _require(slot is None or slot == part_slot, "expression mixes reduction slots")
                _require(not contributors & (1 << source), "expression repeats a contributor")
                slot = part_slot
                contributors |= 1 << source
            result = (slot, contributors)
        expression_cache[token] = result
        return result

    final = _unique_strings(solution.get("final_chunks"), "final_chunks")
    if operator == "all-gather":
        _require(set(final) == set(chunks), "AllGather final_chunks must contain all original chunks")
    else:
        finals = [expression(token) for token in final]
        _require(len(finals) == split and {slot for slot, _ in finals} == set(range(split)), "AllReduce final_chunks must cover every slot exactly once")
        _require(all(mask == full_contributors for _, mask in finals), "AllReduce final_chunks must name complete reductions")

    rounds = solution.get("rounds")
    _require(isinstance(rounds, list), "rounds must be an array")
    _require(_integer(solution.get("round_count")) and solution["round_count"] == len(rounds), "round_count must equal the number of declared rounds")
    max_rounds = task.get("max_nonempty_rounds", 64)
    _require(_integer(max_rounds) and max_rounds >= 0, "invalid max_nonempty_rounds")
    all_chunks_mask = (1 << len(chunks)) - 1
    own_slots_mask = (1 << split) - 1
    ag_state = [own_slots_mask << (i * split) for i in range(len(ranks))]
    ar_state = [[1 << i] * split for i in range(len(ranks))]
    sent_reductions: set[tuple[int, int, int, int]] = set()
    path_cache: dict[tuple[str, ...], tuple[tuple, tuple, float, float]] = {}
    # The same immutable payload is multicast to several destinations. Cache
    # only its validated chunk mask, never sender ownership or receive state:
    # those still have to be checked separately for every transfer and round.
    ag_payloads: dict[tuple[str, ...], int] = {}
    ag_bits = {token: 1 << (owner * split + slot)
               for token, (owner, slot) in chunk_ids.items()} if operator == "all-gather" else {}
    round_times: list[float] = []

    for number, round_data in enumerate(rounds, 1):
        _require(isinstance(round_data, dict), f"round {number} must be an object")
        _require(_integer(round_data.get("round")) and round_data["round"] == number, "round numbers must be sequential starting at 1")
        transfers = round_data.get("transfers")
        _require(isinstance(transfers, list), f"round {number} transfers must be an array")
        if not transfers:
            continue
        _require(len(round_times) < max_rounds, "too many nonempty rounds")
        edge_load: dict[tuple[str, str], int] = defaultdict(int)
        resource_load: dict[str, int] = defaultdict(int)
        pending_ag = [0] * len(ranks)
        pending_ar: dict[tuple[int, int], int] = {}
        max_path_time = 0.0
        max_overhead = 0.0

        for transfer_number, transfer in enumerate(transfers, 1):
            context = f"round {number}, transfer {transfer_number}"
            try:
                _require(isinstance(transfer, dict), "transfer must be an object")
                source, destination = transfer.get("src"), transfer.get("dst")
                _require(source in rank_index and destination in rank_index and source != destination, "transfer endpoints must be distinct ranks")
                src, dst = rank_index[source], rank_index[destination]
                action = transfer.get("action")
                _require(action in ("copy", "reduce"), "unknown transfer action")
                _require(action != "copy" or task.get("allow_copy", True), "copy is disabled by the task")
                tokens = transfer.get("chunk_ranges")
                _require(isinstance(tokens, list), "chunk_ranges must be an array")
                _require(bool(tokens), "transfer must carry at least one chunk")

                if operator == "all-gather":
                    _require(action == "copy", "AllGather only permits copy")
                    payload_key = tuple(tokens)
                    mask = ag_payloads.get(payload_key)
                    if mask is None:
                        _unique_strings(tokens, "chunk_ranges")
                        mask = 0
                        for token in tokens:
                            _require(token in ag_bits, f"unknown AllGather chunk: {token}")
                            mask |= ag_bits[token]
                        ag_payloads[payload_key] = mask
                    _require(mask & ag_state[src] == mask, "sender does not possess chunks at the start of the round")
                    _require(not mask & (ag_state[dst] | pending_ag[dst]), "destination receives an already held or duplicated chunk")
                    pending_ag[dst] |= mask
                    payload = len(tokens) * chunk_bytes
                else:
                    _unique_strings(tokens, "chunk_ranges")
                    payload_slots: set[int] = set()
                    for token in tokens:
                        slot, contributors = expression(token)
                        _require(slot not in payload_slots, "transfer repeats a reduction slot")
                        payload_slots.add(slot)
                        _require(ar_state[src][slot] == contributors, "expression differs from the sender's start-of-round contributions")
                        pair = (slot, contributors, src, dst)
                        _require(pair not in sent_reductions, "duplicate (chunk, src, dst) transmission")
                        sent_reductions.add(pair)
                        pending_key = (dst, slot)
                        previous = pending_ar.get(pending_key, 0)
                        if action == "copy":
                            _require(contributors == full_contributors, "AllReduce copy requires a complete reduction")
                            _require(ar_state[dst][slot] != full_contributors and not previous, "destination receives a duplicate or conflicting complete result")
                            pending_ar[pending_key] = full_contributors
                        else:
                            _require(not contributors & (ar_state[dst][slot] | previous), "reduce includes overlapping contributions")
                            pending_ar[pending_key] = previous | contributors
                    payload = len(payload_slots) * chunk_bytes

                path = transfer.get("path", transfer.get("route"))
                _require(isinstance(path, list) and len(path) >= 2, "transfer needs an explicit path array")
                _require(all(isinstance(node, str) for node in path), "path nodes must be strings")
                _require(path[0] == source and path[-1] == destination, "path endpoints do not match transfer endpoints")
                path_key = tuple(path)
                if path_key not in path_cache:
                    _require(len(set(path)) == len(path), "path repeats a node")
                    _require(all(node in graph.switches for node in path[1:-1]), "path interior must contain only declared switches")
                    edges = tuple(zip(path, path[1:]))
                    _require(all(edge in graph.edges for edge in edges), "path uses an undeclared directed edge")
                    resources = tuple(resource for edge in edges for resource in graph.edge_resources.get(edge, ()))
                    overhead = graph.startup_us + (len(path) - 2) * graph.switch_overhead_us
                    inv_bandwidth = sum(1_000_000.0 / graph.edges[edge] for edge in edges)
                    path_cache[path_key] = edges, resources, overhead, inv_bandwidth
                edges, resources, overhead, inv_bandwidth = path_cache[path_key]
                for edge in edges:
                    edge_load[edge] += payload
                for resource in resources:
                    resource_load[resource] += payload
                max_path_time = max(max_path_time, overhead + payload * inv_bandwidth)
                max_overhead = max(max_overhead, overhead)
            except ReplayError as exc:
                raise ReplayError(f"{context}: {exc}") from exc

        edge_busy = max((1_000_000.0 * amount / graph.edges[edge] for edge, amount in edge_load.items()), default=0.0)
        resource_busy = max((1_000_000.0 * amount / graph.resources[resource] for resource, amount in resource_load.items()), default=0.0)
        round_times.append(max(max_path_time, max_overhead + max(edge_busy, resource_busy)))
        if operator == "all-gather":
            ag_state = [held | received for held, received in zip(ag_state, pending_ag)]
        else:
            for (rank, slot), contributors in pending_ar.items():
                ar_state[rank][slot] |= contributors

    if operator == "all-gather":
        _require(all(held == all_chunks_mask for held in ag_state), "AllGather final state is incomplete")
    else:
        _require(all(mask == full_contributors for rank in ar_state for mask in rank), "AllReduce final state is incomplete")
    return {
        "valid": True,
        "errors": [],
        "total_time_us": sum(round_times),
        "round_times_us": round_times,
        "nonempty_rounds": len(round_times),
        "split": split,
    }
