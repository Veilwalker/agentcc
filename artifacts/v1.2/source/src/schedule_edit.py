"""Concrete, bounded route edits proposed by a model and checked by replay.

An edit moves a prefix of existing atomic chunks in a fixed transfer group to
precomputed legal paths. It cannot change endpoints, actions, rounds, or data.
The model chooses all movement amounts jointly; extraction never selects them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import time

from .replay import replay
from .planning import compact_summary
from .route_balance import _PathCosts, _RoundLoad


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _input_digest(task, graph) -> str:
    # Recompute from live semantic fields: input_sha256 is an ingestion cache
    # and would fail to detect an accidental in-memory capacity mutation.
    return _digest({"task": task, "ranks": graph.ranks, "switches": sorted(graph.switches),
                    "edges": sorted((src, dst, cap) for (src, dst), cap in graph.edges.items()),
                    "resources": graph.resources,
                    "bindings": sorted((src, dst, tuple(ids)) for (src, dst), ids in graph.edge_resources.items()),
                    "startup_us": graph.startup_us, "switch_overhead_us": graph.switch_overhead_us})


@dataclass(frozen=True)
class EditMember:
    transfer_index: int
    chunk_count: int
    path: tuple[str, ...]


@dataclass(frozen=True)
class EditGroup:
    id: str
    round_index: int
    members: tuple[EditMember, ...]
    max_chunks: int


@dataclass(frozen=True)
class EditContext:
    input_sha256: str
    baseline_sha256: str
    base: str
    before_us: float
    before_round_times_us: tuple[float, ...]
    groups: tuple[EditGroup, ...]
    problem_sha256: str


def _legal_path(graph, src, dst, path):
    return (len(path) >= 2 and path[0] == src and path[-1] == dst
            and len(path) == len(set(path))
            and all(node in graph.switches for node in path[1:-1])
            and all(edge in graph.edges for edge in zip(path, path[1:])))


def _round_times(solution, report):
    """Align replay's nonempty-round times with the original round indices."""
    times = iter(report["round_times_us"])
    return tuple(next(times) if round_["transfers"] else 0.0 for round_ in solution["rounds"])


def _avoiding_path(graph, src, dst, load, blocked_edges, blocked_resources, deadline):
    """One current-load-aware shortest path, with ranks restricted to terminals."""
    distance, previous = {src: 0.0}, {}
    pending = [(0.0, src)]
    visits = 0
    while pending:
        cost, node = heapq.heappop(pending)
        if cost != distance[node]:
            continue
        if node == dst:
            path = [dst]
            while path[-1] != src:
                path.append(previous[path[-1]])
            return tuple(reversed(path))
        visits += 1
        if visits % 16 == 0 and time.monotonic() >= deadline:
            return None
        for neighbor, inverse, _ in graph.adjacency[node]:
            if neighbor != dst and neighbor not in graph.switches:
                continue
            edge = node, neighbor
            resources = graph.edge_resources[edge]
            if edge in blocked_edges or any(name in blocked_resources for name in resources):
                continue
            # Cost is a route heuristic only; all proposals use exact replay.
            busy = load.edge_bytes[edge] * inverse
            busy += sum(load.resource_bytes[name] / graph.resources[name] for name in resources)
            candidate = cost + inverse + busy + graph.switch_overhead_us / 1e6
            if candidate < distance.get(neighbor, math.inf):
                distance[neighbor], previous[neighbor] = candidate, node
                heapq.heappush(pending, (candidate, neighbor))
    return None


def _path_options(graph, transfer, load, costs, outgoing, incoming, deadline, peaks):
    src, dst = transfer["src"], transfer["dst"]
    old = tuple(transfer.get("path", transfer.get("route", ())))
    options = set()
    if (src, dst) in graph.edges:
        options.add((src, dst))
    for switch in outgoing[src] & incoming[dst]:
        options.add((src, switch, dst))
    options.discard(old)
    critical_edges, critical_resources = peaks
    old_edges, old_resources, _, _ = costs.describe(old)
    blocked_edges = set(old_edges) & critical_edges
    blocked_resources = set(old_resources) & critical_resources
    # Enumerating common switches discovers equal-cost paths that static
    # Dijkstra tie-breaking never exposes. Multi-switch fabrics need a search.
    if len(old) > 3 or not options:
        path = _avoiding_path(graph, src, dst, load, blocked_edges, blocked_resources, deadline)
        if path:
            options.add(path)
    return [path for path in options if _legal_path(graph, src, dst, path)]


def _signature(graph, costs, old, new):
    old_edges, old_resources, _, _ = costs.describe(old)
    new_edges, new_resources, _, _ = costs.describe(new)
    delta = Counter(new_resources)
    delta.subtract(old_resources)
    # All shared resource effects must be identical within a group. Common
    # switch-to-switch edges also remain explicit; terminal edges are summarized
    # by conservative envelopes and independently accounted for during replay.
    resources = tuple(sorted((name, value) for name, value in delta.items() if value))
    old_internal = set(edge for edge in old_edges if all(n in graph.switches for n in edge))
    new_internal = set(edge for edge in new_edges if all(n in graph.switches for n in edge))
    edge_delta = tuple(sorted([(edge, -1) for edge in old_internal - new_internal]
                             + [(edge, 1) for edge in new_internal - old_internal]))
    return resources, edge_delta, tuple(old[1:-1]), tuple(new[1:-1])


def extract_problem(task: dict, graph, baseline: dict, *, time_budget_s: float = 0.5,
                    max_groups: int = 8, max_rounds: int = 4) -> tuple[dict, EditContext]:
    """Expose nonoverlapping groups and exact shared-resource load deltas.

    ``move_chunks`` always means whole original chunk IDs or whole AllReduce
    expressions, never bytes or individual contributors inside an expression.
    A deadline bounds alternate-path work; baseline validation and hashing are
    mandatory, even when the optional route-search budget is zero.
    """
    if (type(time_budget_s) not in (int, float) or not math.isfinite(time_budget_s)
            or time_budget_s < 0):
        raise ValueError("edit time budget must be finite and nonnegative")
    if type(max_groups) is not int or not 1 <= max_groups <= 16:
        raise ValueError("max_groups must be an integer in [1, 16]")
    if type(max_rounds) is not int or not 1 <= max_rounds <= 8:
        raise ValueError("max_rounds must be an integer in [1, 8]")
    started = time.monotonic()
    end = started + time_budget_s
    report = replay(task, graph, baseline)
    if not report["valid"]:
        raise ValueError("edit baseline is invalid: " + "; ".join(report["errors"]))
    input_hash, baseline_hash = _input_digest(task, graph), _digest(baseline)
    base = _digest([input_hash, baseline_hash])[:16]
    chunk_bytes = task["data_size"]["per_rank_bytes"] // report["split"]
    costs = _PathCosts(graph)
    loads = []
    for round_ in baseline["rounds"]:
        load = _RoundLoad(costs)
        for transfer in round_["transfers"]:
            load.add(transfer.get("path", transfer.get("route")), len(transfer["chunk_ranges"]) * chunk_bytes)
        loads.append(load)
    outgoing, incoming = defaultdict(set), defaultdict(set)
    for src, dst in graph.edges:
        if src not in graph.switches and dst in graph.switches:
            outgoing[src].add(dst)
        if src in graph.switches and dst not in graph.switches:
            incoming[dst].add(src)
    catalog = []
    rounds_used = 0
    per_round = max(1, min(3, max_groups // min(max_rounds, len(loads) or 1)))
    for ri in sorted(range(len(loads)), key=lambda index: -loads[index].time_us):
        if time.monotonic() >= end or rounds_used >= max_rounds or len(catalog) >= max_groups:
            break
        load = loads[ri]
        critical_edges, critical_resources = load.bottlenecks()
        # A small catalog cannot clear dozens of independent, tied terminal
        # bottlenecks. Moving a subset merely leaves the same maximum behind
        # while loading a shared fabric. Concentrate on coupled constraints or
        # on the few-path cases that bounded edits can actually influence.
        if not critical_resources and len(critical_edges) > 4 and load.path_max < load.time_us * 0.98:
            continue
        buckets = defaultdict(list)
        options_cache = {}
        for ti, transfer in enumerate(baseline["rounds"][ri]["transfers"]):
            if time.monotonic() >= end:
                break
            old = tuple(transfer.get("path", transfer.get("route")))
            count = len(transfer["chunk_ranges"])
            old_edges, old_resources, overhead, per_byte = costs.describe(old)
            old_path_time = overhead + count * chunk_bytes * per_byte
            touches = (bool(set(old_edges) & critical_edges) or bool(set(old_resources) & critical_resources))
            if not touches and old_path_time < load.time_us * 0.98:
                continue
            pair = transfer["src"], transfer["dst"], old
            if pair not in options_cache:
                options_cache[pair] = _path_options(graph, transfer, load, costs, outgoing, incoming, end,
                                                   (critical_edges, critical_resources))
            useful = []
            for path in options_cache[pair]:
                edges, resources, new_overhead, new_per_byte = costs.describe(path)
                avoids = (any(edge not in edges for edge in set(old_edges) & critical_edges)
                          or any(resources[name] < old_resources[name] for name in set(old_resources) & critical_resources))
                new_path_time = new_overhead + count * chunk_bytes * new_per_byte
                if not avoids and not (old_path_time >= load.time_us * 0.98 and new_path_time < old_path_time):
                    continue
                # Rank alternatives by their existing congestion and marginal
                # load, without determining how many chunks should be moved.
                busy = max([1e6 * (load.edge_bytes[edge] + chunk_bytes) / graph.edges[edge] for edge in edges]
                           + [1e6 * (load.resource_bytes[name] + chunk_bytes * multiple) / graph.resources[name]
                              for name, multiple in resources.items()] + [0.0])
                useful.append((busy, new_path_time, path))
            if not useful:
                continue
            useful.sort()
            # Distinct, fixed subsets make allocation groups nonoverlapping.
            # Counts are deliberately left for the model, not optimized here.
            path = useful[ti % min(per_round, len(useful))][2]
            buckets[_signature(graph, costs, old, path)].append(EditMember(ti, count, path))
        ranked = sorted(buckets.items(), key=lambda item: -sum(m.chunk_count for m in item[1]))
        chosen = ranked[:min(per_round, max_groups - len(catalog))]
        for signature, members in chosen:
            catalog.append((ri, signature, tuple(members)))
        rounds_used += bool(chosen)

    groups, group_prompts, round_prompts = [], [], {}
    aliases = {}

    def constraint(ri, kind, key):
        identity = ri, kind, key
        if identity not in aliases:
            alias = f"c{len(aliases)}"
            aliases[identity] = alias
            cap = graph.resources[key] if kind == "resource" else graph.edges[key]
            count = loads[ri].resource_bytes[key] if kind == "resource" else loads[ri].edge_bytes[key]
            name = key if kind == "resource" else list(key)
            round_prompts[ri]["constraints"].append({"id": alias, "kind": kind, "name": name,
                                                     "capacity_Bps": cap, "load_bytes": count,
                                                     "busy_us": round(1e6 * count / cap, 6)})
        return aliases[identity]

    for ri, signature, members in catalog:
        if ri not in round_prompts:
            load = loads[ri]
            round_prompts[ri] = {"round": ri + 1, "time_us": load.time_us,
                                 "path_max_us": load.path_max, "overhead_us": load.overhead_max,
                                 "constraints": []}
            # A zero-delta bottleneck can cap the benefit of all listed edits.
            edge_peaks, resource_peaks = load.bottlenecks()
            for name in sorted(resource_peaks):
                constraint(ri, "resource", name)
            for edge in sorted(edge_peaks)[:4]:
                constraint(ri, "edge", edge)
        group = EditGroup(f"g{len(groups)}", ri, members, sum(m.chunk_count for m in members))
        groups.append(group)
        effects = [[constraint(ri, "resource", key), multiple * chunk_bytes] for key, multiple in signature[0]]
        effects.extend([constraint(ri, "edge", key), multiple * chunk_bytes] for key, multiple in signature[1])
        first = baseline["rounds"][ri]["transfers"][members[0].transfer_index]
        path_bound = edge_bound = 0.0
        minimum_capacity = math.inf
        added_edge_bytes = Counter()
        for member in members:
            old_transfer = baseline["rounds"][ri]["transfers"][member.transfer_index]
            old_path = old_transfer.get("path", old_transfer.get("route"))
            old_edges, old_resources, _, _ = costs.describe(old_path)
            edges, resources, overhead, per_byte = costs.describe(member.path)
            path_bound = max(path_bound, overhead + member.chunk_count * chunk_bytes * per_byte)
            for resource in set(old_resources) | set(resources):
                # Include every affected shared capacity, including constraints
                # whose multiplicity is unchanged on the alternate path.
                constraint(ri, "resource", resource)
            for edge in set(old_edges) | set(edges):
                minimum_capacity = min(minimum_capacity, graph.edges[edge])
            for edge in edges:
                if edge not in old_edges:
                    added_edge_bytes[edge] += member.chunk_count * chunk_bytes
        for edge, extra in added_edge_bytes.items():
            edge_bound = max(edge_bound, 1e6 * (loads[ri].edge_bytes[edge] + extra) / graph.edges[edge])
        group_prompts.append({"id": group.id, "round": ri + 1, "max_chunks": group.max_chunks,
                              "member_count": len(members), "effects": effects,
                              "path_sample": {"old": list(first.get("path", first.get("route"))), "new": list(members[0].path)},
                              "full_move_path_max_us": round(path_bound, 6),
                              "full_move_added_edge_busy_upper_us": round(edge_bound, 6),
                              "affected_edge_min_capacity_Bps": minimum_capacity})
    problem = {"base": base, "operator": task["operator"], "chunk_bytes": chunk_bytes,
               "before_us": report["total_time_us"],
               "rounds": [round_prompts[index] for index in sorted(round_prompts)], "groups": group_prompts}
    if not groups:
        problem["topology"] = compact_summary(graph)
    context = EditContext(input_hash, baseline_hash, base, report["total_time_us"],
                          _round_times(baseline, report), tuple(groups), _digest(problem))
    return problem, context


def apply_proposal(task: dict, graph, baseline: dict, context: EditContext, proposal) -> tuple[dict, dict]:
    """Keep exact model edits only in independently improving rounds.

    Full-proposal validity is mandatory. Because paths and packet partitioning
    preserve each round's data state, restoring other rounds cannot introduce a
    dependency violation. A filtered schedule still receives its own replay.
    """
    started = time.monotonic()
    evidence = {"accepted": False, "input_sha256": context.input_sha256,
                "baseline_sha256": context.baseline_sha256, "base": context.base,
                "before_us": context.before_us, "after_us": context.before_us,
                "requested_edits": proposal, "attempted_edits": [], "applied_edits": [],
                "returned_us": context.before_us, "proposal_time_us": None,
                "accepted_rounds": [], "rejected_rounds": [], "frozen_transfers": [],
                "edit_manifest": [], "changed_transfers": 0, "errors": []}

    def reject(reason):
        evidence["errors"].append(reason)
        evidence["elapsed_s"] = round(time.monotonic() - started, 6)
        return baseline, evidence

    try:
        if _input_digest(task, graph) != context.input_sha256:
            return reject("stale input: task or topology differs from the edit context")
        if _digest(baseline) != context.baseline_sha256:
            return reject("stale baseline: schedule differs from the edit context")
        if not isinstance(proposal, dict) or set(proposal) != {"base", "edits"}:
            return reject("proposal must contain exactly base and edits")
        if proposal["base"] != context.base:
            return reject("proposal base does not match the current input and baseline")
        edits = proposal["edits"]
        if not isinstance(edits, list) or len(edits) > len(context.groups):
            return reject("edits must be an array bounded by the number of groups")
        groups = {group.id: group for group in context.groups}
        selected, seen = [], set()
        for edit in edits:
            if not isinstance(edit, list) or len(edit) != 2:
                return reject("each edit must be [group_id, move_chunks]")
            name, amount = edit
            if not isinstance(name, str) or name not in groups or name in seen:
                return reject("unknown or repeated group_id")
            seen.add(name)
            if type(amount) is not int or not 0 <= amount <= groups[name].max_chunks:
                return reject("move_chunks must be an integer within the group limit")
            if amount:
                selected.append((groups[name], amount))
        if not selected:
            return reject("proposal moves no chunks")
        changes, touched, change_audit = {}, set(), []
        for group, amount in selected:
            remaining = amount
            for member in group.members:
                if not remaining:
                    break
                key = group.round_index, member.transfer_index
                if key in touched:
                    return reject("edit groups overlap")
                touched.add(key)
                old = baseline["rounds"][group.round_index]["transfers"][member.transfer_index]
                if not _legal_path(graph, old["src"], old["dst"], member.path):
                    return reject("edit context contains an illegal path")
                count = min(remaining, member.chunk_count)
                tokens = old["chunk_ranges"]
                moved = {**old, "chunk_ranges": tokens[:count], "path": list(member.path)}
                replacements = [moved]
                if count < len(tokens):
                    replacements.append({**old, "chunk_ranges": tokens[count:]})
                changes[key] = replacements
                change_audit.append({"round": group.round_index + 1, "transfer_index": member.transfer_index,
                                     "src": old["src"], "dst": old["dst"], "action": old["action"],
                                     "moved_chunks": count, "moved_chunk_sha256": _digest(tokens[:count]),
                                     "before_path": list(old.get("path", old.get("route"))),
                                     "after_path": list(member.path)})
                remaining -= count
            evidence["attempted_edits"].append([group.id, amount])
        rounds, frozen_transfers = [], []
        for ri, round_ in enumerate(baseline["rounds"]):
            transfers = []
            for ti, transfer in enumerate(round_["transfers"]):
                if (ri, ti) in changes:
                    # Only the moved prefix is fixed in later routing. The
                    # unmoved remainder retains its own optimization freedom.
                    frozen_transfers.append([ri, len(transfers)])
                transfers.extend(changes.get((ri, ti), [transfer]))
            rounds.append({**round_, "transfers": transfers})
        candidate = {**baseline, "rounds": rounds}
        checked = replay(task, graph, candidate)
        evidence["replay_valid"] = checked["valid"]
        evidence.update(attempted_changed_transfers=len(change_audit),
                        attempted_transfer_edits_sha256=_digest(change_audit))
        if not checked["valid"]:
            return reject("edited schedule failed replay: " + "; ".join(checked["errors"]))
        evidence["proposal_time_us"] = checked["total_time_us"]
        proposed_times = _round_times(candidate, checked)
        kept = set()
        for ri in sorted({group.round_index for group, _ in selected}):
            before, after = context.before_round_times_us[ri], proposed_times[ri]
            if after < before - max(1e-6, before * 1e-10):
                kept.add(ri)
            else:
                evidence["rejected_rounds"].append({
                    "round": ri + 1, "before_us": before, "proposal_us": after,
                    "reason": "model edits do not strictly improve this round; original round retained"})
        if not kept:
            return reject("edited schedule has no strictly improving round")
        if evidence["rejected_rounds"]:
            candidate = {**baseline, "rounds": [round_ if ri in kept else baseline["rounds"][ri]
                                                for ri, round_ in enumerate(rounds)]}
            checked = replay(task, graph, candidate)
            evidence["replay_valid"] = checked["valid"]
            if not checked["valid"]:
                return reject("filtered edited schedule failed replay: " + "; ".join(checked["errors"]))
        if checked["total_time_us"] >= context.before_us - max(1e-6, context.before_us * 1e-10):
            return reject("edited schedule does not strictly improve communication time")
        kept_audit = [row for row in change_audit if row["round"] - 1 in kept]
        candidate["metrics"] = {**baseline.get("metrics", {}), "total_time": checked["total_time_us"] / 1e6,
                                "round_times": [value / 1e6 for value in checked["round_times_us"]]}
        evidence.update(accepted=True, improvement_us=context.before_us - checked["total_time_us"],
                        accepted_rounds=[ri + 1 for ri in sorted(kept)],
                        frozen_transfers=[pair for pair in frozen_transfers if pair[0] in kept],
                        applied_edits=[[group.id, amount] for group, amount in selected if group.round_index in kept],
                        changed_transfers=len(kept_audit), transfer_edits_sha256=_digest(kept_audit),
                        transfer_edit_samples=kept_audit[:8],
                        edit_manifest=[[row["round"], row["transfer_index"], row["moved_chunks"],
                                        row["before_path"], row["after_path"]] for row in kept_audit],
                        after_us=checked["total_time_us"], returned_us=checked["total_time_us"],
                        candidate_sha256=_digest(candidate), round_times_us=checked["round_times_us"],
                        elapsed_s=round(time.monotonic() - started, 6))
        return candidate, evidence
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        return reject("invalid proposal or context: " + str(exc))
