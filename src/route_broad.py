"""Bounded route refinement that preserves explicitly fixed model actions.

Enumerate legal direct and common-switch paths, with limited bottleneck-avoiding
search for longer fabrics. Greedy assignment includes all fixed traffic first.
Only strictly improved rounds survive, and the returned schedule is replayed.
"""

from __future__ import annotations

from collections import defaultdict
import math
import time

from .replay import certify as replay
from .route_balance import _PathCosts, _RoundLoad
from .route_work import RouteWorkBudget, RouteWorkMeter
from .schedule_edit import _avoiding_path, _legal_path


def _fixed_indices(solution, frozen):
    if not isinstance(frozen, (tuple, list, set, frozenset)):
        raise ValueError("frozen must be a collection of [round_index, transfer_index] pairs")
    fixed = set()
    for pair in frozen:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(type(index) is not int for index in pair):
            raise ValueError("frozen indices must be pairs of integers, not booleans")
        ri, ti = pair
        if not 0 <= ri < len(solution["rounds"]) or not 0 <= ti < len(solution["rounds"][ri]["transfers"]):
            raise ValueError("frozen index is outside the schedule")
        if (ri, ti) in fixed:
            raise ValueError("frozen contains a duplicate transfer index")
        fixed.add((ri, ti))
    return frozenset(fixed)


def refine_broad(task: dict, graph, solution: dict, *, time_budget_s: float = 0.25,
                 frozen=frozenset(), allow_blocked_search: bool = True,
                 work_budget: RouteWorkBudget | None = None) -> tuple[dict, dict]:
    """Return a replayed schedule and evidence, including ``final_replay``.

    Frozen pairs use zero-based indices into this exact input solution. They
    preserve the entire transfer, including chunk boundaries and its path.
    The routing budget reserves a fraction for validation; final replay always
    finishes, so total elapsed time may exceed the nominal routing budget.
    With an explicit work budget, normal stopping uses deterministic route-score
    and edge-relaxation counts; its safety wall bound replaces the optional
    time fraction. An admitted transfer always completes candidate scoring.
    """
    if (type(time_budget_s) not in (int, float) or not math.isfinite(time_budget_s)
            or time_budget_s < 0):
        raise ValueError("routing time budget must be finite and nonnegative")
    if type(allow_blocked_search) is not bool:
        raise ValueError("allow_blocked_search must be boolean")
    if work_budget is not None and not isinstance(work_budget, RouteWorkBudget):
        raise ValueError("work_budget must be a RouteWorkBudget or None")
    started = time.monotonic()
    meter = RouteWorkMeter(work_budget, started) if work_budget is not None else None
    end = meter.deadline if meter is not None else started + time_budget_s * 0.75
    can_optimize = meter.can_start_transfer if meter is not None else lambda: time.monotonic() < end
    fixed = _fixed_indices(solution, frozen)
    if not time_budget_s or len(fixed) == sum(len(row["transfers"]) for row in solution["rounds"]):
        checked = replay(task, graph, solution)
        if not checked["valid"]:
            raise ValueError("route refinement baseline is invalid: " + "; ".join(checked["errors"]))
        return solution, {"accepted": False, "before_us": checked["total_time_us"],
                          "after_us": checked["total_time_us"], "replay_valid": True,
                          "improved_rounds": 0, "changed_transfers": 0,
                          "pairs_with_alternatives": 0, "blocked_searches": 0,
                          "frozen_count": len(fixed), "frozen_preserved": True,
                          "deadline_reached": time.monotonic() >= end,
                          "elapsed_s": round(time.monotonic() - started, 6), "final_replay": checked}
    costs = _PathCosts(graph)
    chunk_bytes = task["data_size"]["per_rank_bytes"] // solution["split"]
    loads, options = [], defaultdict(set)
    outgoing, incoming = defaultdict(set), defaultdict(set)
    for src, dst in graph.edges:
        if src not in graph.switches and dst in graph.switches:
            outgoing[src].add(dst)
        if src in graph.switches and dst not in graph.switches:
            incoming[dst].add(src)
    for round_ in solution["rounds"]:
        load = _RoundLoad(costs)
        for transfer in round_["transfers"]:
            path = tuple(transfer["path"])
            load.add(path, len(transfer["chunk_ranges"]) * chunk_bytes)
            options[transfer["src"], transfer["dst"]].add(path)
        loads.append(load)
    before_us = sum(load.time_us for load in loads)
    initialized, searched = set(), set()
    changed_rounds = list(solution["rounds"])
    changed_count = changed_transfers = blocked_searches = 0
    for ri in sorted(range(len(loads)), key=lambda index: -loads[index].time_us):
        if not can_optimize():
            break
        round_ = solution["rounds"][ri]
        old_load = loads[ri]
        critical_edges, critical_resources = old_load.bottlenecks()
        assigned = _RoundLoad(costs)
        transfers = list(round_["transfers"])
        for ti, transfer in enumerate(transfers):
            if (ri, ti) in fixed:
                assigned.add(transfer["path"], len(transfer["chunk_ranges"]) * chunk_bytes)
        movable = sorted((ti for ti in range(len(transfers)) if (ri, ti) not in fixed),
                         key=lambda index: -len(transfers[index]["chunk_ranges"]))
        changes = 0
        for ti in movable:
            transfer = transfers[ti]
            pair = transfer["src"], transfer["dst"]
            original = tuple(transfer["path"])
            count = len(transfer["chunk_ranges"]) * chunk_bytes
            if can_optimize():
                if pair not in initialized:
                    initialized.add(pair)
                    if pair in graph.edges:
                        options[pair].add(pair)
                    options[pair].update((pair[0], switch, pair[1])
                                         for switch in outgoing[pair[0]] & incoming[pair[1]])
                if allow_blocked_search and pair not in searched and (len(original) > 3 or len(options[pair]) < 2):
                    searched.add(pair)
                    old_edges, old_resources, _, _ = costs.describe(original)
                    excluded_edges = set(old_edges) & critical_edges
                    excluded_resources = set(old_resources) & critical_resources
                    if excluded_edges or excluded_resources:
                        blocked_searches += 1
                        if meter is None:
                            alternate = _avoiding_path(graph, *pair, old_load, excluded_edges,
                                                       excluded_resources, end)
                        else:
                            alternate = _avoiding_path(graph, *pair, old_load, excluded_edges,
                                                       excluded_resources, end,
                                                       consume_relaxations=meter.consume_relaxations)
                        if alternate and _legal_path(graph, *pair, alternate):
                            options[pair].add(alternate)
                paths = sorted(options[pair]) if meter is not None else options[pair]
                path = min(paths, key=lambda value: assigned.score(value, count))
                if meter is not None:
                    meter.finish_transfer(len(paths))
            else:
                path = original
            assigned.add(path, count)
            if path != original:
                transfers[ti] = {**transfer, "path": list(path)}
                changes += 1
        if assigned.time_us < old_load.time_us - 1e-9:
            changed_rounds[ri] = {**round_, "transfers": transfers}
            changed_count += 1
            changed_transfers += changes
    candidate = {**solution, "rounds": changed_rounds} if changed_count else solution
    checked = replay(task, graph, candidate)
    accepted = (candidate is not solution and checked["valid"]
                and checked["total_time_us"] < before_us - max(1e-6, before_us * 1e-10))
    if accepted:
        candidate["metrics"] = {**solution.get("metrics", {}), "total_time": checked["total_time_us"] / 1e6,
                                "round_times": [value / 1e6 for value in checked["round_times_us"]]}
    else:
        if candidate is not solution:
            # Rejected candidate reports cannot be reused as the final report
            # for the preserved original schedule.
            checked = replay(task, graph, solution)
        candidate = solution
        if not checked["valid"]:
            raise ValueError("route refinement baseline is invalid: " + "; ".join(checked["errors"]))
    frozen_preserved = all(candidate["rounds"][ri]["transfers"][ti] == solution["rounds"][ri]["transfers"][ti]
                           for ri, ti in fixed)
    if not frozen_preserved:
        raise ValueError("route refinement changed a frozen model action")
    evidence = {"before_us": before_us, "after_us": checked["total_time_us"],
                "accepted": accepted, "replay_valid": True,
                "improved_rounds": changed_count if accepted else 0,
                "changed_transfers": changed_transfers if accepted else 0,
                "pairs_with_alternatives": sum(len(paths) > 1 for paths in options.values()),
                "blocked_searches": blocked_searches, "frozen_count": len(fixed),
                "frozen_preserved": frozen_preserved, "deadline_reached": time.monotonic() >= end,
                "elapsed_s": round(time.monotonic() - started, 6), "final_replay": checked}
    if meter is not None:
        evidence["work"] = meter.evidence()
    return candidate, evidence
