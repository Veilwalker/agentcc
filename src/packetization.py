"""Bounded same-path packetization of an already validated schedule.

Whole AllGather chunks and whole AllReduce contribution expressions stay in
their original round, action, endpoints and path. Only transfer grouping changes.
The caller may supply the Replay report of this exact baseline. An internal
route proposal without a report is costed directly, never labeled validated.
The caller must Replay a changed candidate before accepting it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .route_balance import _PathCosts, _RoundLoad


# Engineering limits on optional response growth, not competition constraints.
MAX_EXTRA_PACKETS = 8192
MAX_OUTPUT_TRANSFERS = 65536


def _tolerance(value):
    return max(1e-6, abs(value) * 1e-10)


def _finite_nonnegative(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _round_times(solution, report):
    """Check report shape; its binding to the baseline is trusted by contract."""
    if not isinstance(report, dict) or report.get("valid") is not True:
        raise ValueError("packetization requires a valid baseline Replay report")
    values = report.get("round_times_us")
    rounds = solution["rounds"]
    nonempty = sum(bool(row["transfers"]) for row in rounds)
    if (not isinstance(values, list) or len(values) != nonempty
            or any(not _finite_nonnegative(value) for value in values)):
        raise ValueError("baseline Replay must contain finite matching round times")
    total = report.get("total_time_us")
    summed = sum(values)
    if (not _finite_nonnegative(total) or not _finite_nonnegative(summed)
            or not math.isclose(summed, total, rel_tol=1e-9, abs_tol=1e-6)):
        raise ValueError("baseline Replay total disagrees with its round times")
    iterator = iter(values)
    return [next(iterator) if row["transfers"] else 0.0 for row in rounds]


@dataclass(frozen=True)
class _Shape:
    atoms: int
    overhead_us: float
    atom_path_us: float


@dataclass(frozen=True)
class _Plan:
    sizes: tuple[int, ...]
    extra: int
    before: float
    after: float
    target: float
    resource_floor: float
    atomic_floor: float

    @property
    def gain(self):
        return self.before - self.after


def _plan(shapes, target, before, resource_floor, atomic_floor):
    sizes, extra, longest = [], 0, 0.0
    for shape in shapes:
        n, overhead, per_atom = shape.atoms, shape.overhead_us, shape.atom_path_us
        if overhead + n * per_atom <= target:
            maximum = n
        else:
            maximum = max(1, min(n, math.floor((target - overhead) / per_atom + 1e-10)))
        packets = (n + maximum - 1) // maximum
        # ceil(n/maximum) is the minimum packet count reaching this threshold.
        # Slicing by ceil(n/packets) produces exactly that many packets.
        size = (n + packets - 1) // packets
        sizes.append(size)
        extra += packets - 1
        longest = max(longest, overhead + size * per_atom)
    return _Plan(tuple(sizes), extra, before, max(resource_floor, longest), target,
                 resource_floor, atomic_floor)


def repacketize(task, graph, solution, baseline_report, *, max_extra_packets=2048,
                min_relative_gain=0.002):
    """Return ``(candidate, prediction)`` without performing validation Replay.

    A non-None ``baseline_report`` must come from this exact solution. None is
    used for internal route proposals; all resource/path costs are recomputed.
    A changed candidate retains all complete atomic expressions; stored transfer
    or atom indices may change and must be remapped by the caller's evidence code.
    The total communication gain threshold gates optional work; it is not an
    estimate of official score. Rejected proposals return the original object.
    """
    if type(max_extra_packets) is not int or not 0 <= max_extra_packets <= MAX_EXTRA_PACKETS:
        raise ValueError(f"max_extra_packets must be an integer in 0..{MAX_EXTRA_PACKETS}")
    if (not _finite_nonnegative(min_relative_gain) or min_relative_gain > 1):
        raise ValueError("min_relative_gain must be finite and in 0..1")
    split = solution.get("split")
    data_bytes = task.get("data_size", {}).get("per_rank_bytes")
    if (type(split) is not int or split <= 0 or type(data_bytes) is not int or data_bytes <= 0
            or data_bytes % split):
        raise ValueError("packetization requires an integral positive chunk size")
    chunk_bytes = data_bytes // split
    try:
        finite_chunk = math.isfinite(float(chunk_bytes))
    except OverflowError:
        finite_chunk = False
    if not finite_chunk:
        raise ValueError("chunk size must be representable as a finite value")

    costs = None
    if baseline_report is None:
        costs = _PathCosts(graph)
        times = []
        for row in solution["rounds"]:
            load = _RoundLoad(costs)
            for transfer in row["transfers"]:
                load.add(transfer["path"], len(transfer["chunk_ranges"]) * chunk_bytes)
            if not math.isfinite(load.time_us):
                raise ValueError("internal proposal has non-finite modeled cost")
            times.append(load.time_us)
    else:
        times = _round_times(solution, baseline_report)
    before = sum(times)
    before_count = sum(len(row["transfers"]) for row in solution["rounds"])
    budget = min(max_extra_packets, max(0, MAX_OUTPUT_TRANSFERS - before_count))
    evidence = {
        "applied": False, "replay_performed": False, "requires_final_replay": False,
        "baseline_cost_source": "replay" if baseline_report is not None else "path_resource_model",
        "before_us": before, "predicted_after_us": before, "predicted_gain_us": 0.0,
        "predicted_relative_gain": 0.0, "min_relative_gain": min_relative_gain,
        "max_extra_packets": max_extra_packets, "effective_extra_budget": budget,
        "before_transfer_count": before_count, "after_transfer_count": before_count,
        "actual_extra_packets": 0, "rounds_load_evaluated": 0,
        "rounds_skipped_by_replay": 0, "plans": [],
    }

    def unchanged(reason):
        evidence["reason"] = reason
        return solution, evidence

    if budget == 0:
        return unchanged("zero_budget" if max_extra_packets == 0 else "output_size_limit")
    if not any(len(t["chunk_ranges"]) > 1 for row in solution["rounds"] for t in row["transfers"]):
        return unchanged("no_splittable_transfer")

    costs = costs or _PathCosts(graph)
    frontiers = {}
    for ri, row in enumerate(solution["rounds"]):
        transfers = row["transfers"]
        if not transfers or all(len(t["chunk_ranges"]) == 1 for t in transfers):
            evidence["rounds_skipped_by_replay"] += 1
            continue
        shapes = []
        for transfer in transfers:
            n = len(transfer["chunk_ranges"])
            if n == 0:
                raise ValueError("validated baseline cannot contain empty transfers")
            _, _, overhead, per_byte = costs.describe(transfer["path"])
            per_atom = per_byte * chunk_bytes
            if (not _finite_nonnegative(overhead) or not math.isfinite(per_atom) or per_atom <= 0
                    or not math.isfinite(overhead + n * per_atom)):
                raise ValueError("path costs must be finite with positive transmission time")
            shapes.append(_Shape(n, overhead, per_atom))
        longest_splittable = max((s.overhead_us + s.atoms * s.atom_path_us
                                 for s in shapes if s.atoms > 1), default=0.0)
        if not transfers or longest_splittable < times[ri] - _tolerance(times[ri]):
            evidence["rounds_skipped_by_replay"] += 1
            continue
        load = _RoundLoad(costs)
        for transfer, shape in zip(transfers, shapes):
            load.add(transfer["path"], shape.atoms * chunk_bytes)
        evidence["rounds_load_evaluated"] += 1
        if (not math.isfinite(load.time_us)
                or not math.isclose(load.time_us, times[ri], rel_tol=1e-9, abs_tol=1e-6)):
            raise ValueError("baseline Replay differs from current path/resource costs")
        resource_floor = load.overhead_max + load.busy_max
        atomic_floor = max(s.overhead_us + s.atom_path_us for s in shapes)
        floor = max(resource_floor, atomic_floor)
        if times[ri] <= floor + _tolerance(times[ri]):
            continue
        plans, seen = [], set()
        for fraction in (0.25, 0.5, 0.75, 1.0):
            target = times[ri] - (times[ri] - floor) * fraction
            plan = _plan(shapes, target, times[ri], resource_floor, atomic_floor)
            if plan.extra and plan.gain > _tolerance(times[ri]) and plan.sizes not in seen:
                seen.add(plan.sizes)
                plans.append(plan)
        if plans:
            frontiers[ri] = plans

    # Four targets per round bound the optional frontier. This greedy allocation
    # is deterministic, but is not claimed optimal under the global packet cap.
    selected, used = {}, 0
    while True:
        choices = []
        for ri, plans in frontiers.items():
            previous = selected.get(ri)
            for plan in plans:
                extra = plan.extra - (previous.extra if previous else 0)
                gain = plan.gain - (previous.gain if previous else 0.0)
                if gain > _tolerance(times[ri]) and extra >= 0 and used + extra <= budget:
                    choices.append((-gain / max(1, extra), -gain, extra, ri, plan))
        if not choices:
            break
        _, _, extra, ri, plan = min(choices, key=lambda item: item[:4])
        selected[ri] = plan
        used += extra
    if not selected:
        return unchanged("no_profitable_plan_within_budget")

    predicted_times = times.copy()
    for ri, plan in selected.items():
        predicted_times[ri] = plan.after
    predicted_after = sum(predicted_times)
    gain = before - predicted_after
    ratio = gain / before if before > 0 else 0.0
    evidence.update(proposed_after_us=predicted_after, proposed_relative_gain=ratio,
                    proposed_extra_packets=used)
    if gain <= _tolerance(before) or ratio < min_relative_gain:
        return unchanged("below_min_relative_gain")

    rounds = list(solution["rounds"])
    for ri, plan in selected.items():
        transfers = []
        for original, size in zip(rounds[ri]["transfers"], plan.sizes):
            chunks = original["chunk_ranges"]
            if size >= len(chunks):
                transfers.append(original)
            else:
                transfers.extend({**original, "chunk_ranges": chunks[start:start + size]}
                                 for start in range(0, len(chunks), size))
        rounds[ri] = {**rounds[ri], "transfers": transfers}
    after_count = sum(len(row["transfers"]) for row in rounds)
    if after_count - before_count != used or used > budget or after_count > MAX_OUTPUT_TRANSFERS:
        raise ValueError("packet budget differs from actual transfer count")
    candidate = {**solution, "rounds": rounds,
                 "metrics": {**solution.get("metrics", {}), "total_time": predicted_after / 1e6,
                             "round_times": [value / 1e6 for value, row in zip(predicted_times, rounds)
                                             if row["transfers"]]}}
    evidence.update(
        applied=True, reason="candidate_requires_replay", requires_final_replay=True,
        predicted_after_us=predicted_after, predicted_gain_us=gain, predicted_relative_gain=ratio,
        after_transfer_count=after_count, actual_extra_packets=used,
        plans=[{"round_index": ri, "before_us": plan.before, "predicted_after_us": plan.after,
                "target_us": plan.target, "resource_floor_us": plan.resource_floor,
                "atomic_floor_us": plan.atomic_floor, "extra_packets": plan.extra}
               for ri, plan in sorted(selected.items())],
    )
    return candidate, evidence
