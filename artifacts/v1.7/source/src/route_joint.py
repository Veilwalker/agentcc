"""Bounded whole-atom multipath proposals; the caller performs final Replay.

Only rounds with a dominating packet path time and existing alternative routes
are considered. Two batches of each original packet may use different paths;
batches using the same path are merged again. A smooth load objective permits
joint moves across tied bottlenecks, but exact round costs decide retention.
"""

from collections import defaultdict
import math
import time

from .route_balance import _PathCosts, _RoundLoad
from .schedule_edit import _validated_report


def _power8(value):
    """Finite monotone surrogate even for an extremely small legal capacity."""
    if not math.isfinite(value):
        return 1e96
    value = min(max(value, 0.0), 1e12)
    return value ** 8


def _load_power(amount, factor):
    if amount == 0:
        return 0.0
    try:
        value = amount * factor
    except OverflowError:
        value = math.inf
    return _power8(value)


class _Work:
    def __init__(self, updates, scores, deadline):
        self.max_updates, self.max_scores, self.deadline = updates, scores, deadline
        self.updates = self.scores = self.bounds = self.transactions = 0
        self.stop_reason = None

    def may_start(self, updates=0, scores=0):
        if self.stop_reason:
            return False
        if self.updates + updates > self.max_updates:
            self.stop_reason = "batch_update_limit"
        elif self.scores + self.bounds + scores > self.max_scores:
            self.stop_reason = "path_evaluation_limit"
        elif time.perf_counter() >= self.deadline:
            self.stop_reason = "wall_safety"
        return self.stop_reason is None

    def check_bound(self):
        if not self.may_start(scores=1):
            return False
        self.bounds += 1
        return True


def _path_time(costs, path, count):
    try:
        _, _, overhead, per_byte = costs.describe(path)
        value = overhead + count * per_byte
        return value if math.isfinite(value) else math.inf
    except (OverflowError, ZeroDivisionError):
        return math.inf


class _Assignment:
    def __init__(self, transfers, costs, chunk_bytes, eligible, scale):
        self.rows, self.costs, self.scale = transfers, costs, scale
        self.paths, self.sizes = [], []
        self.edges, self.resources = defaultdict(int), defaultdict(int)
        for ti, row in enumerate(transfers):
            path, count = tuple(row["path"]), len(row["chunk_ranges"])
            counts = [(count + 1) // 2, count // 2] if ti in eligible else [count]
            self.paths.append([path] * len(counts))
            self.sizes.append([n * chunk_bytes for n in counts])
            self.add(path, count * chunk_bytes)
        graph = costs.graph
        self.edge_factor, self.resource_factor = {}, {}
        self.graph = graph

    def add(self, path, count):
        edges, resources, _, _ = self.costs.describe(path)
        for edge in edges:
            self.edges[edge] += count
        for name, multiple in resources.items():
            self.resources[name] += count * multiple

    def score(self, path, count, already=0):
        graph = self.graph
        edges, resources, _, _ = self.costs.describe(path)
        value = 0.0
        for edge in edges:
            if edge not in self.edge_factor:
                self.edge_factor[edge] = (1e6 / self.scale) / graph.edges[edge]
            factor = self.edge_factor[edge]
            value += _load_power(self.edges[edge] + count, factor) - _load_power(self.edges[edge], factor)
        for name, multiple in resources.items():
            if name not in self.resource_factor:
                self.resource_factor[name] = (1e6 / self.scale) / graph.resources[name]
            factor = self.resource_factor[name]
            value += (_load_power(self.resources[name] + count * multiple, factor)
                      - _load_power(self.resources[name], factor))
        # The other batch may already use this path. Score the merged packet,
        # otherwise keeping both batches on a fast path would receive a false
        # reduction in its serial-path term before any real route change.
        duration = _path_time(self.costs, path, already + count)
        value += _power8(duration / self.scale)
        if already:
            value -= _power8(_path_time(self.costs, path, already) / self.scale)
        return value, duration, path

    def exact_cost(self):
        load = _RoundLoad(self.costs)
        try:
            for paths, sizes in zip(self.paths, self.sizes):
                grouped = defaultdict(int)
                for path, count in zip(paths, sizes):
                    grouped[path] += count
                for path, count in grouped.items():
                    load.add(path, count)
            result = load.time_us
            return result if math.isfinite(result) else math.inf
        except (OverflowError, ZeroDivisionError):
            return math.inf

    def compile(self, chosen):
        result = []
        for row, paths in zip(self.rows, chosen):
            if len(set(paths)) == 1:
                result.append(row if tuple(row["path"]) == paths[0] else {**row, "path": list(paths[0])})
                continue
            # The two batches are prefixes of the original list; an entire
            # AllReduce contribution expression remains one indivisible item.
            middle = (len(row["chunk_ranges"]) + 1) // 2
            for path, tokens in zip(paths, (row["chunk_ranges"][:middle], row["chunk_ranges"][middle:])):
                result.append({**row, "path": list(path), "chunk_ranges": tokens})
        return result


def propose_joint(task, graph, solution, validated_report, *, max_rounds=2,
                  max_batch_updates=6000, max_path_scores=40000, max_sweeps=2,
                  max_extra_packets=1024, min_relative_gain=0.002,
                  path_dominance=0.85, wall_safety_s=2.0):
    """Return an unvalidated proposal or the identical input on no improvement.

    ``validated_report`` must be the actual Replay report for this unchanged
    solution. This internal trust contract is shape-checked, not authenticated.
    No Replay is performed here. Exact-cost estimates and a proposal flag must
    not be presented as validation; the caller replays the final composed answer.

    The path budget counts both serial path-bound checks and insertion scores.
    A complete original-packet transaction reserves both batch updates and all
    candidate scores before starting. Setup and final compilation still finish
    outside these counters; wall time is only an exceptional safety bound.
    """
    for name, value in (("max_rounds", max_rounds), ("max_batch_updates", max_batch_updates),
                        ("max_path_scores", max_path_scores), ("max_sweeps", max_sweeps),
                        ("max_extra_packets", max_extra_packets)):
        if type(value) is not int or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    for name, value in (("min_relative_gain", min_relative_gain), ("path_dominance", path_dominance),
                        ("wall_safety_s", wall_safety_s)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(name + " must be finite and nonnegative")
    if path_dominance > 1 or min_relative_gain >= 1:
        raise ValueError("relative thresholds must be below or equal to one")
    start = time.perf_counter()
    report = _validated_report(solution, validated_report)
    before = report["total_time_us"]
    evidence = {"proposed": False, "requires_replay": True, "before_us": before,
                "candidate_time_us": before, "gain_relative": 0.0, "extra_packets": 0,
                "rounds": [], "baseline_report_reused": True}
    work = _Work(max_batch_updates, max_path_scores, start + wall_safety_s)
    costs = _PathCosts(graph)
    chunk_bytes = task["data_size"]["per_rank_bytes"] // solution["split"]
    times = iter(report["round_times_us"])
    original_times = [next(times) if row["transfers"] else 0.0 for row in solution["rounds"]]
    resulting_times, changed_rounds = list(original_times), list(solution["rounds"])
    considered = eligible_rounds = added_packets = 0
    alternatives, filtered = {}, {}

    def options(row, round_time):
        pair, original = (row["src"], row["dst"]), tuple(row["path"])
        if pair not in alternatives:
            choices = [pair] if pair in graph.edges else []
            choices.extend((pair[0], switch, pair[1]) for switch, _, _ in graph.adjacency[pair[0]]
                           if switch in graph.switches and (switch, pair[1]) in graph.edges)
            alternatives[pair] = tuple(sorted(choices))
        minimum_batch = len(row["chunk_ranges"]) // 2 * chunk_bytes
        key = pair, minimum_batch, round_time
        if key not in filtered:
            valid = []
            for path in alternatives[pair]:
                if not work.check_bound():
                    break
                # Infinite serial cost cannot improve this finite round.
                if _path_time(costs, path, minimum_batch) <= round_time:
                    valid.append(path)
            filtered[key] = tuple(valid)
        return tuple(sorted(set(filtered[key]) | {original}))

    selected = sorted(range(len(original_times)), key=lambda ri: (-original_times[ri], ri))[:max_rounds]
    for ri in selected:
        if (not before or not max_extra_packets or not max_sweeps or not max_batch_updates
                or not max_path_scores or not work.may_start()):
            break
        rows, round_time = solution["rounds"][ri]["transfers"], original_times[ri]
        if not rows or not round_time:
            continue
        considered += 1
        # Cheap structural trigger: inspect packet lengths and their own path
        # costs, without expanding any atomic expressions or scanning all edges.
        dominant = False
        for row in rows:
            count = len(row["chunk_ranges"])
            if count < 2:
                continue
            if not work.check_bound():
                break
            if _path_time(costs, row["path"], count * chunk_bytes) >= path_dominance * round_time:
                if len(options(row, round_time)) > 1:
                    dominant = True
                    break
        if not dominant or not work.may_start():
            evidence["rounds"].append({"round": ri + 1, "proposed": False,
                                       "reason": "no_dominating_splittable_path_with_alternatives"})
            continue
        eligible = {}
        for ti, row in enumerate(rows):
            if len(row["chunk_ranges"]) < 2 or not work.may_start():
                continue
            paths = options(row, round_time)
            if len(paths) > 1:
                eligible[ti] = paths
        if not eligible or not work.may_start():
            continue
        eligible_rounds += 1
        state = _Assignment(rows, costs, chunk_bytes, eligible, round_time)
        actual_before = state.exact_cost()
        if not math.isclose(actual_before, round_time, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("joint route baseline cost disagrees with its trusted report")
        best_cost, best_paths, best_extra = actual_before, None, 0
        local_extra = 0
        rounds_log = []
        for sweep in range(max_sweeps):
            changes = 0
            order = sorted(eligible, key=lambda ti: (-sum(state.sizes[ti]), ti))
            if sweep % 2:
                order.reverse()
            for ti in order:
                paths = eligible[ti]
                scores_needed = 2 * len(paths)
                if not work.may_start(updates=2, scores=scores_needed):
                    break
                previous = list(state.paths[ti])
                old_extra = len(set(previous)) - 1
                for batch, count in enumerate(state.sizes[ti]):
                    old = state.paths[ti][batch]
                    state.add(old, -count)
                    other = 1 - batch
                    chosen = min(paths, key=lambda path: state.score(
                        path, count, state.sizes[ti][other] if state.paths[ti][other] == path else 0))
                    state.paths[ti][batch] = chosen
                    state.add(chosen, count)
                new_extra = len(set(state.paths[ti])) - 1
                if added_packets + local_extra - old_extra + new_extra > max_extra_packets:
                    for batch, count in enumerate(state.sizes[ti]):
                        state.add(state.paths[ti][batch], -count)
                        state.paths[ti][batch] = previous[batch]
                        state.add(previous[batch], count)
                    new_extra = old_extra
                local_extra += new_extra - old_extra
                work.updates += 2
                work.scores += scores_needed
                work.transactions += 1
                changes += state.paths[ti] != previous
            exact = state.exact_cost()
            rounds_log.append({"sweep": sweep + 1, "changed_original_packets": changes,
                               "time_us": exact if math.isfinite(exact) else None})
            if math.isfinite(exact) and exact < best_cost - max(1e-6, best_cost * 1e-10):
                best_cost, best_paths, best_extra = exact, [list(paths) for paths in state.paths], local_extra
            if changes == 0 or work.stop_reason:
                break
        if best_paths is not None:
            changed_rounds[ri] = {**solution["rounds"][ri], "transfers": state.compile(best_paths)}
            resulting_times[ri] = best_cost
            added_packets += best_extra
        evidence["rounds"].append({"round": ri + 1, "proposed": best_paths is not None,
                                   "before_us": actual_before, "after_us": best_cost,
                                   "extra_packets": best_extra, "sweeps": rounds_log})
    total = sum(resulting_times)
    gain = (before - total) / before if before else 0.0
    proposal = solution
    if gain >= min_relative_gain and total < before - max(1e-6, before * 1e-10):
        proposal = {**solution, "rounds": changed_rounds,
                    "metrics": {**solution.get("metrics", {}), "total_time": total / 1e6,
                                "round_times": [value / 1e6 for value, row in zip(resulting_times, changed_rounds) if row["transfers"]]}}
        evidence.update(proposed=True, candidate_time_us=total, gain_relative=gain, extra_packets=added_packets)
    else:
        evidence["reason"] = "below_minimum_gain" if gain > 0 else "no_improvement"
        evidence["attempted_gain_relative"] = gain
    evidence.update(considered_rounds=considered, eligible_rounds=eligible_rounds,
        work={"batch_updates": work.updates, "path_scores": work.scores, "path_bound_checks": work.bounds,
              "path_evaluations": work.scores + work.bounds, "packet_transactions": work.transactions,
              "stop_reason": work.stop_reason or "completed", "limits": {"batch_updates": max_batch_updates,
              "path_evaluations": max_path_scores, "extra_packets": max_extra_packets}},
        elapsed_s=time.perf_counter() - start)
    return proposal, evidence
