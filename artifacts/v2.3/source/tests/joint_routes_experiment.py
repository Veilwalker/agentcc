"""Offline joint path assignment on archived, validated collective schedules.

Production code does not import this experiment. Data actions, atomic reduction
expressions and rounds stay fixed. Optional atomic batches permit multipath
splitting; batches from one original packet on the same path are merged again.
Coordinate moves minimize
a smooth congestion potential, so an individual move need not reduce the true
round maximum. Only complete assignment checkpoints with a lower exact round cost are retained;
the final complete schedule must pass Replay and strictly improve the baseline.

This explores deterministic control potential, not LLM or online latency gains.
Run from the project root: python3 -m tests.joint_routes_experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import time

from src.replay import replay
from src.route_balance import _PathCosts, _RoundLoad
from src.schedule_edit import _legal_path
from src.topology import Topology


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "materials/spec/examples/256-rank"
ARCHIVE = ROOT / "artifacts/v1.4/runs/public-narrow/v1.4"
OUTPUT = ROOT / "artifacts/v1.5/analysis/joint-routes"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_actions(round_):
    return Counter((row["action"], row["src"], row["dst"], token)
                   for row in round_["transfers"] for token in row["chunk_ranges"])


@dataclass(frozen=True)
class Limits:
    max_rounds: int = 2
    sweeps: int = 4
    max_paths: int = 8
    priced_searches: int = 512
    search_seconds: float = 3.0
    exponent: int = 8
    batches: int = 1
    max_updates: int = 65536
    method: str = "potential"

    def __post_init__(self):
        for name in ("max_rounds", "sweeps", "max_paths", "exponent", "batches", "max_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if type(self.priced_searches) is not int or self.priced_searches < 0:
            raise ValueError("priced_searches must be a nonnegative integer")
        if (type(self.search_seconds) not in (int, float)
                or not math.isfinite(self.search_seconds) or self.search_seconds < 0):
            raise ValueError("search_seconds must be finite and nonnegative")
        if self.method not in ("potential", "capacity_repair"):
            raise ValueError("unknown joint route experiment method")


class Assignment:
    """Mutable integer loads; candidate scoring uses the full residual traffic."""

    def __init__(self, transfers, costs, chunk_bytes, exponent, parents=None):
        self.transfers, self.costs = transfers, costs
        self.parents = list(range(len(transfers))) if parents is None else parents
        self.paths = [tuple(row["path"]) for row in transfers]
        self.sizes = [len(row["chunk_ranges"]) * chunk_bytes for row in transfers]
        self.edges, self.resources = defaultdict(int), defaultdict(int)
        self.exponent = exponent
        for path, count in zip(self.paths, self.sizes):
            self.add(path, count)
        self.scale = max(self.exact_cost(), 1e-9)

    def add(self, path, count):
        edges, resources, _, _ = self.costs.describe(path)
        for edge in edges:
            self.edges[edge] += count
        for name, multiple in resources.items():
            self.resources[name] += count * multiple

    def exact_cost(self):
        load = _RoundLoad(self.costs)
        grouped = defaultdict(int)
        for parent, path, count in zip(self.parents, self.paths, self.sizes):
            grouped[parent, path] += count
        for (_, path), count in grouped.items():
            load.add(path, count)
        return load.time_us

    def merged_transfers(self, paths):
        """Never create a same-path split solely to lower serial path time."""
        groups = {}
        for parent, row, path in zip(self.parents, self.transfers, paths):
            key = parent, path
            if key not in groups:
                groups[key] = {**row, "path": list(path), "chunk_ranges": []}
            groups[key]["chunk_ranges"].extend(row["chunk_ranges"])
        return list(groups.values())

    def _increment(self, before, extra, capacity):
        factor = 1e6 / (capacity * self.scale)
        return ((before + extra) * factor) ** self.exponent - (before * factor) ** self.exponent

    def score(self, path, count):
        """Potential insertion after removing this transfer's current path.

        Shared resources retain their exact path multiplicity. The path term
        prevents relieving aggregate congestion by creating a very slow path.
        With atomic batching the path term is deliberately a surrogate;
        exact_cost merges same-parent/same-path batches before evaluation.
        This objective is never used as an acceptance or Replay substitute.
        """
        graph = self.costs.graph
        edges, resources, overhead, per_byte = self.costs.describe(path)
        value = sum(self._increment(self.edges[edge], count, graph.edges[edge]) for edge in edges)
        value += sum(self._increment(self.resources[name], count * multiple, graph.resources[name])
                     for name, multiple in resources.items())
        path_time = overhead + count * per_byte
        value += (path_time / self.scale) ** self.exponent
        return value, path_time, tuple(path)

    def priced_path(self, src, dst, count, deadline):
        """One positive-cost legal path under current residual-load prices.

        Dijkstra's per-edge resource price is only a candidate heuristic: a
        resource encountered multiple times is evaluated exactly by score().
        """
        graph = self.costs.graph
        distances, previous, pending = {src: 0.0}, {}, [(0.0, src)]
        expanded = 0
        while pending:
            distance, node = heapq.heappop(pending)
            if distance != distances[node]:
                continue
            if node == dst:
                result = [dst]
                while result[-1] != src:
                    result.append(previous[result[-1]])
                return tuple(reversed(result))
            expanded += 1
            if expanded % 16 == 0 and time.perf_counter() >= deadline:
                return None
            for target, inverse, _ in graph.adjacency[node]:
                if target != dst and target not in graph.switches:
                    continue
                edge = node, target
                price = self._increment(self.edges[edge], count, graph.edges[edge])
                price += sum(self._increment(self.resources[name], count, graph.resources[name])
                             for name in graph.edge_resources[edge])
                # Positive latency discourages detours even below congestion.
                price += 0.02 * (1e6 * count * inverse + graph.switch_overhead_us) / self.scale
                proposed = distance + price + 1e-15
                if proposed < distances.get(target, math.inf):
                    distances[target], previous[target] = proposed, node
                    heapq.heappush(pending, (proposed, target))
        return None

    def potential(self):
        graph = self.costs.graph
        value = sum((1e6 * amount / (graph.edges[edge] * self.scale)) ** self.exponent
                    for edge, amount in self.edges.items())
        value += sum((1e6 * amount / (graph.resources[name] * self.scale)) ** self.exponent
                     for name, amount in self.resources.items())
        for path, count in zip(self.paths, self.sizes):
            _, _, overhead, per_byte = self.costs.describe(path)
            value += ((overhead + count * per_byte) / self.scale) ** self.exponent
        return value


def _round_load(round_, costs, chunk_bytes):
    load = _RoundLoad(costs)
    for transfer in round_["transfers"]:
        load.add(transfer["path"], len(transfer["chunk_ranges"]) * chunk_bytes)
    return load


def _batch_transfers(transfers, batches):
    expanded, parents = [], []
    for ti, row in enumerate(transfers):
        tokens = row["chunk_ranges"]
        size = max(1, (len(tokens) + batches - 1) // batches)
        for first in range(0, len(tokens), size):
            expanded.append({**row, "chunk_ranges": tokens[first:first + size]})
            parents.append(ti)
    return expanded, parents


def repair_to_target(transfers, costs, chunk_bytes, options, target_us, deadline, max_updates):
    """Move minimal integer prefixes off all constraints above a trial target.

    A move cannot worsen any overloaded constraint or create another one.
    No single move needs to improve the whole-round maximum. This bounded
    greedy feasibility attempt has no infeasibility or optimality certificate.
    """
    graph = costs.graph
    capacities = {("edge", edge): cap for edge, cap in graph.edges.items()}
    capacities.update({("resource", name): cap for name, cap in graph.resources.items()})
    descriptions = {}
    def describe(path):
        if path not in descriptions:
            edges, resources, overhead, per_byte = costs.describe(path)
            uses = Counter(("edge", edge) for edge in edges)
            uses.update({("resource", name): multiple for name, multiple in resources.items()})
            descriptions[path] = uses, overhead, per_byte
        return descriptions[path]
    paths = [tuple(row["path"]) for row in transfers]
    alternatives = [sorted(options[row["src"], row["dst"]] | {path}) for row, path in zip(transfers, paths)]
    max_overhead = max((describe(path)[1] for candidates in alternatives for path in candidates), default=0)
    if target_us <= max_overhead:
        return transfers, {"target_us": target_us, "moves": 0, "path_scores": 0, "target_met": False}
    limits = {key: cap * (target_us - max_overhead) / 1e6 for key, cap in capacities.items()}
    current, members = defaultdict(int), defaultdict(list)
    remaining = [len(row["chunk_ranges"]) for row in transfers]
    consumed = [0] * len(transfers)
    allocations = [{} for _ in transfers]
    for ti, path in enumerate(paths):
        for key, multiple in describe(path)[0].items():
            current[key] += remaining[ti] * chunk_bytes * multiple
            members[key].append(ti)
    moves = scores = 0
    reason = "all_capacity_constraints_met"
    while moves < max_updates and time.perf_counter() < deadline:
        overloaded = [key for key, amount in current.items() if amount > limits[key] + 1e-6]
        if not overloaded:
            break
        critical = max(overloaded, key=lambda key: (current[key] / limits[key], repr(key)))
        best = None
        for ti in members[critical]:
            if not remaining[ti]:
                continue
            old_uses = describe(paths[ti])[0]
            for path in alternatives[ti]:
                uses, overhead, per_byte = describe(path)
                relief = old_uses[critical] - uses[critical]
                if relief <= 0:
                    continue
                scores += 1
                required = math.ceil((current[critical] - limits[critical]) / (relief * chunk_bytes))
                available = min(required, remaining[ti])
                deltas = {key: uses[key] - old_uses[key] for key in uses.keys() | old_uses.keys()}
                for key, delta in deltas.items():
                    if delta > 0:
                        available = min(available, max(0, math.floor((limits[key] - current[key]) / (delta * chunk_bytes) + 1e-9)))
                already = len(allocations[ti].get(path, ()))
                available = min(available, max(0, math.floor((target_us - overhead) / (per_byte * chunk_bytes) + 1e-9) - already))
                if available <= 0:
                    continue
                highest = max(((current[key] + delta * available * chunk_bytes) / limits[key]
                               for key, delta in deltas.items() if delta > 0), default=0.0)
                rank = (-relief * available, highest, overhead + available * chunk_bytes * per_byte, ti, path)
                if best is None or rank < best[0]:
                    best = rank, ti, path, available, deltas
        if best is None:
            reason = "no_integer_move_fits_remaining_capacity"
            break
        _, ti, path, count, deltas = best
        for key, delta in deltas.items():
            current[key] += delta * count * chunk_bytes
        first = consumed[ti]
        allocations[ti].setdefault(path, []).extend(transfers[ti]["chunk_ranges"][first:first + count])
        consumed[ti] += count
        remaining[ti] -= count
        moves += 1
    else:
        reason = "update_limit" if moves >= max_updates else "wall_safety"
    result = []
    for ti, row in enumerate(transfers):
        if remaining[ti]:
            result.append({**row, "chunk_ranges": row["chunk_ranges"][consumed[ti]:]})
        result.extend({**row, "path": list(path), "chunk_ranges": tokens}
                      for path, tokens in allocations[ti].items())
    actual = _round_load({"transfers": result}, costs, chunk_bytes).time_us
    return result, {"target_us": target_us, "moves": moves, "path_scores": scores,
                    "time_us": actual, "target_met": actual <= target_us + 1e-6, "stop_reason": reason}


def optimize(task, graph, solution, limits=Limits()):
    """Finite offline experiment, including baseline and final full Replay."""
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    baseline = replay(task, graph, solution)
    if not baseline["valid"]:
        raise ValueError("invalid experiment baseline: " + str(baseline["errors"]))
    baseline_replay_cpu = time.process_time() - start_cpu
    before = baseline["total_time_us"]
    deadline = time.perf_counter() + limits.search_seconds
    costs = _PathCosts(graph)
    chunk_bytes = task["data_size"]["per_rank_bytes"] // solution["split"]
    loads = [_round_load(row, costs, chunk_bytes) for row in solution["rounds"]]
    outgoing, incoming, catalog = defaultdict(set), defaultdict(set), defaultdict(set)
    for src, dst in graph.edges:
        if src not in graph.switches and dst in graph.switches:
            outgoing[src].add(dst)
        if src in graph.switches and dst not in graph.switches:
            incoming[dst].add(src)
    for round_ in solution["rounds"]:
        for row in round_["transfers"]:
            catalog[row["src"], row["dst"]].add(tuple(row["path"]))
    changed_rounds = list(solution["rounds"])
    records, changed_indices, searches, scores, updates = [], [], 0, 0, 0
    for ri in sorted(range(len(loads)), key=lambda index: (-loads[index].time_us, index))[:limits.max_rounds]:
        if time.perf_counter() >= deadline or updates >= limits.max_updates:
            break
        round_ = solution["rounds"][ri]
        if limits.method == "capacity_repair":
            for row in round_["transfers"]:
                pair = row["src"], row["dst"]
                if pair in graph.edges:
                    catalog[pair].add(pair)
                catalog[pair].update((pair[0], switch, pair[1]) for switch in sorted(outgoing[pair[0]] & incoming[pair[1]]))
            best_round, best_cost = round_, loads[ri].time_us
            record = {"round": ri + 1, "before_us": best_cost, "targets": []}
            for ratio in (0.995, 0.99, 0.98, 0.95, 0.90):
                if time.perf_counter() >= deadline or updates >= limits.max_updates:
                    break
                transfers, trial = repair_to_target(round_["transfers"], costs, chunk_bytes, catalog,
                                                     loads[ri].time_us * ratio, deadline, limits.max_updates - updates)
                updates += trial["moves"]
                scores += trial["path_scores"]
                record["targets"].append(trial)
                if trial.get("time_us", math.inf) < best_cost - max(1e-6, best_cost * 1e-10):
                    best_round, best_cost = {**round_, "transfers": transfers}, trial["time_us"]
            record["after_us"] = best_cost
            if best_round is not round_:
                if atomic_actions(best_round) != atomic_actions(round_):
                    raise AssertionError("capacity repair changed a data action")
                changed_rounds[ri] = best_round
                changed_indices.append(ri)
            records.append(record)
            continue
        expanded, parents = _batch_transfers(round_["transfers"], limits.batches)
        state = Assignment(expanded, costs, chunk_bytes, limits.exponent, parents)
        best_cost, best_paths = state.exact_cost(), list(state.paths)
        initial_paths = list(state.paths)
        peaks = loads[ri].bottlenecks()
        record = {"round": ri + 1, "before_us": best_cost, "tied_near_peak_edges": len(peaks[0]),
                  "tied_near_peak_resources": len(peaks[1]), "sweeps": []}
        for sweep in range(limits.sweeps):
            changes, visited, completed = 0, 0, True
            order = sorted(range(len(state.paths)), key=lambda ti: (-state.sizes[ti], ti))
            if sweep % 2:
                order.reverse()
            for ti in order:
                if time.perf_counter() >= deadline or updates >= limits.max_updates:
                    completed = False
                    break
                row, original, count = state.transfers[ti], state.paths[ti], state.sizes[ti]
                pair = row["src"], row["dst"]
                paths = catalog[pair]
                if pair in graph.edges:
                    paths.add(pair)
                paths.update((pair[0], switch, pair[1]) for switch in sorted(outgoing[pair[0]] & incoming[pair[1]]))
                state.add(original, -count)
                try:
                    if searches < limits.priced_searches:
                        alternate = state.priced_path(*pair, count, deadline)
                        searches += 1
                        if alternate and _legal_path(graph, *pair, alternate):
                            paths.add(alternate)
                    # Score every discovered alternative; retain at most K in
                    # the reusable per-pair catalog, always keeping originals.
                    ranked = sorted(paths | {original, initial_paths[ti]}, key=lambda path: state.score(path, count))
                    scores += len(ranked)
                    chosen = ranked[0]
                    keep = set(ranked[:limits.max_paths]) | {original, initial_paths[ti]}
                    catalog[pair] = keep
                    state.paths[ti] = chosen
                finally:
                    state.add(state.paths[ti], count)
                changes += state.paths[ti] != original
                visited += 1
                updates += 1
            exact = state.exact_cost()
            record["sweeps"].append({"index": sweep + 1, "changes": changes, "visited": visited,
                                     "completed": completed, "time_us": exact, "potential": state.potential()})
            # Even a deadline-interrupted sweep has complete transfer moves;
            # its unvisited traffic remains in the exact full assignment.
            if exact < best_cost - max(1e-6, best_cost * 1e-10):
                best_cost, best_paths = exact, list(state.paths)
            if not completed or changes == 0:
                break
        moved = [i for i, (a, b) in enumerate(zip(initial_paths, best_paths)) if a != b]
        record.update(after_us=best_cost, changed_original_transfers=len({parents[i] for i in moved}),
                      changed_batches=len(moved), moved_atomic_items=sum(len(expanded[i]["chunk_ranges"]) for i in moved))
        if moved:
            updated = {**round_, "transfers": state.merged_transfers(best_paths)}
            if atomic_actions(updated) != atomic_actions(round_):
                raise AssertionError("joint assignment changed an atomic data action")
            changed_rounds[ri] = updated
            changed_indices.append(ri)
        records.append(record)
    search_cpu = time.process_time() - start_cpu - baseline_replay_cpu
    candidate = {**solution, "rounds": changed_rounds} if changed_indices else solution
    replay_started = time.process_time()
    checked = replay(task, graph, candidate)
    final_replay_cpu = time.process_time() - replay_started
    accepted = bool(changed_indices) and checked["valid"] and checked["total_time_us"] < before - max(1e-6, before * 1e-10)
    if accepted:
        candidate["metrics"] = {**solution.get("metrics", {}), "total_time": checked["total_time_us"] / 1e6,
                                 "round_times": [value / 1e6 for value in checked["round_times_us"]]}
    else:
        candidate = solution
    return candidate, {"accepted": accepted, "before_us": before,
        "after_us": checked["total_time_us"] if accepted else before,
        "gain_percent": 100 * (1 - checked["total_time_us"] / before) if accepted and before else 0,
        "candidate_replay_valid": checked["valid"], "candidate_replay_errors": checked["errors"],
        "returned_replay_valid": True, "rounds": records, "priced_searches": searches, "path_scores": scores,
        "completed_updates": updates, "update_limit_reached": updates >= limits.max_updates,
        "baseline_replay_cpu_s": baseline_replay_cpu, "search_cpu_s": search_cpu,
        "final_replay_cpu_s": final_replay_cpu, "cpu_s": time.process_time() - start_cpu,
        "wall_s": time.perf_counter() - start_wall, "search_deadline_reached": time.perf_counter() >= deadline}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=[f"{name}/{op}" for name in ("mesh2d", "cloudmatrix", "nvlink_like") for op in ("ag", "ar")])
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--searches", type=int, default=512)
    parser.add_argument("--exponent", type=int, default=8)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--updates", type=int, default=65536)
    parser.add_argument("--method", choices=("potential", "capacity_repair"), default="potential")
    parser.add_argument("--output", default="pilot.json")
    args = parser.parse_args()
    target = (OUTPUT / args.output).resolve()
    if not target.is_relative_to(OUTPUT):
        parser.error("output must remain under the joint-routes artifact directory")
    limits = Limits(args.rounds, args.sweeps, 8, args.searches, args.seconds, args.exponent, args.batches, args.updates, args.method)
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((ROOT / "src").glob("*.py"))}
    report = {"scope": __doc__, "limits": vars(limits), "source_sha256_at_start": sources,
              "experiment_file": str(Path(__file__).relative_to(ROOT)),
              "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "attribution": "deterministic offline control; no LLM called; not an online latency comparison", "rows": []}
    target.parent.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        folder = PUBLIC / case
        task = json.loads((folder / "task.json").read_text())
        topo = json.loads((folder / "topology.json").read_text())
        archived = ARCHIVE / (case.replace("/", "-") + "-r01.json")
        baseline = json.loads(archived.read_text())["output"]["answer"]
        original_hash = digest(baseline)
        final, info = optimize(task, Topology.from_data(topo), baseline, limits)
        if digest(baseline) != original_hash:
            raise AssertionError("experiment mutated its archived baseline")
        output = target.with_name(target.stem + "-" + case.replace("/", "-") + "-answer.json")
        output.write_text(json.dumps(final, separators=(",", ":")) + "\n")
        row = {"case": case, "input_sha256": digest({"task": task, "topology": topo}),
               "task_file_sha256": hashlib.sha256((folder / "task.json").read_bytes()).hexdigest(),
               "topology_file_sha256": hashlib.sha256((folder / "topology.json").read_bytes()).hexdigest(),
               "baseline_file": str(archived.relative_to(ROOT)), "baseline_sha256": original_hash,
               "baseline_file_sha256": hashlib.sha256(archived.read_bytes()).hexdigest(),
               "baseline_transfers": sum(len(r["transfers"]) for r in baseline["rounds"]),
               "final_transfers": sum(len(r["transfers"]) for r in final["rounds"]),
               "baseline_answer_bytes": len(json.dumps(baseline, separators=(",", ":")).encode()),
               "final_answer_bytes": len(json.dumps(final, separators=(",", ":")).encode()),
               "answer_file": str(output.relative_to(ROOT)), "answer_sha256": digest(final), **info}
        report["rows"].append(row)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in ("case", "accepted", "gain_percent", "cpu_s", "wall_s", "priced_searches", "path_scores")}), flush=True)
    after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted((ROOT / "src").glob("*.py"))}
    report["source_sha256_at_end"] = after
    report["production_source_unchanged"] = after == sources
    report["experiment_source_unchanged"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == report["experiment_sha256"]
    target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
