"""Offline v1.4 experiment: integer path splitting on a validated schedule.

This is deliberately not imported by production code. It holds the data DAG,
rounds, split, endpoints and complete reduction expressions fixed. The only
variable is how many whole atomic items of one transfer use a second legal
path. Its interior objective is a maximum of affine functions; a crossing of
the increasing/decreasing envelopes gives a small set of integer candidates.

Run from the project root with ``python3 -m tests.route_split_experiment``.
All reported gains belong to this deterministic control experiment, not LLM
actions. Archived v1.3 final answers may already include accepted model edits.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
import unittest

from src.replay import replay
from src.route_balance import _PathCosts, _RoundLoad
from src.schedule_edit import _avoiding_path, _legal_path
from src.topology import Topology


def atomic_actions(round_):
    return Counter((row["action"], row["src"], row["dst"], item)
                   for row in round_["transfers"] for item in row["chunk_ranges"])


class RoundState:
    """Exact baseline loads and order statistics for cheap local deltas."""

    def __init__(self, transfers, costs, chunk_bytes):
        self.transfers, self.costs, self.chunk_bytes = transfers, costs, chunk_bytes
        self.load = _RoundLoad(costs)
        self.path_times, self.overheads = [], []
        for index, row in enumerate(transfers):
            count = len(row["chunk_ranges"]) * chunk_bytes
            _, _, overhead, per_byte = costs.describe(row["path"])
            self.path_times.append((overhead + count * per_byte, index))
            self.overheads.append((overhead, index))
            self.load.add(row["path"], count)
        graph = costs.graph
        self.busy = sorted(
            [(1e6 * count / graph.edges[edge], "edge", edge)
             for edge, count in self.load.edge_bytes.items()]
            + [(1e6 * count / graph.resources[name], "resource", name)
               for name, count in self.load.resource_bytes.items()], reverse=True)
        self.path_times.sort(reverse=True)
        self.overheads.sort(reverse=True)

    def oracle(self, index, alternate):
        """Return exact cost(k), with k complete items moved to alternate."""
        row, graph = self.transfers[index], self.costs.graph
        size, unit = len(row["chunk_ranges"]), self.chunk_bytes
        old_edges, old_resources, old_overhead, old_per_byte = self.costs.describe(row["path"])
        new_edges, new_resources, new_overhead, new_per_byte = self.costs.describe(alternate)
        old_edges, new_edges = set(old_edges), set(new_edges)
        edges, resources = old_edges | new_edges, old_resources.keys() | new_resources.keys()
        other_busy = next((value for value, kind, key in self.busy
                           if key not in (edges if kind == "edge" else resources)), 0.0)
        other_path = next((value for value, ti in self.path_times if ti != index), 0.0)
        other_overhead = next((value for value, ti in self.overheads if ti != index), 0.0)
        busy_lines = [(other_busy, 0.0)]
        for edge in edges:
            factor = 1e6 / graph.edges[edge]
            busy_lines.append((factor * self.load.edge_bytes.get(edge, 0),
                               factor * unit * (int(edge in new_edges) - int(edge in old_edges))))
        for name in resources:
            factor = 1e6 / graph.resources[name]
            busy_lines.append((factor * self.load.resource_bytes.get(name, 0),
                               factor * unit * (new_resources[name] - old_resources[name])))

        def cost(moved):
            overhead = other_overhead
            path = other_path
            if moved < size:
                overhead = max(overhead, old_overhead)
                path = max(path, old_overhead + (size - moved) * unit * old_per_byte)
            if moved:
                overhead = max(overhead, new_overhead)
                path = max(path, new_overhead + moved * unit * new_per_byte)
            return max(path, overhead + max(a + b * moved for a, b in busy_lines))

        overhead = max(other_overhead, old_overhead, new_overhead)
        lines = [(other_path, 0.0),
                 (old_overhead + size * unit * old_per_byte, -unit * old_per_byte),
                 (new_overhead, unit * new_per_byte)]
        lines += [(overhead + a, b) for a, b in busy_lines]
        return cost, lines


def integer_candidates(size, lines):
    """Exact interior convex minimizer candidates plus endpoints and half."""
    increasing = [(a, b) for a, b in lines if b > 0]
    decreasing = [(a, b) for a, b in lines if b < 0]
    low, high = 1, size - 1
    while low < high:
        middle = (low + high) // 2
        up = max((a + b * middle for a, b in increasing), default=-math.inf)
        down = max((a + b * middle for a, b in decreasing), default=-math.inf)
        if up < down:
            low = middle + 1
        else:
            high = middle
    return sorted({0, size, size // 2, (size + 1) // 2,
                   max(0, low - 1), low, min(size, low + 1)})


def split_transfer(row, alternate, count):
    parts = []
    if count:
        parts.append({**row, "path": list(alternate), "chunk_ranges": row["chunk_ranges"][:count]})
    if count < len(row["chunk_ranges"]):
        parts.append({**row, "chunk_ranges": row["chunk_ranges"][count:]})
    return parts


def optimize(task, graph, solution, *, search_budget_s=0.06, max_edits=4, max_paths=6):
    """Search valid baseline; final full Replay is included in measured cost."""
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    end = started_wall + search_budget_s
    costs = _PathCosts(graph)
    chunk_bytes = task["data_size"]["per_rank_bytes"] // solution["split"]
    states = [RoundState(row["transfers"], costs, chunk_bytes) for row in solution["rounds"]]
    initialized_s = time.process_time() - started_cpu
    before = sum(state.load.time_us for state in states)
    options, outgoing, incoming = defaultdict(set), defaultdict(set), defaultdict(set)
    for src, dst in graph.edges:
        if src not in graph.switches and dst in graph.switches:
            outgoing[src].add(dst)
        if src in graph.switches and dst not in graph.switches:
            incoming[dst].add(src)
    for row in solution["rounds"]:
        for transfer in row["transfers"]:
            options[transfer["src"], transfer["dst"]].add(tuple(transfer["path"]))
    rounds, edits, trials = list(solution["rounds"]), [], 0
    searched = set()
    for ri in sorted(range(len(states)), key=lambda i: -states[i].load.time_us):
        if time.perf_counter() >= end or len(edits) >= max_edits:
            break
        state = states[ri]
        critical_edges, critical_resources = state.load.bottlenecks()
        transfers = state.transfers
        best = None
        for ti in sorted(range(len(transfers)), key=lambda i: -len(transfers[i]["chunk_ranges"])):
            if time.perf_counter() >= end:
                break
            transfer = transfers[ti]
            size = len(transfer["chunk_ranges"])
            if size < 2:
                continue
            original = tuple(transfer["path"])
            old_edges, old_resources, overhead, per_byte = costs.describe(original)
            if not (critical_edges.intersection(old_edges) or critical_resources.intersection(old_resources)
                    or overhead + size * chunk_bytes * per_byte >= state.load.time_us - 1e-8):
                continue
            pair = transfer["src"], transfer["dst"]
            alternatives = options[pair]
            if pair in graph.edges:
                alternatives.add(pair)
            alternatives.update((pair[0], switch, pair[1])
                                for switch in outgoing[pair[0]] & incoming[pair[1]])
            if pair not in searched and (len(original) > 3 or len(alternatives) < 2):
                searched.add(pair)
                path = _avoiding_path(graph, *pair, state.load, set(old_edges) & critical_edges,
                                      set(old_resources) & critical_resources,
                                      time.monotonic() + max(0, end - time.perf_counter()))
                if path and _legal_path(graph, *pair, path):
                    alternatives.add(tuple(path))
            ordered = sorted((path for path in alternatives if path != original),
                             key=lambda path: state.load.score(path, size * chunk_bytes))[:max_paths]
            for path in ordered:
                if time.perf_counter() >= end:
                    break
                cost, lines = state.oracle(ti, path)
                values = integer_candidates(size, lines)
                trials += len(values)
                value, moved = min((cost(k), k) for k in values)
                if value < state.load.time_us - 1e-8 and (best is None or value < best[0]):
                    best = value, ti, path, moved
        if best is not None:
            predicted, ti, path, moved = best
            updated = transfers[:ti] + split_transfer(transfers[ti], path, moved) + transfers[ti + 1:]
            checked_state = RoundState(updated, costs, chunk_bytes)
            if not math.isclose(checked_state.load.time_us, predicted, rel_tol=1e-11, abs_tol=1e-7):
                raise AssertionError("local delta does not equal full round cost")
            rounds[ri] = {**rounds[ri], "transfers": updated}
            if atomic_actions(rounds[ri]) != atomic_actions(solution["rounds"][ri]):
                raise AssertionError("route split changed data actions")
            edits.append({"round": ri, "transfer": ti, "moved": moved,
                          "items": len(transfers[ti]["chunk_ranges"]), "path": list(path),
                          "before_us": state.load.time_us, "after_us": predicted})
    search_cpu = time.process_time() - started_cpu
    candidate = {**solution, "rounds": rounds} if edits else solution
    replay_started = time.process_time()
    report = replay(task, graph, candidate)
    replay_cpu = time.process_time() - replay_started
    if not report["valid"] or report["total_time_us"] > before + 1e-6:
        raise AssertionError(report)
    if edits:
        candidate["metrics"] = {**solution.get("metrics", {}), "total_time": report["total_time_us"] / 1e6,
                                "round_times": [value / 1e6 for value in report["round_times_us"]]}
    return candidate, {"before_us": before, "after_us": report["total_time_us"],
                       "gain_pct": 100 * (1 - report["total_time_us"] / before),
                       "wall_s": time.perf_counter() - started_wall,
                       "cpu_s": time.process_time() - started_cpu,
                       "initialize_cpu_s": initialized_s, "search_cpu_s": search_cpu,
                       "replay_cpu_s": replay_cpu, "trials": trials, "edits": edits,
                       "deadline_reached": time.perf_counter() >= end, "replay_valid": True}


class SplitExperimentTests(unittest.TestCase):
    def test_complete_collectives_remain_valid_and_immutable(self):
        from src.collectives import build
        from tests.test_route_balance import parallel_case
        for operator, kind in (("all-gather", "reverse_doubling"), ("all-reduce", "xor_tree")):
            with self.subTest(operator=operator):
                task, graph = parallel_case(operator)
                baseline = build(task, graph, 4, kind=kind, resource_weight=0, packets=1)
                original = deepcopy(baseline)
                changed, evidence = optimize(task, graph, baseline, search_budget_s=0.10)
                report = replay(task, graph, changed)
                self.assertTrue(report["valid"], report)
                self.assertLess(evidence["after_us"], evidence["before_us"])
                self.assertEqual(baseline, original)
                for old, new in zip(baseline["rounds"], changed["rounds"]):
                    self.assertEqual(atomic_actions(old), atomic_actions(new))

    def test_integer_crossing_matches_exhaustive_convex_minimum(self):
        for size in range(2, 66):
            lines = [(11.0, 0.0), (3.0, 1.3), (size * 1.9, -1.9), (size * 0.8, -0.4)]
            cost = lambda k: max(a + b * k for a, b in lines)
            self.assertEqual(min(cost(k) for k in range(size + 1)),
                             min(cost(k) for k in integer_candidates(size, lines)))

    def test_delta_matches_rebuilt_round_with_multiplicity_and_overhead(self):
        from tests.test_solver import make_case
        task, raw = make_case(4)
        raw["switches"] = ["a", "b", "c"]
        raw["edges"] = [
            {"src": "r0", "dst": "a", "bw": "9GB/s"},
            {"src": "a", "dst": "r1", "bw": "7GB/s"},
            {"src": "r0", "dst": "b", "bw": "8GB/s"},
            {"src": "b", "dst": "c", "bw": "6GB/s"},
            {"src": "c", "dst": "r1", "bw": "5GB/s"},
            {"src": "r2", "dst": "a", "bw": "9GB/s"},
            {"src": "a", "dst": "r3", "bw": "7GB/s"}]
        raw["resources"] = [
            {"id": "shared", "capacity": "2GB/s", "edges": [
                {"src": "r0", "dst": "a"}, {"src": "a", "dst": "r1"},
                {"src": "r2", "dst": "a"}, {"src": "b", "dst": "c"}]},
            {"id": "overlap", "capacity": "3GB/s", "edges": [
                {"src": "a", "dst": "r1"}, {"src": "c", "dst": "r1"}]}]
        costs = _PathCosts(Topology.from_data(raw))
        rows = [{"action": "copy", "src": "r0", "dst": "r1", "path": ["r0", "a", "r1"],
                 "chunk_ranges": ["r0_c0", "r0_c1", "r0_c2", "r0_c3"]},
                {"action": "copy", "src": "r2", "dst": "r3", "path": ["r2", "a", "r3"],
                 "chunk_ranges": ["r2_c0", "r2_c1"]}]
        alternate = ("r0", "b", "c", "r1")
        state = RoundState(rows, costs, 1 << 20)
        cost, _ = state.oracle(0, alternate)
        for moved in range(5):
            changed = split_transfer(rows[0], alternate, moved) + rows[1:]
            self.assertAlmostEqual(cost(moved), RoundState(changed, costs, 1 << 20).load.time_us)
            self.assertEqual(atomic_actions({"transfers": rows}), atomic_actions({"transfers": changed}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--archive", type=Path, default=Path("artifacts/v1.3/runs/public-01"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/v1.4/analysis/route-split/pilot.json"))
    parser.add_argument("--cases", nargs="+", default=[f"{kind}-{op}" for kind in
                        ("cloudmatrix", "mesh2d", "nvlink_like") for op in ("ag", "ar")])
    parser.add_argument("--search-budget", type=float, default=0.06)
    parser.add_argument("--max-paths", type=int, default=6)
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=[__file__])
        return
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    results = []
    for case in args.cases:
        request = json.loads((args.archive / "inputs" / f"{case}.json").read_text())["input"]
        response = json.loads((args.archive / "v1.3" / f"{case}-r01.json").read_text())
        solution = response["output"]["answer"]
        graph = Topology.from_data(request["topology"])
        baseline = replay(request["task"], graph, solution)
        if not baseline["valid"]:
            raise ValueError(baseline)
        _, evidence = optimize(request["task"], graph, solution,
                               search_budget_s=args.search_budget, max_paths=args.max_paths)
        evidence["case"] = case
        results.append(evidence)
        print(case, json.dumps({k: evidence[k] for k in
              ("gain_pct", "cpu_s", "search_cpu_s", "replay_cpu_s", "trials")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"baseline": "archived v1.3 final answer, including any model edits",
                                     "scope": "deterministic control experiment; not LLM-attributed",
                                     "affinity": sorted(os.sched_getaffinity(0)),
                                     "search_budget_s": args.search_budget, "rows": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
