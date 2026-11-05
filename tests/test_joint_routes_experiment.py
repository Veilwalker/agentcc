"""Correctness checks for the isolated joint-route research prototype."""

from copy import deepcopy
import unittest

from src.collectives import build
from src.replay import replay
from src.route_balance import _PathCosts
from src.topology import Topology
from tests.joint_routes_experiment import Assignment, Limits, _batch_transfers, atomic_actions, optimize
from tests.test_route_balance import parallel_case


class JointRoutesExperimentTests(unittest.TestCase):
    def test_joint_moves_cross_a_tied_maximum_without_single_move_improvement(self):
        graph = Topology.from_data({"ranks": ["r0", "r1", "r2", "r3"], "switches": ["a", "b"],
            "edges": [{"src": "r0", "dst": "r1", "bw": "1GB/s"},
                      {"src": "r2", "dst": "r3", "bw": "1GB/s"},
                      {"src": "r0", "dst": "a", "bw": "10GB/s"},
                      {"src": "a", "dst": "r1", "bw": "10GB/s"},
                      {"src": "r2", "dst": "b", "bw": "10GB/s"},
                      {"src": "b", "dst": "r3", "bw": "10GB/s"}]})
        rows = [{"action": "copy", "src": src, "dst": dst, "path": [src, dst],
                 "chunk_ranges": [src + "_c0"]} for src, dst in [("r0", "r1"), ("r2", "r3")]]
        state = Assignment(rows, _PathCosts(graph), 8192, 8)
        before = state.exact_cost()
        for ti, switch in enumerate(("a", "b")):
            old = state.paths[ti]
            new = old[0], switch, old[-1]
            state.add(old, -8192)
            self.assertLess(state.score(new, 8192), state.score(old, 8192))
            state.paths[ti] = new
            state.add(new, 8192)
            if ti == 0:
                self.assertGreaterEqual(state.exact_cost(), before)
        self.assertLess(state.exact_cost(), before)

    def test_batches_on_same_path_are_merged_before_exact_cost(self):
        task, graph = parallel_case()
        baseline = build(task, graph, 4, packets=1)
        rows = baseline["rounds"][-1]["transfers"]
        costs = _PathCosts(graph)
        size = task["data_size"]["per_rank_bytes"] // 4
        original = Assignment(rows, costs, size, 8)
        expanded, parents = _batch_transfers(rows, 8)
        batched = Assignment(expanded, costs, size, 8, parents)
        self.assertGreater(len(expanded), len(rows))
        self.assertEqual(batched.exact_cost(), original.exact_cost())
        self.assertEqual(batched.merged_transfers(batched.paths), rows)

    def test_potential_delta_accounts_for_repeated_overlapping_resources(self):
        _, graph = parallel_case()
        transfers = [{"action": "copy", "src": "r0", "dst": "r1", "chunk_ranges": ["r0_c0"],
                      "path": ["r0", "fast", "r1"]},
                     {"action": "copy", "src": "r2", "dst": "r3", "chunk_ranges": ["r2_c0"],
                      "path": ["r2", "fast", "r3"]}]
        state = Assignment(transfers, _PathCosts(graph), 8192, 8)
        self.assertEqual(state.resources["shared-a"], 4 * 8192)
        self.assertEqual(state.resources["shared-b"], 4 * 8192)
        before = state.potential()
        original, new = state.paths[0], ("r0", "alternate", "r1")
        state.add(original, -8192)
        predicted = state.score(new, 8192)[0] - state.score(original, 8192)[0]
        state.paths[0] = new
        state.add(new, 8192)
        self.assertAlmostEqual(state.potential() - before, predicted, places=12)

    def test_complete_collectives_preserve_atoms_packets_and_baseline(self):
        for operator, kind in (("all-gather", "reverse_doubling"), ("all-reduce", "xor_tree")):
            with self.subTest(operator=operator):
                task, graph = parallel_case(operator)
                baseline = build(task, graph, 4, kind=kind, resource_weight=0, packets=2)
                original = deepcopy(baseline)
                result, info = optimize(task, graph, baseline, Limits(search_seconds=1.0, priced_searches=0))
                self.assertTrue(info["accepted"])
                self.assertTrue(replay(task, graph, result)["valid"])
                self.assertLess(info["after_us"], info["before_us"])
                self.assertEqual(baseline, original)
                for old, new in zip(baseline["rounds"], result["rounds"]):
                    self.assertEqual(atomic_actions(old), atomic_actions(new))
                    self.assertEqual([row["chunk_ranges"] for row in old["transfers"]],
                                     [row["chunk_ranges"] for row in new["transfers"]])

    def test_zero_optional_budget_returns_exact_valid_baseline(self):
        task, graph = parallel_case()
        baseline = build(task, graph)
        result, info = optimize(task, graph, baseline, Limits(search_seconds=0))
        self.assertIs(result, baseline)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["priced_searches"], 0)
        self.assertTrue(info["returned_replay_valid"])

    def test_update_cap_keeps_a_complete_replayed_assignment(self):
        task, graph = parallel_case()
        baseline = build(task, graph, 4, resource_weight=0, packets=2)
        result, info = optimize(task, graph, baseline, Limits(max_updates=3, priced_searches=0))
        self.assertEqual(info["completed_updates"], 3)
        self.assertTrue(info["update_limit_reached"])
        self.assertTrue(replay(task, graph, result)["valid"])
        self.assertLessEqual(info["after_us"], info["before_us"])

    def test_capacity_target_repair_preserves_complete_reduce_expressions(self):
        task, graph = parallel_case("all-reduce")
        baseline = build(task, graph, 4, kind="xor_tree", resource_weight=0, packets=2)
        original = deepcopy(baseline)
        result, info = optimize(task, graph, baseline, Limits(method="capacity_repair", search_seconds=1.0))
        self.assertTrue(info["accepted"])
        self.assertTrue(replay(task, graph, result)["valid"])
        self.assertEqual(baseline, original)
        for old, new in zip(baseline["rounds"], result["rounds"]):
            self.assertEqual(atomic_actions(old), atomic_actions(new))


if __name__ == "__main__":
    unittest.main()
