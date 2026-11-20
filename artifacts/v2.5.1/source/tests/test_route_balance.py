"""Routing refinement must preserve schedules and exact resource accounting."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.route_balance import refine
from src.topology import Topology
from tests.test_solver import make_case


def parallel_case(operator="all-gather"):
    task, topology = make_case(8, operator)
    topology["switches"] = ["fast", "alternate"]
    topology["edges"] = []
    for rank in task["ranks"]:
        topology["edges"].extend([
            {"src": rank, "dst": "fast", "bw": "10GB/s", "bidirectional": True},
            {"src": rank, "dst": "alternate", "bw": "8GB/s", "bidirectional": True},
        ])
    all_fast = [{"src": rank, "dst": "fast", "bidirectional": True} for rank in task["ranks"]]
    topology["resources"] = [
        {"id": "shared-a", "capacity": "4GB/s", "edges": deepcopy(all_fast)},
        {"id": "shared-b", "capacity": "5GB/s", "edges": deepcopy(all_fast)},
    ]
    return task, Topology.from_data(topology)


class RouteBalanceTests(unittest.TestCase):
    def test_improves_congestion_for_both_operators(self):
        for operator, kind in [("all-gather", "reverse_doubling"), ("all-reduce", "xor_tree")]:
            with self.subTest(operator=operator):
                task, graph = parallel_case(operator)
                solution = build(task, graph, 4, kind=kind, resource_weight=0, packets=2)
                before = replay(task, graph, solution)
                original = deepcopy(solution)
                result, evidence = refine(task, graph, solution, time_budget_s=1)
                after = replay(task, graph, result)
                self.assertTrue(after["valid"], after["errors"])
                self.assertLess(after["total_time_us"], before["total_time_us"])
                self.assertAlmostEqual(evidence["before_us"], before["total_time_us"])
                self.assertAlmostEqual(evidence["after_us"], after["total_time_us"])
                self.assertGreater(evidence["improved_rounds"], 0)
                self.assertEqual(solution, original)

    def test_expired_budget_keeps_valid_original_without_path_search(self):
        task, graph = parallel_case()
        solution = build(task, graph, resource_weight=0)
        with patch.object(graph, "path", side_effect=AssertionError("unexpected path search")):
            result, evidence = refine(task, graph, solution, time_budget_s=0)
        self.assertIs(result, solution)
        self.assertTrue(evidence["replay_valid"])
        self.assertEqual(evidence["improved_rounds"], 0)

    def test_no_alternative_paths_skip_refinement(self):
        task, topology = make_case(8)
        graph = Topology.from_data(topology)
        solution = build(task, graph)
        result, evidence = refine(task, graph, solution)
        self.assertIs(result, solution)
        self.assertFalse(evidence["promising"])
        self.assertEqual(evidence["pairs_with_alternatives"], 0)

    def test_changed_routes_update_optional_metrics(self):
        task, graph = parallel_case()
        solution = build(task, graph, resource_weight=0)
        solution["metrics"] = {"total_time": 999, "round_times": []}
        result, evidence = refine(task, graph, solution, time_budget_s=1)
        self.assertGreater(evidence["improved_rounds"], 0)
        self.assertAlmostEqual(result["metrics"]["total_time"] * 1e6, evidence["after_us"])
        self.assertEqual(solution["metrics"]["total_time"], 999)


if __name__ == "__main__":
    unittest.main()
