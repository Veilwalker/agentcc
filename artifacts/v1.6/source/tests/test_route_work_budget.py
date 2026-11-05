"""A deterministic route work prefix must remain complete, legal and replayed."""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.route_balance import _PathCosts, _RoundLoad
from src.route_broad import refine_broad
from src.route_work import RouteWorkBudget, RouteWorkMeter
from src.schedule_edit import _avoiding_path
from tests.test_edit_invariants import actions
from tests.test_route_balance import parallel_case


class RouteWorkTests(unittest.TestCase):
    def fixture(self, operator="all-gather"):
        task, graph = parallel_case(operator)
        kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
        solution = build(task, graph, 4, kind=kind, resource_weight=0, packets=1)
        return task, graph, solution

    def test_explicit_budget_is_reproducible_and_preserves_data_actions(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline = self.fixture(operator)
                original = deepcopy(baseline)
                results, work = [], []
                for _ in range(3):
                    result, info = refine_broad(task, graph, baseline,
                                               work_budget=RouteWorkBudget(13, 10_000))
                    self.assertTrue(info["final_replay"]["valid"])
                    self.assertEqual(info["final_replay"], replay(task, graph, result))
                    self.assertEqual(actions(result), actions(baseline))
                    self.assertLess(info["after_us"], info["before_us"])
                    results.append(result)
                    work.append(info["work"])
                self.assertEqual(results[0], results[1])
                self.assertEqual(results[1], results[2])
                self.assertEqual(work[0], work[1])
                self.assertEqual(work[1], work[2])
                self.assertEqual(baseline, original)

    def test_admitted_transfer_scores_all_paths_before_stopping(self):
        task, graph, baseline = self.fixture()
        _, info = refine_broad(task, graph, baseline, work_budget=RouteWorkBudget(1, 10_000))
        self.assertEqual(info["work"]["completed_transfers"], 1)
        self.assertEqual(info["work"]["path_scores"], 2)
        self.assertEqual(info["work"]["stop_reason"], "path_score_limit")
        self.assertTrue(info["final_replay"]["valid"])

    def test_frozen_transfers_and_zero_work_preserve_baseline(self):
        task, graph, baseline = self.fixture()
        fixed = [[ri, 0] for ri in range(len(baseline["rounds"]))]
        result, info = refine_broad(task, graph, baseline, frozen=fixed,
                                   work_budget=RouteWorkBudget(20, 10_000))
        self.assertTrue(info["frozen_preserved"])
        for ri, ti in fixed:
            self.assertEqual(result["rounds"][ri]["transfers"][ti], baseline["rounds"][ri]["transfers"][ti])
        for budget in (RouteWorkBudget(0, 10_000), RouteWorkBudget(20, 0)):
            result, info = refine_broad(task, graph, baseline, work_budget=budget)
            self.assertIs(result, baseline)
            self.assertEqual(info["work"]["completed_transfers"], 0)
            self.assertTrue(info["final_replay"]["valid"])

    def test_batched_edge_relaxation_callback_preserves_unlimited_path(self):
        _, graph, _ = self.fixture()
        load = _RoundLoad(_PathCosts(graph))
        expected = _avoiding_path(graph, "r0", "r1", load, set(), set(), math.inf)
        counts = []
        def record(count):
            counts.append(count)
            return True
        actual = _avoiding_path(graph, "r0", "r1", load, set(), set(), math.inf,
                                consume_relaxations=record)
        self.assertEqual(expected, actual)
        self.assertGreater(sum(counts), 0)
        rejected = _avoiding_path(graph, "r0", "r1", load, set(), set(), math.inf,
                                  consume_relaxations=lambda count: False)
        self.assertIsNone(rejected)

    def test_work_counter_and_safety_are_distinct_stop_reasons(self):
        meter = RouteWorkMeter(RouteWorkBudget(10, 5, safety_s=2), started=100)
        with patch("src.route_work.time.monotonic", return_value=101):
            self.assertTrue(meter.can_start_transfer())
            self.assertTrue(meter.consume_relaxations(3))
            self.assertFalse(meter.consume_relaxations(3))
            meter.finish_transfer(2)
            self.assertFalse(meter.can_start_transfer())
        self.assertEqual(meter.evidence()["stop_reason"], "edge_relaxation_limit")
        meter = RouteWorkMeter(RouteWorkBudget(10, 10), started=100)
        with patch("src.route_work.time.monotonic", return_value=103):
            self.assertFalse(meter.can_start_transfer())
        self.assertEqual(meter.evidence()["stop_reason"], "wall_safety")

    def test_default_keeps_legacy_evidence_and_rejects_invalid_work_limits(self):
        task, graph, baseline = self.fixture()
        _, info = refine_broad(task, graph, baseline)
        self.assertNotIn("work", info)
        for value in (True, -1, 1.5, math.inf, "10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RouteWorkBudget(value, 100)
            with self.subTest(value=value), self.assertRaises(ValueError):
                RouteWorkBudget(100, value)
        for value in (True, 0, -1, math.inf, math.nan, "2"):
            with self.subTest(safety=value), self.assertRaises(ValueError):
                RouteWorkBudget(100, 100, safety_s=value)
        with self.assertRaises(ValueError):
            refine_broad(task, graph, baseline, work_budget={})


if __name__ == "__main__":
    unittest.main()
