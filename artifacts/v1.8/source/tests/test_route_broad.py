"""Broad legal path search preserves model-fixed atomic transfers exactly."""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.route_broad import refine_broad
from src.schedule_edit import extract_problem, apply_proposal
from src.topology import Topology
from tests.test_edit_invariants import actions
from tests.test_route_balance import parallel_case
from tests.test_solver import make_case


class BroadRouteTests(unittest.TestCase):
    def fixture(self, operator="all-gather"):
        task, graph = parallel_case(operator)
        kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
        baseline = build(task, graph, 4, kind=kind, resource_weight=0, packets=1)
        return task, graph, baseline

    def test_improves_both_operators_with_exact_final_report(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline = self.fixture(operator)
                original = deepcopy(baseline)
                result, evidence = refine_broad(task, graph, baseline, time_budget_s=1)
                self.assertTrue(evidence["accepted"], evidence)
                self.assertEqual(evidence["final_replay"], replay(task, graph, result))
                self.assertLess(evidence["after_us"], evidence["before_us"])
                self.assertEqual(actions(result), actions(baseline))
                self.assertEqual(baseline, original)
                self.assertAlmostEqual(result["metrics"]["total_time"] * 1e6, evidence["after_us"])
                # Ranking transfers must not reorder the externally bound list.
                for before, after in zip(baseline["rounds"], result["rounds"]):
                    for old, new in zip(before["transfers"], after["transfers"]):
                        self.assertEqual({k: v for k, v in old.items() if k != "path"},
                                         {k: v for k, v in new.items() if k != "path"})

    def test_model_partial_chunk_boundary_and_exact_frozen_indices_survive(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline = self.fixture(operator)
                problem, context = extract_problem(task, graph, baseline, time_budget_s=1)
                group = next(g for g in context.groups if g.members[0].chunk_count > 1)
                edited, edit_evidence = apply_proposal(task, graph, baseline, context,
                                                      {"base": problem["base"], "edits": [[group.id, 1]]})
                self.assertTrue(edit_evidence["accepted"], edit_evidence)
                frozen = edit_evidence["frozen_transfers"]
                self.assertEqual(len(frozen), 1)
                ri, ti = frozen[0]
                self.assertEqual(len(edited["rounds"][ri]["transfers"][ti]["chunk_ranges"]), 1)
                self.assertEqual(len(edited["rounds"][ri]["transfers"]), len(baseline["rounds"][ri]["transfers"]) + 1)
                original, fixed_copy = deepcopy(edited), deepcopy(frozen)
                result, evidence = refine_broad(task, graph, edited, time_budget_s=1, frozen=frozen)
                self.assertTrue(evidence["frozen_preserved"])
                self.assertEqual(evidence["frozen_count"], 1)
                self.assertEqual(frozen, fixed_copy)
                self.assertEqual(edited, original)
                self.assertEqual(result["rounds"][ri]["transfers"][ti], original["rounds"][ri]["transfers"][ti])
                self.assertEqual(actions(result), actions(baseline))
                self.assertEqual(evidence["final_replay"], replay(task, graph, result))

    def test_fixed_traffic_is_counted_before_assigning_remaining_routes(self):
        task, graph, baseline = self.fixture()
        fixed = [[ri, 0] for ri in range(len(baseline["rounds"]))]
        result, evidence = refine_broad(task, graph, baseline, time_budget_s=1, frozen=fixed)
        self.assertTrue(evidence["accepted"])
        self.assertTrue(evidence["frozen_preserved"])
        for ri, ti in fixed:
            self.assertEqual(result["rounds"][ri]["transfers"][ti], baseline["rounds"][ri]["transfers"][ti])
        self.assertEqual(evidence["final_replay"], replay(task, graph, result))

    def test_all_fixed_or_zero_budget_returns_original_and_valid_final_report(self):
        task, graph, baseline = self.fixture()
        all_fixed = [[ri, ti] for ri, row in enumerate(baseline["rounds"]) for ti in range(len(row["transfers"]))]
        for budget, fixed in ((1, all_fixed), (0, [])):
            with self.subTest(budget=budget):
                with patch("src.route_broad._avoiding_path", side_effect=AssertionError("unexpected route search")):
                    result, evidence = refine_broad(task, graph, baseline, time_budget_s=budget, frozen=fixed)
                self.assertIs(result, baseline)
                self.assertFalse(evidence["accepted"])
                self.assertTrue(evidence["frozen_preserved"])
                self.assertEqual(evidence["final_replay"], replay(task, graph, baseline))

    def test_rejects_invalid_budgets_and_frozen_indices(self):
        task, graph, baseline = self.fixture()
        for value in (True, False, -1, math.inf, math.nan, "0.25", None):
            with self.subTest(budget=value), self.assertRaises(ValueError):
                refine_broad(task, graph, baseline, time_budget_s=value)
        invalid = (None, "0,0", {}, [[True, 0]], [[0, False]], [[0.0, 0]], [[-1, 0]],
                   [[len(baseline["rounds"]), 0]], [[0, 10_000]], [[0]], [[0, 0, 0]], [[0, 0], [0, 0]])
        for value in invalid:
            with self.subTest(frozen=value), self.assertRaises(ValueError):
                refine_broad(task, graph, baseline, frozen=value)
        with self.assertRaises(ValueError):
            refine_broad(task, graph, baseline, allow_blocked_search=1)

    def test_directed_routes_and_shared_resource_multiplicity_remain_legal(self):
        task, raw = make_case(8)
        raw["switches"] = ["fast", "left", "right"]
        raw["edges"] = [{"src": rank, "dst": "fast", "bw": "10GB/s", "bidirectional": True}
                        for rank in task["ranks"]]
        raw["edges"] += [{"src": rank, "dst": "left", "bw": "8GB/s"} for rank in task["ranks"][:4]]
        raw["edges"] += [{"src": "left", "dst": "right", "bw": "8GB/s"}]
        raw["edges"] += [{"src": "right", "dst": rank, "bw": "8GB/s"} for rank in task["ranks"][4:]]
        refs = [{"src": rank, "dst": "fast", "bidirectional": True} for rank in task["ranks"]]
        raw["resources"] = [{"id": "pool-a", "capacity": "1GB/s", "edges": refs},
                            {"id": "pool-b", "capacity": "2GB/s", "edges": deepcopy(refs)}]
        graph = Topology.from_data(raw)
        baseline = build(task, graph, 4, resource_weight=0)
        result, evidence = refine_broad(task, graph, baseline, time_budget_s=1)
        self.assertTrue(evidence["accepted"], evidence)
        self.assertGreater(evidence["blocked_searches"], 0)
        self.assertEqual(evidence["final_replay"], replay(task, graph, result))
        for round_ in result["rounds"]:
            for transfer in round_["transfers"]:
                path = transfer["path"]
                self.assertTrue(all(edge in graph.edges for edge in zip(path, path[1:])))
                self.assertTrue(all(node in graph.switches for node in path[1:-1]))

    def test_invalid_baseline_is_not_returned_as_verified(self):
        task, graph, baseline = self.fixture()
        baseline["rounds"][0]["transfers"][0]["chunk_ranges"] = ["invented"]
        for budget in (0, 1):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "baseline is invalid"):
                refine_broad(task, graph, baseline, time_budget_s=budget)


if __name__ == "__main__":
    unittest.main()
