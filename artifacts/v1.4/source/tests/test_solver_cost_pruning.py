"""Cost-only pruning never promotes an unverified communication schedule."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src import solver
from src.collectives import build
from src.replay import replay
from src.topology import Topology
from tests.test_route_balance import parallel_case
from tests.test_solver import make_case


class SolverCostPruningTests(unittest.TestCase):
    def test_fast_cost_matches_replay_including_shared_resource_multiplicity(self):
        for operator, kind in (("all-gather", "reverse_doubling"), ("all-reduce", "xor_tree")):
            task, graph = parallel_case(operator)
            cost = solver._ConstructedScheduleCost(task, graph)
            for split in (1, 2, 4):
                with self.subTest(operator=operator, split=split):
                    solution = build(task, graph, split, kind=kind, resource_weight=0, packets=2)
                    checked = replay(task, graph, solution)
                    total, rounds = cost.evaluate(solution)
                    self.assertTrue(checked["valid"], checked)
                    self.assertAlmostEqual(total, checked["total_time_us"])
                    self.assertEqual(len(rounds), len(checked["round_times_us"]))
                    for actual, expected in zip(rounds, checked["round_times_us"]):
                        self.assertAlmostEqual(actual, expected)

    def test_nonwinning_candidates_are_explicitly_unverified(self):
        task, topology = make_case(8, "all-reduce")
        with patch.object(solver, "replay", wraps=replay) as checked:
            result = solver.solve_schedule(task, topology, time_budget_s=3)
        records = result["strategy"]["candidates"]
        skipped = [row for row in records if row.get("cost_only")]
        self.assertTrue(skipped)
        self.assertTrue(records[0]["valid"])
        self.assertEqual(result["strategy"]["candidates_replayed"], checked.call_count)
        for record in skipped:
            self.assertIsNone(record["valid"])
            self.assertTrue(record["constructed"])
            self.assertFalse(record["selected"])
            self.assertGreaterEqual(record["total_time_us"], result["metrics"]["total_time"] * 1e6)
        selected = [row for row in records if row["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0]["valid"], True)
        self.assertFalse(selected[0].get("cost_only", False))
        self.assertTrue(replay(task, topology, result)["valid"])

    def test_model_proposal_is_replayed_even_when_cost_cannot_win(self):
        task, topology = make_case(8, "all-reduce")
        hint = {"algorithm": "xor_tree", "split": 4, "rank_order": "topology", "resource_weight": 2.0}
        result = solver.solve_schedule(task, topology, hint, time_budget_s=3)
        model = [row for row in result["strategy"]["candidates"] if row["source"].startswith("model")]
        self.assertEqual({row["source"] for row in model}, {"model_seed", "model"})
        self.assertTrue(all(row["valid"] is True and not row.get("cost_only") for row in model))

    def test_promising_but_semantically_invalid_candidate_is_never_selected(self):
        task, topology = make_case(8, "all-reduce")
        original = solver.build_collective
        injected = []

        def corrupt_later(*args, **kwargs):
            candidate = original(*args, **kwargs)
            if args[2] > 1:
                candidate = deepcopy(candidate)
                candidate["rounds"][0]["transfers"][0]["chunk_ranges"] = ["invented"]
                injected.append(True)
            return candidate

        # A cheap estimated cost is insufficient: replay must reject corruption.
        with patch.object(solver, "build_collective", side_effect=corrupt_later), \
                patch.object(solver._ConstructedScheduleCost, "evaluate", return_value=(0.0, [])):
            result = solver.solve_schedule(task, topology, time_budget_s=3)
        self.assertTrue(injected)
        failures = [row for row in result["strategy"]["candidates"] if row.get("errors")]
        self.assertTrue(failures)
        self.assertTrue(all(row["valid"] is False for row in failures))
        self.assertTrue(replay(task, topology, result)["valid"])

    def test_relay_fallback_always_uses_semantic_replay(self):
        task, topology = make_case(5, "all-reduce", direct=True)
        with patch.object(solver._ConstructedScheduleCost, "evaluate", side_effect=AssertionError("fallback cannot be cost-pruned")):
            result = solver.solve_schedule(task, topology, time_budget_s=3)
        self.assertTrue(replay(task, topology, result)["valid"])
        selected = next(row for row in result["strategy"]["candidates"] if row["selected"])
        self.assertEqual(selected["actual"]["algorithm"], "rank_relay")
        self.assertTrue(selected["valid"])

    def test_unexpected_cost_filter_input_falls_back_to_semantic_replay(self):
        task, topology = make_case(8, "all-reduce")
        with patch.object(solver._ConstructedScheduleCost, "evaluate", side_effect=KeyError("missing path")), \
                patch.object(solver, "replay", wraps=replay) as checked:
            result = solver.solve_schedule(task, topology, time_budget_s=3)
        self.assertTrue(replay(task, topology, result)["valid"])
        self.assertFalse(any(row.get("cost_only") for row in result["strategy"]["candidates"]))
        self.assertEqual(result["strategy"]["candidates_replayed"], checked.call_count)

    def test_constructed_weights_include_cost_only_candidates(self):
        task, graph = parallel_case("all-reduce")
        result = solver.solve_schedule(task, graph, time_budget_s=3)
        evidence = result["strategy"]
        expected = {row["actual"]["resource_weight"] for row in evidence["candidates"] if row.get("constructed")}
        self.assertEqual(evidence["constructed_route_weights"], sorted(expected))
        self.assertTrue(any(row.get("cost_only") for row in evidence["candidates"]))


if __name__ == "__main__":
    unittest.main()
