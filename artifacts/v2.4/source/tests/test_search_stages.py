"""Protected first-stage construction and resumed search keep exact evidence."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src import solver
from src.replay import replay
from tests.test_solver import make_case


class SearchStageTests(unittest.TestCase):
    def test_seed_only_uses_legal_requested_split_without_optional_work(self):
        for operator in ("all-gather", "all-reduce"):
            task, topology = make_case(8, operator)
            with self.subTest(operator=operator), \
                    patch.object(solver, "locality_order", side_effect=AssertionError("optional locality work")), \
                    patch.object(solver, "build_collective", wraps=solver.build_collective) as build:
                answer = solver.solve_schedule(task, topology, seed_only=True, seed_split=4)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(answer["split"], 4)
            self.assertEqual(len(answer["strategy"]["candidates"]), 1)
            self.assertEqual(answer["strategy"]["alternate_route_probes"], {})
            self.assertEqual(answer["strategy"]["validation_report"], replay(task, topology, answer))

    def test_explicit_topology_hint_remains_effective_in_seed_stage(self):
        task, topology = make_case(8, "all-reduce")
        hint = {"rank_order": "topology", "resource_weight": 2, "split": 4}
        with patch.object(solver, "locality_order", wraps=solver.locality_order) as order:
            answer = solver.solve_schedule(task, topology, hint, seed_only=True, seed_split=4)
        order.assert_called_once()
        record = answer["strategy"]["candidates"][0]
        self.assertEqual(record["source"], "model")
        self.assertEqual(record["actual"]["rank_order"], "topology")
        self.assertEqual(record["actual"]["resource_weight"], 2)
        self.assertTrue(record["valid"])

    def test_resume_does_not_rebuild_revalidated_seed(self):
        task, topology = make_case(8, "all-reduce")
        initial = solver.solve_schedule(task, topology, seed_only=True, seed_split=4)
        snapshot = deepcopy(initial)
        with patch.object(solver, "build_collective", wraps=solver.build_collective) as build:
            resumed = solver.solve_schedule(task, topology, incumbent=initial)
        attempted = [(call.args[2], call.kwargs["resource_weight"]) for call in build.call_args_list]
        self.assertNotIn((4, 1.0), attempted)
        self.assertNotIn((1, 1.0), attempted)
        self.assertTrue(resumed["strategy"]["candidates"][0]["incumbent_revalidated"])
        self.assertEqual(resumed["strategy"]["validation_report"], replay(task, topology, resumed))
        self.assertEqual(initial, snapshot)

    def test_completed_search_reuses_cost_rejections_without_rebuilding_them(self):
        task, topology = make_case(8, "all-reduce")
        initial = solver.solve_schedule(task, topology)
        self.assertTrue(any(record.get("cost_only") for record in initial["strategy"]["candidates"]))
        with patch.object(solver, "build_collective", side_effect=AssertionError("duplicate construction")):
            resumed = solver.solve_schedule(task, topology, incumbent=initial)
        self.assertEqual(resumed["strategy"]["validation_report"], replay(task, topology, resumed))
        self.assertEqual(resumed["metrics"], initial["metrics"])
        self.assertEqual(len(resumed["strategy"]["candidates"]), len(initial["strategy"]["candidates"]))

    def test_invalid_incumbent_cannot_bypass_protected_seed(self):
        task, topology = make_case(8)
        invalid = solver.solve_schedule(task, topology, seed_only=True)
        invalid["rounds"][0]["transfers"][0]["chunk_ranges"] = ["unknown-chunk"]
        with patch.object(solver, "build_collective", wraps=solver.build_collective) as build:
            answer = solver.solve_schedule(task, topology, incumbent=invalid, seed_only=True)
        build.assert_called_once()
        self.assertEqual(answer["strategy"]["validation_report"], replay(task, topology, answer))
        self.assertTrue(answer["strategy"]["validation_report"]["valid"])

    def test_intermediate_route_penalties_are_available_as_candidates(self):
        task, topology = make_case(8, "all-reduce")
        answer = solver.solve_schedule(task, topology)
        self.assertEqual(set(answer["strategy"]["constructed_route_weights"]), {0, 1, 2, 3, 4})
        self.assertTrue(answer["strategy"]["validation_report"]["valid"])

    def test_seed_options_are_strictly_validated(self):
        task, topology = make_case(8)
        for split in (True, False, 0, -1, 3, 4.0, "4"):
            with self.subTest(split=split), self.assertRaises(solver.SolverError):
                solver.solve_schedule(task, topology, seed_split=split)
        for option in (0, 1, None, "yes"):
            with self.subTest(seed_only=option), self.assertRaises(solver.SolverError):
                solver.solve_schedule(task, topology, seed_only=option)
        for weight in (True, False, -1, 5, float("nan"), float("inf"), "4"):
            with self.subTest(seed_weight=weight), self.assertRaises(solver.SolverError):
                solver.solve_schedule(task, topology, seed_weight=weight)
        with self.assertRaises(solver.SolverError):
            solver.solve_schedule(task, topology, {"resource_weight": 2}, seed_weight=4)

    def test_explicit_seed_weight_is_not_misreported_as_a_model_decision(self):
        task, topology = make_case(8, "all-reduce")
        answer = solver.solve_schedule(task, topology, seed_only=True, seed_split=4, seed_weight=4)
        evidence = answer["strategy"]
        self.assertEqual(evidence["model_requested"], {})
        self.assertIsNone(evidence["model_adopted"])
        self.assertEqual(evidence["candidates"][0]["source"], "baseline")
        self.assertEqual(evidence["candidates"][0]["actual"]["resource_weight"], 4)
        self.assertTrue(evidence["validation_report"]["valid"])

    def test_core_routing_alternatives_precede_rank_reordering(self):
        task, topology = make_case(8, "all-reduce")
        answer = solver.solve_schedule(task, topology)
        core = [record["actual"] for record in answer["strategy"]["candidates"]
                if record["source"] == "local_search"]
        at_four = next(i for i, params in enumerate(core)
                       if params["resource_weight"] == 4 and params["rank_order"] == "input")
        at_topology = next((i for i, params in enumerate(core) if params["rank_order"] == "topology"), len(core))
        self.assertLess(at_four, at_topology)


if __name__ == "__main__":
    unittest.main()
