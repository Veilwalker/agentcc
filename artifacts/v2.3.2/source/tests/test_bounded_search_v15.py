"""Work caps, immutable structure reuse and request-local model decisions."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from src import solver
from src.collectives import build
from src.llm import edit_schedule
from src.replay import replay
from src.route_balance import refine
from tests import test_llm_edits as fixtures
from tests.test_route_balance import parallel_case


class BoundedSearchTests(unittest.TestCase):
    def test_fixed_queries_ignore_nominal_soft_time_but_zero_disables(self):
        task, graph = parallel_case("all-reduce")
        seed = build(task, graph, 4, kind="xor_tree", resource_weight=0)
        original = deepcopy(seed)
        results = []
        for soft_time in (1e-9, 10):
            result, info = refine(task, graph, seed, time_budget_s=soft_time,
                                  max_route_queries=40, max_path_scores=200)
            self.assertLessEqual(info["work"]["route_queries"], 40)
            self.assertTrue(replay(task, graph, result)["valid"])
            results.append(result)
        self.assertEqual(results[0], results[1])
        result, info = refine(task, graph, seed, time_budget_s=0,
                              max_route_queries=40, max_path_scores=200)
        self.assertEqual(result, seed)
        self.assertEqual(info["work"]["route_queries"], 0)
        self.assertEqual(seed, original)

    def test_routing_candidates_reuse_expressions_without_mutating_input(self):
        task, graph = parallel_case("all-reduce")
        before = deepcopy(task)
        with patch.object(solver, "build_collective", wraps=solver.build_collective) as constructor:
            result = solver.solve_schedule(task, graph, max_candidates=10)
        reused = [row for row in result["strategy"]["candidates"] if row.get("structure_reused")]
        self.assertTrue(reused)
        built = [row for row in result["strategy"]["candidates"] if row.get("constructed")]
        self.assertLess(constructor.call_count, len(built))
        self.assertTrue(replay(task, graph, result)["valid"])
        self.assertEqual(task, before)

    def test_candidate_limit_keeps_a_valid_incumbent(self):
        task, graph = parallel_case("all-reduce")
        seed = solver.solve_schedule(task, graph, seed_only=True, seed_split=4)
        original = deepcopy(seed)
        result = solver.solve_schedule(task, graph, incumbent=seed, max_candidates=0)
        self.assertEqual(result["strategy"]["admitted_candidates"], 0)
        self.assertTrue(replay(task, graph, result)["valid"])
        self.assertLessEqual(result["metrics"]["total_time"], seed["metrics"]["total_time"])
        self.assertEqual(seed, original)


class RequestBindingTests(unittest.TestCase):
    @patch("src.llm.build_opener")
    def test_model_counts_are_bound_to_each_private_request_without_invention(self, factory):
        fixture = fixtures.EditModelTests()
        fixture.setUp()
        for base, count in (("first-request", 1), ("second-request", 2)):
            fixture.problem["base"] = base
            raw = json.dumps({"counts": [count]})
            fixture.response(factory, raw)
            proposal, evidence = edit_schedule(fixture.task, fixture.problem, fixture.settings)
            self.assertEqual(proposal, {"base": base, "edits": [["g0", count]]})
            self.assertEqual(evidence["response_wire_proposal"], {"counts": [count]})
            self.assertEqual(evidence["response_evidence"]["content"], raw)
            self.assertEqual(evidence["base_binding"], "request_context")
            self.assertNotIn(base, evidence["request_evidence"]["messages"][1]["content"])

    @patch("src.llm.build_opener")
    def test_malformed_reasoning_metadata_cannot_break_a_valid_reply(self, factory):
        fixture = fixtures.EditModelTests()
        fixture.setUp()
        response = fixture.response(factory, '{"counts":[1]}')
        body = json.loads(response.read.return_value)
        body["choices"][0]["message"]["reasoning_content"] = 7
        response.read.return_value = json.dumps(body).encode()
        proposal, evidence = edit_schedule(fixture.task, fixture.problem, fixture.settings)
        self.assertEqual(proposal["edits"], [["g0", 1]])
        self.assertIsNone(evidence["reasoning_evidence"]["characters"])
        self.assertFalse(evidence["reasoning_evidence"]["absence_proves_disabled"])


if __name__ == "__main__":
    unittest.main()
