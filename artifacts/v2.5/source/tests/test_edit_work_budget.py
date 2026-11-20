"""Fixed edit-extraction work stabilizes the model's concrete action catalog."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.route_balance import _PathCosts, _RoundLoad
from src.route_work import RouteWorkBudget
from src.schedule_edit import _path_options, apply_proposal, extract_problem
from tests.test_route_balance import parallel_case


class EditWorkBudgetTests(unittest.TestCase):
    def fixture(self, operator="all-gather"):
        task, graph = parallel_case(operator)
        kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
        baseline = build(task, graph, 4, kind=kind, resource_weight=0)
        return task, graph, baseline, replay(task, graph, baseline)

    def test_repeated_catalog_binding_and_work_are_identical(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline, report = self.fixture(operator)
                original = deepcopy(baseline)
                rows = [extract_problem(task, graph, baseline, validated_report=report,
                                         work_budget=RouteWorkBudget(13, 10_000)) for _ in range(3)]
                self.assertEqual(rows[0], rows[1])
                self.assertEqual(rows[1], rows[2])
                problem, context = rows[0]
                self.assertTrue(problem["groups"])
                self.assertNotIn("extraction_work", problem)
                self.assertEqual(context.extraction_work["stop_reason"], "path_score_limit")
                proposal = {"base": problem["base"], "edits": [[problem["groups"][0]["id"], 1]]}
                result, evidence = apply_proposal(task, graph, baseline, context, proposal)
                self.assertEqual(evidence["errors"], [])
                self.assertTrue(replay(task, graph, result)["valid"])
                self.assertEqual(baseline, original)

    def test_zero_time_disables_extraction_even_with_work_available(self):
        task, graph, baseline, report = self.fixture()
        with patch("src.schedule_edit._path_options", side_effect=AssertionError("unexpected route search")):
            problem, context = extract_problem(task, graph, baseline, time_budget_s=0,
                                               validated_report=report,
                                               work_budget=RouteWorkBudget(10_000, 10_000))
        self.assertEqual(problem["groups"], [])
        self.assertEqual(context.extraction_work["completed_transfers"], 0)
        self.assertEqual(context.extraction_work["stop_reason"], "disabled_time_budget")

    def test_complete_transfer_is_catalogued_before_score_limit_stops(self):
        task, graph, baseline, report = self.fixture()
        problem, context = extract_problem(task, graph, baseline, validated_report=report,
                                           work_budget=RouteWorkBudget(1, 10_000))
        self.assertEqual(context.extraction_work["completed_transfers"], 1)
        self.assertEqual(len(context.groups), 1)
        member = context.groups[0].members[0]
        transfer = baseline["rounds"][context.groups[0].round_index]["transfers"][member.transfer_index]
        self.assertEqual(member.chunk_count, len(transfer["chunk_ranges"]))
        self.assertEqual(problem["groups"][0]["max_chunks"], member.chunk_count)

    def test_path_search_receives_optional_relaxation_callback(self):
        _, graph, baseline, _ = self.fixture()
        costs = _PathCosts(graph)
        load = _RoundLoad(costs)
        transfer = baseline["rounds"][0]["transfers"][0]
        outgoing = {transfer["src"]: set()}
        incoming = {transfer["dst"]: set()}
        consume = lambda count: True
        with patch("src.schedule_edit._avoiding_path", return_value=None) as search:
            _path_options(graph, transfer, load, costs, outgoing, incoming, float("inf"),
                          (set(), set()), consume_relaxations=consume)
        self.assertIs(search.call_args.kwargs["consume_relaxations"], consume)

    def test_default_context_has_no_work_metadata_and_invalid_budget_is_rejected(self):
        task, graph, baseline, report = self.fixture()
        _, context = extract_problem(task, graph, baseline, validated_report=report)
        self.assertIsNone(context.extraction_work)
        with self.assertRaises(ValueError):
            extract_problem(task, graph, baseline, validated_report=report, work_budget={})


if __name__ == "__main__":
    unittest.main()
