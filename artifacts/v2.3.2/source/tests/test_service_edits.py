"""Actual edit compilation and final selection at the service boundary."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.config import Settings
from src.replay import replay
from src.service import solve_input
from src.schedule_edit import extract_problem
from src.llm import compact_edit_problem
from src.topology import Topology
from tests.test_route_refinement_audit import congested_fixture


SETTINGS = Settings("http://unused.test/v1/chat/completions", "synthetic-key", "test-model")


class ServiceEditTests(unittest.TestCase):
    def test_empty_extracted_window_keeps_current_graph_in_model_input(self):
        task, topology, baseline = congested_fixture("all-gather")
        graph = Topology.from_data(topology)
        problem, _ = extract_problem(task, graph, baseline, time_budget_s=0)
        projected = compact_edit_problem(problem)
        self.assertFalse(projected["groups"])
        summary = projected["topology"]
        self.assertEqual(summary["ranks"], len(task["ranks"]))
        self.assertTrue(summary["edge_examples"])
        self.assertTrue(summary["tight_pools"][0]["member_samples"])

    def run_case(self, amount, *, control=True):
        task, topology, baseline = congested_fixture("all-gather")
        original = deepcopy(baseline)

        def editor(task, problem, **kwargs):
            edits = [[row["id"], amount(row)] for row in problem["groups"]]
            return {"base": problem["base"], "edits": edits}, {
                "called": True, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        with patch("src.service.load_settings", return_value=SETTINGS), \
                patch("src.service.solve_schedule", return_value=baseline), \
                patch("src.service.edit_schedule", side_effect=editor):
            if control:
                output = solve_input({"task": task, "topology": topology}, refinement_budget_s=0)
            else:
                with patch("src.service._refined_candidate", side_effect=lambda task, graph, initial: deepcopy(initial)):
                    output = solve_input({"task": task, "topology": topology}, refinement_budget_s=0)
        self.assertEqual(baseline, original, "the edit base must remain immutable")
        checked = replay(task, topology, output["answer"])
        self.assertTrue(checked["valid"], checked)
        self.assertAlmostEqual(output["answer"]["metrics"]["total_time"] * 1e6, checked["total_time_us"])
        self.assertEqual(sum(row["selected"] for row in output["diagnostics"]["search"]["candidates"]), 1)
        return output, checked

    def test_model_edit_is_compiled_into_the_returned_schedule(self):
        output, checked = self.run_case(lambda row: max(1, row["max_chunks"] // 2), control=False)
        search = output["diagnostics"]["search"]
        edit = search["model_edit"]
        self.assertTrue(edit["accepted"], edit)
        self.assertTrue(edit["selected"], edit)
        selected = next(row for row in search["candidates"] if row["selected"])
        self.assertEqual(selected["source"], "model_edit")
        self.assertEqual(selected["actual"]["edits"], edit["applied_edits"])
        self.assertAlmostEqual(edit["after_us"], checked["total_time_us"])
        self.assertTrue(edit["edit_manifest"])

    def test_better_control_wins_without_claiming_model_adoption(self):
        output, checked = self.run_case(lambda row: 1)
        search = output["diagnostics"]["search"]
        edit = search["model_edit"]
        self.assertTrue(edit["accepted"], edit)
        self.assertFalse(edit["selected"], edit)
        self.assertLess(checked["total_time_us"], edit["after_us"])
        self.assertEqual(next(row["source"] for row in search["candidates"] if row["selected"]), "route_refinement")

    def test_no_op_keeps_the_control_and_records_the_rejection(self):
        output, _ = self.run_case(lambda row: 0)
        edit = output["diagnostics"]["search"]["model_edit"]
        self.assertFalse(edit["accepted"])
        self.assertFalse(edit["selected"])
        self.assertEqual(edit["applied_edits"], [])
        self.assertTrue(edit["errors"])

    def test_bad_model_json_preserves_parse_evidence_and_valid_answer(self):
        task, topology, baseline = congested_fixture("all-gather")
        metadata = {"called": True, "proposal_json_valid": False,
                    "proposal_parse_error": "synthetic malformed JSON",
                    "response_evidence": {"content": "broken"}}
        with patch("src.service.load_settings", return_value=SETTINGS), \
                patch("src.service.solve_schedule", return_value=baseline), \
                patch("src.service.edit_schedule", return_value=(None, metadata)):
            output = solve_input({"task": task, "topology": topology}, refinement_budget_s=0)
        self.assertTrue(replay(task, topology, output["answer"])["valid"])
        self.assertEqual(output["diagnostics"]["llm"]["response_evidence"]["content"], "broken")
        self.assertFalse(output["diagnostics"]["search"]["model_edit"]["accepted"])


if __name__ == "__main__":
    unittest.main()
