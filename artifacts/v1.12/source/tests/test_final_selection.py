"""Model edits compete with a fully refined, unrestricted deterministic answer."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src import service
from src.config import Settings
from src.replay import replay
from src.route_broad import refine_broad
from src.schedule_edit import apply_proposal
from tests.test_edit_invariants import actions
from tests.test_route_refinement_audit import congested_fixture


SETTINGS = Settings("http://unused.test/v1/chat/completions", "synthetic-only", "synthetic-model")


class FinalSelectionTests(unittest.TestCase):
    def run_service(self, operator="all-gather", mode="small", control=None, finish=None):
        task, topology, baseline = congested_fixture(operator)
        original = deepcopy(baseline)
        captured = {"routing_calls": []}

        def editor(_task, problem, **_kwargs):
            if mode == "invalid":
                proposal = None
            elif mode == "noop":
                proposal = {"base": problem["base"], "edits": []}
            else:
                proposal = {"base": problem["base"], "edits": [
                    [group["id"], group["max_chunks"] if mode == "all" else 1]
                    for group in problem["groups"]]}
            return proposal, {"called": True, "synthetic": True, "usage": {}}

        def compile_edit(*args, **kwargs):
            result, evidence = apply_proposal(*args, **kwargs)
            captured["compiled"] = deepcopy(result)
            captured["compile_evidence"] = deepcopy(evidence)
            return result, evidence

        def finish_routes(*args, **kwargs):
            captured["routing_calls"].append(deepcopy(kwargs))
            result, details = (finish or refine_broad)(*args, **kwargs)
            captured["control"] = deepcopy(result)
            captured["control_report"] = deepcopy(details["final_replay"])
            return result, details

        with patch("src.service.load_settings", return_value=SETTINGS), \
                patch("src.service.solve_schedule", return_value=baseline), \
                patch("src.service.edit_schedule", side_effect=editor), \
                patch("src.service._refined_candidate", side_effect=lambda _task, _graph, initial: deepcopy(initial)), \
                patch("src.service.apply_proposal", side_effect=compile_edit), \
                patch("src.service.refine_broad", side_effect=finish_routes):
            if control:
                with patch("src.service._control_candidate", side_effect=control):
                    result = service.solve_input({"task": task, "topology": topology}, refinement_budget_s=1)
            else:
                result = service.solve_input({"task": task, "topology": topology}, refinement_budget_s=1)
        self.assertEqual(baseline, original, "the frozen edit base must remain immutable")
        checked = replay(task, topology, result["answer"])
        self.assertTrue(checked["valid"], checked)
        self.assertEqual(result["diagnostics"]["local_replay"], checked)
        self.assertEqual(actions(result["answer"]), actions(baseline))
        self.assertAlmostEqual(result["answer"]["metrics"]["total_time"] * 1e6, checked["total_time_us"])
        winners = [record for record in result["diagnostics"]["search"]["candidates"] if record["selected"]]
        self.assertEqual(len(winners), 1)
        self.assertAlmostEqual(winners[0]["total_time_us"], checked["total_time_us"])
        return result, captured

    def test_strong_control_is_never_worsened_to_keep_model_actions(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                result, captured = self.run_service(operator)
                edit = result["diagnostics"]["search"]["model_edit"]
                self.assertTrue(edit["accepted"], "fixture must contain a valid improving model edit")
                self.assertFalse(edit["selected"], "a weaker model proposal must lose to full routing")
                self.assertFalse(edit["preserved_in_final"])
                self.assertLessEqual(result["diagnostics"]["local_replay"]["total_time_us"],
                                     captured["control_report"]["total_time_us"])
                self.assertTrue(captured["routing_calls"])
                self.assertTrue(all(not call["frozen"] for call in captured["routing_calls"]))
                self.assertEqual(result["answer"]["rounds"], captured["control"]["rounds"])

    def test_better_model_still_changes_the_final_schedule(self):
        def unchanged(task, graph, base, **_kwargs):
            return deepcopy(base), replay(task, graph, base), 0.0

        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                result, captured = self.run_service(operator, control=unchanged)
                edit = result["diagnostics"]["search"]["model_edit"]
                self.assertTrue(edit["accepted"] and edit["selected"])
                self.assertEqual(edit["preservation_scope"], "observed_not_enforced")
                self.assertEqual(edit["selected_frozen_transfers"], [])
                self.assertTrue(edit["preserved_in_final"])
                self.assertTrue(edit["selected_edits"])
                self.assertEqual(result["answer"]["rounds"], captured["compiled"]["rounds"])
                self.assertLess(edit["final_time_us"], edit["control_time_us"])

    def test_invalid_and_noop_models_keep_valid_control_without_false_adoption(self):
        for mode in ("invalid", "noop"):
            with self.subTest(mode=mode):
                result, captured = self.run_service(mode=mode)
                edit = result["diagnostics"]["search"]["model_edit"]
                self.assertFalse(edit["accepted"] or edit["selected"] or edit["preserved_in_final"])
                self.assertEqual(result["answer"]["rounds"], captured["control"]["rounds"])
                self.assertEqual(edit["selected_edits"], [])

    def test_optional_control_routing_failure_still_allows_verified_model_candidate(self):
        def failed_finish(*_args, **_kwargs):
            raise ValueError("synthetic optional routing failure")

        result, captured = self.run_service(finish=failed_finish)
        search = result["diagnostics"]["search"]
        self.assertFalse(search["route_finish"]["accepted"])
        self.assertTrue(search["model_edit"]["selected"])
        self.assertEqual(result["answer"]["rounds"], captured["compiled"]["rounds"])

    def test_rejected_control_candidate_does_not_supply_the_final_report(self):
        calls = []

        def reject_first(task, graph, candidate):
            calls.append(deepcopy(candidate["rounds"]))
            if len(calls) == 1:
                return {"valid": False, "errors": ["synthetic rejection"], "total_time_us": 0,
                        "round_times_us": [], "nonempty_rounds": 0, "split": candidate["split"]}
            return replay(task, graph, candidate)

        with patch("src.route_broad.replay", side_effect=reject_first):
            result, captured = self.run_service(mode="noop")
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])
        self.assertEqual(result["answer"]["rounds"], calls[1])
        self.assertEqual(result["diagnostics"]["local_replay"], captured["control_report"])


if __name__ == "__main__":
    unittest.main()
