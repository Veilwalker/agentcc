"""End-to-end selection and exact model-prefix preservation after final routing.

Models are synthetic, but extraction, edit compilation, broad routing and the
final symbolic Replay are real. No credentials or network are used.
"""

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
    def run_service(self, operator="all-gather", mode="small", finish=None):
        task, topology, baseline = congested_fixture(operator)
        original = deepcopy(baseline)
        captured = {}

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
            captured["finish_input"] = deepcopy(args[2])
            captured["frozen"] = deepcopy(kwargs["frozen"])
            result, details = (finish or refine_broad)(*args, **kwargs)
            captured["finish_details"] = deepcopy(details)
            return result, details

        with patch("src.service.load_settings", return_value=SETTINGS), \
                patch("src.service.solve_schedule", return_value=baseline), \
                patch("src.service.edit_schedule", side_effect=editor), \
                patch("src.service._refined_candidate", side_effect=lambda _task, _graph, initial: deepcopy(initial)), \
                patch("src.service.apply_proposal", side_effect=compile_edit), \
                patch("src.service.refine_broad", side_effect=finish_routes):
            result = service.solve_input({"task": task, "topology": topology}, refinement_budget_s=1)
        self.assertEqual(baseline, original, "neither routing stage may mutate the frozen base")
        checked = replay(task, topology, result["answer"])
        self.assertTrue(checked["valid"], checked)
        self.assertEqual(result["diagnostics"]["local_replay"], checked)
        self.assertEqual(actions(result["answer"]), actions(baseline))
        self.assertAlmostEqual(result["answer"]["metrics"]["total_time"] * 1e6, checked["total_time_us"])
        search = result["diagnostics"]["search"]
        winners = [record for record in search["candidates"] if record["selected"]]
        self.assertEqual(len(winners), 1)
        self.assertAlmostEqual(winners[0]["total_time_us"], checked["total_time_us"])
        return result, captured

    def test_actual_final_routing_preserves_each_model_transfer_for_both_operators(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                result, captured = self.run_service(operator)
                search = result["diagnostics"]["search"]
                edit, finish = search["model_edit"], search["route_finish"]
                self.assertTrue(edit["accepted"] and edit["selected"])
                self.assertTrue(edit["preserved_in_final"])
                self.assertTrue(finish["accepted"], "fixture must execute an actual later route change")
                self.assertGreater(finish["changed_transfers"], 0)
                self.assertTrue(captured["frozen"])
                for ri, ti in captured["frozen"]:
                    self.assertEqual(result["answer"]["rounds"][ri]["transfers"][ti],
                                     captured["compiled"]["rounds"][ri]["transfers"][ti])
                self.assertEqual(finish["frozen_count"], len(captured["frozen"]))
                self.assertLess(edit["final_time_us"], edit["after_us"])
                selected = next(record for record in search["candidates"] if record["selected"])
                self.assertEqual(selected["source"], "route_finish")
                parent = search["candidates"][selected["parent_candidate"]]
                self.assertEqual(parent["source"], "model_edit")
                self.assertFalse(parent["selected"])
                self.assertEqual(selected["actual"]["fixed_model_transfers"], len(captured["frozen"]))
                self.assertNotIn("final_replay", finish)
                self.assertEqual(result["diagnostics"]["local_replay"],
                                 captured["finish_details"]["final_replay"])

    def test_all_frozen_actions_survive_noop_final_routing(self):
        result, captured = self.run_service(mode="all")
        search = result["diagnostics"]["search"]
        total_transfers = sum(len(round_["transfers"]) for round_ in result["answer"]["rounds"])
        self.assertEqual(len(captured["frozen"]), total_transfers)
        self.assertFalse(search["route_finish"]["accepted"])
        self.assertTrue(search["model_edit"]["preserved_in_final"])
        self.assertEqual(next(record["source"] for record in search["candidates"] if record["selected"]), "model_edit")

    def test_invalid_and_noop_models_do_not_claim_preservation_of_control_edits(self):
        for mode in ("invalid", "noop"):
            with self.subTest(mode=mode):
                result, captured = self.run_service(mode=mode)
                search = result["diagnostics"]["search"]
                self.assertFalse(search["model_edit"]["accepted"])
                self.assertFalse(search["model_edit"]["selected"])
                self.assertFalse(search["model_edit"]["preserved_in_final"])
                self.assertEqual(captured["frozen"], [])
                self.assertTrue(search["route_finish"]["accepted"])
                self.assertEqual(next(record["source"] for record in search["candidates"] if record["selected"]), "route_finish")

    def test_final_routing_exception_keeps_compiled_model_answer_and_fresh_replay(self):
        def failed_finish(*_args, **_kwargs):
            raise ValueError("synthetic final routing failure")

        with patch("src.service.replay", wraps=replay) as final_check:
            result, captured = self.run_service(finish=failed_finish)
        self.assertEqual(final_check.call_args.args[2]["rounds"], result["answer"]["rounds"])
        search = result["diagnostics"]["search"]
        self.assertFalse(search["route_finish"]["accepted"])
        self.assertTrue(search["model_edit"]["selected"])
        self.assertTrue(search["model_edit"]["preserved_in_final"])
        self.assertEqual(result["answer"]["rounds"], captured["compiled"]["rounds"])
        self.assertEqual(next(record["source"] for record in search["candidates"] if record["selected"]), "model_edit")

    def test_rejected_final_candidate_returns_original_report_not_rejected_report(self):
        calls = []

        def reject_first_candidate(task, graph, candidate):
            calls.append(deepcopy(candidate["rounds"]))
            if len(calls) == 1:
                return {"valid": False, "errors": ["synthetic candidate rejection"],
                        "total_time_us": 0, "round_times_us": [], "nonempty_rounds": 0, "split": candidate["split"]}
            return replay(task, graph, candidate)

        with patch("src.route_broad.replay", side_effect=reject_first_candidate):
            result, captured = self.run_service()
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1], "fixture must have proposed a different final candidate")
        self.assertEqual(result["answer"]["rounds"], captured["compiled"]["rounds"])
        search = result["diagnostics"]["search"]
        self.assertFalse(search["route_finish"]["accepted"])
        self.assertTrue(search["model_edit"]["preserved_in_final"])
        self.assertEqual(next(record["source"] for record in search["candidates"] if record["selected"]), "model_edit")

    def test_complementary_model_and_control_rounds_beat_both_whole_candidates(self):
        for empty_round in (False, True):
            with self.subTest(empty_round=empty_round):
                task, topology, baseline = congested_fixture("all-gather")
                if empty_round:
                    baseline["rounds"].insert(0, {"round": 1, "transfers": []})
                    for index, round_ in enumerate(baseline["rounds"], 1):
                        round_["round"] = index
                    baseline["round_count"] = len(baseline["rounds"])
                control = deepcopy(baseline)
                for transfer in control["rounds"][-1]["transfers"]:
                    transfer["path"] = [transfer["src"], "alternate", transfer["dst"]]
                control_report = replay(task, topology, control)
                control["metrics"] = {"total_time": control_report["total_time_us"] / 1e6,
                                      "round_times": [value / 1e6 for value in control_report["round_times_us"]]}
                for record in control["strategy"]["candidates"]:
                    record["selected"] = False
                control["strategy"]["candidates"].append({
                    "source": "synthetic_control", "actual": {}, "valid": True,
                    "selected": True, "total_time_us": control_report["total_time_us"]})
                chosen = {}

                def editor(_task, problem, **_kwargs):
                    group = min(problem["groups"], key=lambda row: row["round"])
                    chosen.update(round=group["round"], edits=[[group["id"], max(1, group["max_chunks"] // 4)]])
                    return {"base": problem["base"], "edits": chosen["edits"]}, {"called": True, "usage": {}}

                with patch("src.service.load_settings", return_value=SETTINGS), \
                        patch("src.service.solve_schedule", return_value=deepcopy(baseline)), \
                        patch("src.service._refined_candidate", side_effect=lambda *_args: deepcopy(control)), \
                        patch("src.service.edit_schedule", side_effect=editor):
                    result = service.solve_input({"task": task, "topology": topology}, refinement_budget_s=0)
                checked = replay(task, topology, result["answer"])
                search = result["diagnostics"]["search"]
                edit = search["model_edit"]
                self.assertTrue(checked["valid"], checked)
                self.assertTrue(edit["selected"] and edit["preserved_in_final"])
                self.assertGreater(edit["after_us"], edit["control_time_us"], "whole-candidate comparison would lose this model edit")
                self.assertLess(checked["total_time_us"], edit["after_us"])
                self.assertLess(checked["total_time_us"], edit["control_time_us"])
                self.assertEqual(edit["selected_rounds"], [chosen["round"]])
                self.assertEqual(edit["selected_edits"], chosen["edits"])
                self.assertEqual(result["answer"]["rounds"][-1], control["rounds"][-1])
                self.assertEqual(actions(result["answer"]), actions(baseline))
                selected = next(record for record in search["candidates"] if record["selected"])
                self.assertEqual(selected["actual"]["edits"], edit["selected_edits"])
                self.assertNotEqual(selected["control_candidate"], selected["parent_candidate"])


if __name__ == "__main__":
    unittest.main()
