"""Service integration with real compilation, offline wire replies and Replay."""

from contextlib import ExitStack
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from src import service
from src.collectives import build
from src.config import Settings
from src.provenance import build_model_ledger
from src.replay import replay
from src.schedule_edit import apply_proposal
from src.selection import choose_model_candidate
from src.topology import Topology
from tests.test_route_refinement_audit import congested_fixture


SETTINGS = Settings("http://unused.invalid/v1/chat/completions", "synthetic-key", "synthetic-model")


def fixture(operator):
    task, topology, _ = congested_fixture(operator)
    # A third legal route makes it possible to test complete overwriting of an
    # initially useful model route without violating any communication action.
    topology["switches"].append("third")
    topology["edges"].extend({"src": rank, "dst": "third", "bw": "9GB/s", "bidirectional": True}
                             for rank in task["ranks"])
    graph = Topology.from_data(topology)
    algorithm = "xor_tree" if operator == "all-reduce" else "reverse_doubling"
    seed = build(task, graph, 4, kind=algorithm, packets=1)
    for row in seed["rounds"]:
        for transfer in row["transfers"]:
            transfer["path"] = [transfer["src"], "fast", transfer["dst"]]
    report = replay(task, graph, seed)
    assert report["valid"]
    seed["metrics"] = {"total_time": report["total_time_us"] / 1e6,
                       "round_times": [value / 1e6 for value in report["round_times_us"]]}
    params = {"algorithm": algorithm, "split": 4, "rank_order": "input", "resource_weight": 0, "packets": 1}
    seed["strategy"] = {**params, "route_variation": False, "routing_bytes": 8192,
                        "validation_report": report,
                        "candidates": [{"source": "fixture_seed", "actual": params,
                                        "valid": True, "selected": True,
                                        "total_time_us": report["total_time_us"]}]}
    return task, topology, seed


class CapacityServiceTests(unittest.TestCase):
    def run_service(self, operator="all-gather", mode="real", reply="positive"):
        task, topology, seed = fixture(operator)
        request = {"task": task, "topology": topology}
        original = deepcopy((request, seed))
        captured = {}

        def raw_reply(system, content, settings, **kwargs):
            captured["raw_calls"] = captured.get("raw_calls", 0) + 1
            problem = json.loads(content)["problem"]
            counts = [0 if reply == "noop" else 1 for _ in problem["groups"]]
            wire = {"counts": counts}
            raw = "invalid JSON" if reply == "invalid" else json.dumps(wire, separators=(",", ":"))
            captured.update(raw=raw, wire=wire)
            return raw, {"finish_reason": "stop", "usage": {}, "synthetic": True,
                         "real_model_calls": 0, "network_calls": 0}

        def raw_control(_task, graph, base, **kwargs):
            control = deepcopy(base)
            return control, replay(_task, graph, control), 0.0

        def compile_model(*args, **kwargs):
            candidate, evidence = apply_proposal(*args, **kwargs)
            captured["compiled"] = candidate
            captured["compile_evidence"] = deepcopy(evidence)
            captured["ledger"] = build_model_ledger(args[2], evidence, context=args[3])
            return candidate, evidence

        real_compare = service.compare_with_control
        def compare(*args, **kwargs):
            task, graph, baseline, raw, raw_report, strong, strong_report, edited, edited_report, rounds = args
            old, old_report, _ = choose_model_candidate(task, graph, raw, raw_report, edited, edited_report,
                                                        rounds, baseline=baseline, overlay_min_relative_gain=.001)
            captured.update(old=old, old_report=old_report, strong=strong, strong_report=strong_report)
            return real_compare(*args, **kwargs)

        def controlled_joint(_task, _graph, candidate, report, **kwargs):
            if "ledger" not in captured or not captured["ledger"].available:
                return candidate, {"accepted": False, "reason": "controlled_pre_model_stage"}
            targets = {atom.key: atom.target_path for atom in captured["ledger"].atoms}
            rounds = []
            for ri, row in enumerate(candidate["rounds"]):
                transfers = []
                for transfer in reversed(row["transfers"]):
                    for token in reversed(transfer["chunk_ranges"]):
                        key = ri, transfer["action"], transfer["src"], transfer["dst"], token
                        if mode == "preserve" and key in targets:
                            path = targets[key]
                        else:
                            switch = ("third" if key in targets and targets[key][1] == "alternate"
                                      else "alternate")
                            path = (transfer["src"], switch, transfer["dst"])
                        transfers.append({**transfer, "chunk_ranges": [token], "path": list(path)})
                rounds.append({**row, "transfers": transfers})
            return {**candidate, "rounds": rounds}, {"accepted": True, "scope": "semantic-test-proposal"}

        with ExitStack() as stack:
            stack.enter_context(patch("src.service.load_settings", return_value=SETTINGS))
            stack.enter_context(patch("src.service.solve_schedule", return_value=seed))
            stack.enter_context(patch("src.service._control_candidate", side_effect=raw_control))
            stack.enter_context(patch("src.llm._chat_completion", side_effect=raw_reply))
            stack.enter_context(patch("src.llm.build_opener", side_effect=AssertionError("network forbidden")))
            stack.enter_context(patch("src.service.apply_proposal", side_effect=compile_model))
            stack.enter_context(patch("src.service.compare_with_control", side_effect=compare))
            if mode != "real":
                stack.enter_context(patch("src.postprocess.propose_joint", side_effect=controlled_joint))
            output = service.solve_input(request)
        checked = replay(task, topology, output["answer"])
        self.assertTrue(checked["valid"], checked["errors"])
        self.assertEqual(output["diagnostics"]["local_replay"], checked)
        self.assertLessEqual(checked["total_time_us"], captured["old_report"]["total_time_us"])
        self.assertLessEqual(checked["total_time_us"], captured["strong_report"]["total_time_us"])
        self.assertEqual((request, seed), original)
        winners = [row for row in output["diagnostics"]["search"]["candidates"] if row["selected"]]
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["total_time_us"], checked["total_time_us"])
        self.assertEqual(output["answer"]["metrics"]["total_time"], checked["total_time_us"] / 1e6)
        metadata = output["diagnostics"]["llm"]
        self.assertEqual(metadata["response_evidence"]["content"], captured["raw"])
        self.assertEqual(captured["raw_calls"], 1)
        self.assertTrue(metadata["called"])
        self.assertTrue(metadata["synthetic"])
        if reply != "invalid":
            self.assertEqual(metadata["response_wire_proposal"], captured["wire"])
            self.assertEqual([row[1] for row in metadata["response_proposal"]["edits"]], captured["wire"]["counts"])
        return output, captured

    def test_real_capacity_chain_keeps_both_validated_performance_bounds(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                output, captured = self.run_service(operator)
                evidence = output["diagnostics"]["search"]["model_edit"]
                self.assertEqual(evidence["requested_edits"], captured["compile_evidence"]["requested_edits"])
                self.assertEqual(evidence["applied_edits"], captured["compile_evidence"]["applied_edits"])

    def test_preserved_final_atoms_use_rebuilt_indices_after_actual_service_polish(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                output, captured = self.run_service(operator, mode="preserve")
                evidence = output["diagnostics"]["search"]["model_edit"]
                self.assertTrue(evidence["model_branch_won"])
                self.assertTrue(evidence["selected"])
                self.assertTrue(evidence["preserved_in_final"])
                self.assertEqual(evidence["selected_edits"], evidence["applied_edits"])
                coordinates = []
                for row in evidence["selected_edit_manifest"]:
                    ri, ti = row["round"] - 1, row["transfer_index"]
                    transfer = output["answer"]["rounds"][ri]["transfers"][ti]
                    self.assertEqual(transfer["path"], row["model_path"])
                    self.assertEqual([transfer["chunk_ranges"][ci] for ci in row["chunk_indices"]], row["chunks"])
                    coordinates.extend([ri, ti, ci] for ci in row["chunk_indices"])
                self.assertEqual(sorted(coordinates), evidence["selected_atomic_positions"])
                self.assertNotEqual(evidence["selected_transfers"], captured["compile_evidence"]["frozen_transfers"])
                self.assertFalse(evidence["selected_frozen_transfers"])

    def test_overwriting_all_model_paths_does_not_reject_the_faster_winner(self):
        output, captured = self.run_service(mode="overwrite")
        evidence = output["diagnostics"]["search"]["model_edit"]
        self.assertTrue(evidence["model_branch_won"])
        self.assertFalse(evidence["selected"])
        self.assertFalse(evidence["preserved_in_final"])
        self.assertEqual(evidence["model_provenance"]["preserved_atomic_count"], 0)
        self.assertGreater(evidence["model_provenance"]["overwritten_atomic_count"], 0)
        self.assertEqual(evidence["selected_edits"], [])
        self.assertEqual(evidence["applied_edits"], captured["compile_evidence"]["applied_edits"])
        selected = next(row for row in output["diagnostics"]["search"]["candidates"] if row["selected"])
        self.assertEqual(selected["source"], "model_branch_postprocess")
        self.assertLess(output["diagnostics"]["local_replay"]["total_time_us"], captured["old_report"]["total_time_us"])

    def test_bad_and_noop_wire_replies_keep_strong_control_without_fake_actions(self):
        for reply in ("invalid", "noop"):
            with self.subTest(reply=reply):
                output, _ = self.run_service(reply=reply)
                evidence = output["diagnostics"]["search"]["model_edit"]
                self.assertFalse(evidence["accepted"])
                self.assertFalse(evidence["selected"])
                self.assertFalse(evidence["preserved_in_final"])


if __name__ == "__main__":
    unittest.main()
