"""Independent rollback and original-index checks for per-round acceptance."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.schedule_edit import apply_proposal, extract_problem
from tests.test_edit_invariants import actions
from tests.test_route_balance import parallel_case


class RoundFilterInvariantTests(unittest.TestCase):
    def fixture(self, operator):
        task, graph = parallel_case(operator)
        graph.resources["alternate-pool"] = 1e9
        for edge in graph.edges:
            if "alternate" in edge:
                graph.edge_resources[edge] = ("alternate-pool",)
        kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
        baseline = build(task, graph, 4, kind=kind, resource_weight=0)
        baseline["rounds"].insert(0, {"round": 0, "transfers": []})
        baseline["rounds"].insert(2, {"round": 0, "transfers": []})
        for index, round_ in enumerate(baseline["rounds"], 1):
            round_["round"] = index
        baseline["round_count"] = len(baseline["rounds"])
        before = replay(task, graph, baseline)
        self.assertTrue(before["valid"], before)
        problem, context = extract_problem(task, graph, baseline, time_budget_s=1, max_rounds=8)
        good = context.groups[0]
        bad = next(group for group in context.groups if group.round_index != good.round_index)
        amount = max(1, good.max_chunks // 5)
        proposal = {"base": problem["base"], "edits": [[good.id, amount], [bad.id, bad.max_chunks]]}
        return task, graph, baseline, context, proposal, good, bad

    def test_empty_rounds_keep_original_indices_and_only_good_actions_for_both_operators(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline, context, proposal, good, bad = self.fixture(operator)
                original = deepcopy(baseline)
                result, evidence = apply_proposal(task, graph, baseline, context, proposal)
                self.assertTrue(evidence["accepted"], evidence)
                self.assertEqual(context.before_round_times_us[0], 0)
                self.assertEqual(context.before_round_times_us[2], 0)
                self.assertEqual(evidence["accepted_rounds"], [good.round_index + 1])
                self.assertEqual(evidence["applied_edits"], [proposal["edits"][0]])
                self.assertGreater(evidence["proposal_time_us"], context.before_us)
                self.assertLess(evidence["returned_us"], context.before_us)
                self.assertEqual(actions(result), actions(baseline))
                self.assertEqual(baseline, original)
                for ri, round_ in enumerate(result["rounds"]):
                    if ri != good.round_index:
                        self.assertEqual(round_, baseline["rounds"][ri])
                self.assertEqual({row[0] for row in evidence["frozen_transfers"]}, {good.round_index})
                moved = [result["rounds"][ri]["transfers"][ti] for ri, ti in evidence["frozen_transfers"]]
                self.assertEqual(sum(len(row["chunk_ranges"]) for row in moved), proposal["edits"][0][1])
                self.assertTrue(replay(task, graph, result)["valid"])

    def test_schema_error_is_not_repaired_by_accepting_other_valid_rounds(self):
        task, graph, baseline, context, proposal, _, _ = self.fixture("all-gather")
        proposal["edits"].append(["unknown-group", 1])
        result, evidence = apply_proposal(task, graph, baseline, context, proposal)
        self.assertIs(result, baseline)
        self.assertFalse(evidence["accepted"])
        self.assertEqual(evidence["applied_edits"], [])
        self.assertEqual(evidence["frozen_transfers"], [])

    def test_filtered_candidate_replay_failure_rolls_back_all_partial_acceptance(self):
        task, graph, baseline, context, proposal, _, _ = self.fixture("all-reduce")
        calls = []

        def validate(task, graph, candidate):
            calls.append(candidate)
            if len(calls) == 2:
                return {"valid": False, "errors": ["synthetic filtered validation failure"],
                        "total_time_us": 0, "round_times_us": []}
            return replay(task, graph, candidate)

        with patch("src.schedule_edit.replay", side_effect=validate):
            result, evidence = apply_proposal(task, graph, baseline, context, proposal)
        self.assertEqual(len(calls), 2)
        self.assertIs(result, baseline)
        self.assertFalse(evidence["accepted"])
        self.assertFalse(evidence["replay_valid"])
        self.assertEqual(evidence["after_us"], context.before_us)
        self.assertEqual(evidence["returned_us"], context.before_us)
        for key in ("applied_edits", "accepted_rounds", "frozen_transfers", "edit_manifest"):
            self.assertEqual(evidence[key], [], key)


if __name__ == "__main__":
    unittest.main()
