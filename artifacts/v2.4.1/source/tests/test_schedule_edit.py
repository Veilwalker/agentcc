"""Concrete edits must preserve data dependencies and improve exact replay."""

from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError
import unittest

from src.collectives import build
from src.replay import replay
from src.schedule_edit import extract_problem, apply_proposal
from src.topology import Topology
from tests.test_route_balance import parallel_case


def data_actions(solution):
    return Counter((ri, transfer["src"], transfer["dst"], transfer["action"], token)
                   for ri, round_ in enumerate(solution["rounds"])
                   for transfer in round_["transfers"] for token in transfer["chunk_ranges"])


class ScheduleEditTests(unittest.TestCase):
    def fixture(self, operator="all-gather"):
        task, graph = parallel_case(operator)
        kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
        baseline = build(task, graph, 4, kind=kind, resource_weight=0, packets=1)
        problem, context = extract_problem(task, graph, baseline, time_budget_s=1)
        self.assertTrue(problem["groups"])
        return task, graph, baseline, problem, context

    def test_prefix_partition_preserves_actions_and_improves_both_operators(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline, problem, context = self.fixture(operator)
                saved = deepcopy(baseline)
                proposal = {"base": problem["base"], "edits": [[g["id"], max(1, g["max_chunks"] // 2 - 1)]
                                                              for g in problem["groups"]]}
                result, evidence = apply_proposal(task, graph, baseline, context, proposal)
                self.assertTrue(evidence["accepted"], evidence)
                self.assertEqual(data_actions(result), data_actions(baseline))
                self.assertEqual(baseline, saved)
                checked = replay(task, graph, result)
                self.assertTrue(checked["valid"], checked)
                self.assertLess(checked["total_time_us"], context.before_us)
                self.assertAlmostEqual(result["metrics"]["total_time"] * 1e6, checked["total_time_us"])
                self.assertEqual(evidence["applied_edits"], proposal["edits"])
                self.assertGreater(evidence["changed_transfers"], 0)
                self.assertEqual(evidence["returned_us"], checked["total_time_us"])

    def test_shared_resource_effects_are_exact_with_repeated_path_memberships(self):
        task, graph, baseline, problem, context = self.fixture()
        by_id = {row["id"]: row for round_ in problem["rounds"] for row in round_["constraints"]}
        for group in problem["groups"]:
            effects = {by_id[key]["name"]: delta for key, delta in group["effects"]
                       if by_id[key]["kind"] == "resource"}
            # Every old fast route traverses two edges belonging to both pools.
            self.assertEqual(effects["shared-a"], -2 * problem["chunk_bytes"])
            self.assertEqual(effects["shared-b"], -2 * problem["chunk_bytes"])
        identities = [(group.round_index, member.transfer_index)
                      for group in context.groups for member in group.members]
        self.assertEqual(len(identities), len(set(identities)))

    def test_bad_schema_cannot_change_the_baseline(self):
        task, graph, baseline, problem, context = self.fixture()
        name, maximum = problem["groups"][0]["id"], problem["groups"][0]["max_chunks"]
        bad = [None, {}, {"base": problem["base"], "edits": [], "path": ["evil"]},
               {"base": "0" * 16, "edits": [[name, 1]]}]
        bad += [{"base": problem["base"], "edits": value} for value in
                ({}, [[name, True]], [[name, -1]], [[name, maximum + 1]], [[name, 1.0]],
                 [[name, 1], [name, 1]], [["unknown", 1]], [{"group_id": name, "move_chunks": 1}])]
        for proposal in bad:
            with self.subTest(proposal=proposal):
                result, evidence = apply_proposal(task, graph, baseline, context, proposal)
                self.assertIs(result, baseline)
                self.assertFalse(evidence["accepted"])
                self.assertTrue(evidence["errors"])
                self.assertFalse(evidence["applied_edits"])

    def test_stale_task_graph_or_baseline_is_rejected(self):
        for changed in ("task", "graph", "baseline"):
            with self.subTest(changed=changed):
                task, graph, baseline, problem, context = self.fixture()
                proposal = {"base": problem["base"], "edits": [[g["id"], 1] for g in problem["groups"]]}
                if changed == "task":
                    task["max_nonempty_rounds"] += 1
                elif changed == "graph":
                    graph.resources["shared-a"] *= 2
                else:
                    baseline["rounds"][0]["transfers"][0]["chunk_ranges"].reverse()
                result, evidence = apply_proposal(task, graph, baseline, context, proposal)
                self.assertIs(result, baseline)
                self.assertFalse(evidence["accepted"])
                self.assertIn("stale", evidence["errors"][0])

    def test_overloading_alternate_resource_is_rejected_after_replay(self):
        task, graph = parallel_case()
        graph.resources["alternate-pool"] = 1e9
        for edge in graph.edges:
            if "alternate" in edge:
                graph.edge_resources[edge] = ("alternate-pool",)
        baseline = build(task, graph, 4, resource_weight=0)
        problem, context = extract_problem(task, graph, baseline, time_budget_s=1)
        proposal = {"base": problem["base"], "edits": [[g["id"], g["max_chunks"]] for g in problem["groups"]]}
        result, evidence = apply_proposal(task, graph, baseline, context, proposal)
        self.assertIs(result, baseline)
        self.assertFalse(evidence["accepted"])
        self.assertTrue(evidence["replay_valid"])
        self.assertGreater(evidence["proposal_time_us"], evidence["before_us"])
        self.assertEqual(evidence["after_us"], evidence["before_us"])
        self.assertEqual(evidence["returned_us"], evidence["before_us"])
        self.assertTrue(evidence["attempted_edits"])
        self.assertFalse(evidence["applied_edits"])

    def test_mixed_good_and_bad_rounds_keep_exact_model_actions_only_where_better(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph = parallel_case(operator)
                graph.resources["alternate-pool"] = 1e9
                for edge in graph.edges:
                    if "alternate" in edge:
                        graph.edge_resources[edge] = ("alternate-pool",)
                kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
                baseline = build(task, graph, 4, kind=kind, resource_weight=0, packets=1)
                before = replay(task, graph, baseline)
                # Costs in the context must come from Replay, not submitted
                # metrics; this deliberately misleading metadata is immutable.
                baseline["metrics"] = {"total_time": -1, "round_times": [-999] * len(baseline["rounds"])}
                saved = deepcopy(baseline)
                problem, context = extract_problem(task, graph, baseline, time_budget_s=1)
                self.assertEqual(context.before_round_times_us, tuple(before["round_times_us"]))
                good = context.groups[0]
                bad = next(group for group in context.groups if group.round_index != good.round_index)
                good_count = max(1, good.max_chunks // 5)
                proposal = {"base": problem["base"], "edits": [[good.id, good_count], [bad.id, bad.max_chunks]]}
                result, evidence = apply_proposal(task, graph, baseline, context, proposal)
                self.assertTrue(evidence["accepted"], evidence)
                self.assertGreater(evidence["proposal_time_us"], before["total_time_us"])
                self.assertLess(evidence["after_us"], before["total_time_us"])
                self.assertEqual(evidence["attempted_edits"], proposal["edits"])
                self.assertEqual(evidence["applied_edits"], [[good.id, good_count]])
                self.assertEqual(evidence["accepted_rounds"], [good.round_index + 1])
                self.assertEqual([row["round"] for row in evidence["rejected_rounds"]], [bad.round_index + 1])
                self.assertEqual(result["rounds"][bad.round_index], baseline["rounds"][bad.round_index])
                self.assertEqual(data_actions(result), data_actions(baseline))
                self.assertEqual(baseline, saved)
                self.assertEqual({row[0] for row in evidence["edit_manifest"]}, {good.round_index + 1})
                self.assertEqual({pair[0] for pair in evidence["frozen_transfers"]}, {good.round_index})
                moved = [result["rounds"][ri]["transfers"][ti] for ri, ti in evidence["frozen_transfers"]]
                self.assertEqual(sum(len(transfer["chunk_ranges"]) for transfer in moved), good_count)
                for transfer in moved:
                    self.assertTrue(any(tuple(transfer["path"]) == member.path for member in good.members))
                final = replay(task, graph, result)
                self.assertTrue(final["valid"], final)
                self.assertAlmostEqual(evidence["after_us"], final["total_time_us"])
                self.assertEqual(evidence["after_us"], evidence["returned_us"])
                self.assertAlmostEqual(result["metrics"]["total_time"] * 1e6, final["total_time_us"])

    def test_prompt_mutation_cannot_rewrite_private_context_or_baseline(self):
        task, graph, baseline, problem, context = self.fixture()
        saved = deepcopy(baseline)
        problem["groups"][0]["path_sample"]["old"][0] = "invalid"
        problem["groups"][0]["path_sample"]["new"][0] = "invalid"
        problem["groups"][0]["effects"].clear()
        with self.assertRaises(FrozenInstanceError):
            context.base = "changed"
        self.assertEqual(baseline, saved)
        proposal = {"base": context.base, "edits": [[context.groups[0].id, 1]]}
        result, evidence = apply_proposal(task, graph, baseline, context, proposal)
        self.assertTrue(replay(task, graph, result)["valid"])
        self.assertEqual(baseline, saved)

    def test_zero_budget_and_invalid_baseline(self):
        task, graph, baseline, _, _ = self.fixture()
        problem, context = extract_problem(task, graph, baseline, time_budget_s=0)
        self.assertEqual(problem["groups"], [])
        result, evidence = apply_proposal(task, graph, baseline, context, {"base": problem["base"], "edits": []})
        self.assertIs(result, baseline)
        self.assertFalse(evidence["accepted"])
        baseline["rounds"][0]["transfers"][0]["path"] = ["invalid"]
        with self.assertRaisesRegex(ValueError, "baseline is invalid"):
            extract_problem(task, graph, baseline)


if __name__ == "__main__":
    unittest.main()
