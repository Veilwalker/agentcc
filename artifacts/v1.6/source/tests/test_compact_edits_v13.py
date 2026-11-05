"""Compact model tables retain physical constraints and trusted-report boundaries."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.collectives import build
from src.llm import compact_edit_problem
from src.replay import replay
from src.schedule_edit import apply_proposal, extract_problem
from tests.test_route_balance import parallel_case


class CompactEditTests(unittest.TestCase):
    def fixture(self, operator="all-gather"):
        task, graph = parallel_case(operator)
        kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
        baseline = build(task, graph, 4, kind=kind, resource_weight=0)
        return task, graph, baseline

    def test_capacities_deduplicate_across_rounds_without_merging_directions(self):
        constraints = [
            {"id": "a", "kind": "resource", "name": "shared", "capacity_Bps": 2e9, "load_bytes": 1000},
            {"id": "b", "kind": "edge", "name": ["s0", "s1"], "capacity_Bps": 3e9, "load_bytes": 3000},
        ]
        repeated = deepcopy(constraints[0]); repeated.update(id="c", load_bytes=4000)
        reverse = deepcopy(constraints[1]); reverse.update(id="d", name=["s1", "s0"])
        problem = {"base": "0123456789abcdef", "groups": [], "rounds": [
            {"round": 1, "constraints": constraints}, {"round": 2, "constraints": [repeated, reverse]}]}
        compact = compact_edit_problem(problem)
        self.assertEqual(compact["constraints"], [["shared", 2.0], [["s0", "s1"], 3.0], [["s1", "s0"], 3.0]])
        self.assertEqual(compact["rounds"][0]["loads"], [0.5, 1.0, None])
        self.assertEqual(compact["rounds"][1]["loads"], [2.0, None, 1.0])

    def test_all_affected_pools_and_repeated_memberships_survive_projection(self):
        task, graph, baseline = self.fixture()
        # A pool common to both paths has zero delta but remains a constraint.
        graph.resources["common"] = 12e9
        for edge in graph.edge_resources:
            graph.edge_resources[edge] += ("common",)
        problem, _ = extract_problem(task, graph, baseline, time_budget_s=1)
        compact = compact_edit_problem(problem)
        columns = {row[0]: index for index, row in enumerate(compact["constraints"]) if isinstance(row[0], str)}
        self.assertIn("common", columns)
        for group in compact["groups"]:
            self.assertEqual(group["delta"][columns["common"]], 0)
            # Two edges per route each consume both overlapping old pools.
            for name in ("shared-a", "shared-b"):
                expected = -2 * problem["chunk_bytes"] * group["max_chunks"] / graph.resources[name] * 1e6
                self.assertAlmostEqual(group["delta"][columns[name]], expected, delta=abs(expected) * 1e-5)
        self.assertLessEqual(len(problem["groups"]), 4)
        self.assertLessEqual(len(problem["rounds"]), 2)

    def test_trusted_replay_reuse_ignores_misleading_schedule_metrics(self):
        task, graph, baseline = self.fixture()
        report = replay(task, graph, baseline)
        baseline["metrics"] = {"total_time": -42, "round_times": []}
        with patch("src.schedule_edit.replay", side_effect=AssertionError("duplicate replay")):
            problem, context = extract_problem(task, graph, baseline, time_budget_s=0,
                                               validated_report=report)
        self.assertEqual(context.before_us, report["total_time_us"])
        self.assertEqual(problem["before_us"], report["total_time_us"])
        self.assertEqual(context.before_round_times_us, tuple(report["round_times_us"]))

    def test_default_still_independently_replays_and_rejects_invalid_seed(self):
        task, graph, baseline = self.fixture()
        baseline["rounds"][0]["transfers"][0]["path"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "baseline is invalid"):
            extract_problem(task, graph, baseline, time_budget_s=0)

    def test_inconsistent_internal_reports_cannot_enter_context(self):
        task, graph, baseline = self.fixture()
        report = replay(task, graph, baseline)
        for change in ({"valid": 1}, {"errors": ["failure"]}, {"split": True},
                       {"split": 999}, {"nonempty_rounds": True},
                       {"round_times_us": []}, {"total_time_us": float("nan")},
                       {"total_time_us": report["total_time_us"] + 1},
                       {"round_times_us": [False] * report["nonempty_rounds"]}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "trusted validation report"):
                extract_problem(task, graph, baseline, time_budget_s=0,
                                validated_report={**report, **change})

    def test_reused_report_does_not_bypass_live_input_or_baseline_binding(self):
        for mutate in ("graph", "baseline"):
            task, graph, baseline = self.fixture()
            problem, context = extract_problem(task, graph, baseline, time_budget_s=1,
                                               validated_report=replay(task, graph, baseline))
            if mutate == "graph":
                graph.resources["shared-a"] *= 2
            else:
                baseline["rounds"][0]["transfers"][0]["chunk_ranges"].reverse()
            returned, evidence = apply_proposal(task, graph, baseline, context,
                {"base": problem["base"], "edits": [[problem["groups"][0]["id"], 1]]})
            self.assertIs(returned, baseline)
            self.assertIn("stale", evidence["errors"][0])
            self.assertNotIn("_validation_report", evidence)

    def test_report_handoff_is_for_filtered_returned_schedule_only(self):
        task, graph, baseline = self.fixture("all-reduce")
        graph.resources["alternate-pool"] = 1e9
        for edge in graph.edges:
            if "alternate" in edge:
                graph.edge_resources[edge] = ("alternate-pool",)
        problem, context = extract_problem(task, graph, baseline, time_budget_s=1)
        good, bad = context.groups[:2]
        self.assertNotEqual(good.round_index, bad.round_index)
        proposal = {"base": problem["base"], "edits": [[good.id, max(1, good.max_chunks // 5)],
                                                        [bad.id, bad.max_chunks]]}
        result, evidence = apply_proposal(task, graph, baseline, context, proposal)
        self.assertTrue(evidence["accepted"], evidence)
        self.assertTrue(evidence["rejected_rounds"])
        self.assertEqual(evidence["_validation_report"], replay(task, graph, result))
        self.assertLess(evidence["_validation_report"]["total_time_us"], evidence["proposal_time_us"])
        _, rejected = apply_proposal(task, graph, baseline, context, {"base": problem["base"], "edits": []})
        self.assertNotIn("_validation_report", rejected)


if __name__ == "__main__":
    unittest.main()
