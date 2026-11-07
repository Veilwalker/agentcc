"""Joint proposals preserve data semantics and leave final Replay to callers."""

from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.route_joint import propose_joint
from src.selection import round_actions
from src.topology import Topology
from tests.test_route_balance import parallel_case
from tests.test_solver import make_case


def path_limited_case(operator="all-gather"):
    task, topology = make_case(8, operator)
    topology["switches"] = ["fast", "alternate"]
    topology["edges"] = [{"src": rank, "dst": switch, "bw": capacity, "bidirectional": True}
                         for rank in task["ranks"]
                         for switch, capacity in (("fast", "10GB/s"), ("alternate", "8GB/s"))]
    graph = Topology.from_data(topology)
    kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
    solution = build(task, graph, 4, kind=kind, resource_weight=0, packets=1)
    return task, graph, solution, replay(task, graph, solution)


class RouteJointTests(unittest.TestCase):
    def test_valid_proposal_preserves_complete_atoms_and_exact_round_costs(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline, report = path_limited_case(operator)
                original = deepcopy(baseline)
                with patch("src.replay.replay", side_effect=AssertionError("proposer must not Replay")):
                    result, info = propose_joint(task, graph, baseline, report)
                self.assertTrue(info["proposed"])
                self.assertIsNot(result, baseline)
                checked = replay(task, graph, result)
                self.assertTrue(checked["valid"], checked["errors"])
                self.assertAlmostEqual(checked["total_time_us"], info["candidate_time_us"])
                self.assertLess(checked["total_time_us"], report["total_time_us"])
                self.assertEqual(baseline, original)
                for old, new, old_time, new_time in zip(baseline["rounds"], result["rounds"],
                                                        report["round_times_us"], checked["round_times_us"]):
                    self.assertEqual(round_actions(old), round_actions(new))
                    self.assertLessEqual(new_time, old_time + 1e-8)
                self.assertTrue(info["requires_replay"])
                self.assertNotIn("replay_valid", info)

    def test_same_path_batches_merge_and_no_route_opportunity_is_identity(self):
        task, topology = make_case(8)
        graph = Topology.from_data(topology)
        baseline = build(task, graph, 4, packets=1)
        report = replay(task, graph, baseline)
        result, info = propose_joint(task, graph, baseline, report)
        self.assertIs(result, baseline)
        self.assertFalse(info["proposed"])
        self.assertEqual(info["extra_packets"], 0)

    def test_work_limits_stop_at_original_packet_boundaries(self):
        task, graph, baseline, report = path_limited_case()
        for updates, scores in ((1, 40000), (6, 40000), (6000, 20)):
            with self.subTest(updates=updates, scores=scores):
                result, info = propose_joint(task, graph, baseline, report,
                    max_batch_updates=updates, max_path_scores=scores, min_relative_gain=0)
                work = info["work"]
                self.assertLessEqual(work["batch_updates"], updates)
                self.assertEqual(work["batch_updates"] % 2, 0)
                self.assertLessEqual(work["path_evaluations"], scores)
                self.assertEqual(work["path_evaluations"], work["path_scores"] + work["path_bound_checks"])
                self.assertTrue(replay(task, graph, result)["valid"])

    def test_packet_growth_cap_and_gain_threshold_preserve_safe_fallback(self):
        task, graph, baseline, report = path_limited_case()
        for cap in (0, 1, 3):
            result, info = propose_joint(task, graph, baseline, report, max_extra_packets=cap)
            extra = sum(len(r["transfers"]) for r in result["rounds"]) - sum(len(r["transfers"]) for r in baseline["rounds"])
            self.assertLessEqual(extra, cap)
            self.assertEqual(extra, info["extra_packets"])
            self.assertTrue(replay(task, graph, result)["valid"])
        result, info = propose_joint(task, graph, baseline, report, min_relative_gain=0.99)
        self.assertIs(result, baseline)
        self.assertFalse(info["proposed"])

    def test_tiny_legal_alternative_resource_does_not_overflow_or_emit_nan(self):
        task, graph, baseline, report = path_limited_case()
        graph.resources["tiny"] = 1e-300
        for edge in graph.edges:
            if "alternate" in edge:
                graph.edge_resources[edge] = ("tiny",)
        report = replay(task, graph, baseline)
        result, info = propose_joint(task, graph, baseline, report)
        self.assertIs(result, baseline)
        self.assertTrue(math.isfinite(info["candidate_time_us"]))
        json.dumps(info, allow_nan=False)
        self.assertTrue(replay(task, graph, result)["valid"])

    def test_overlapping_resource_multiplicity_matches_independent_replay(self):
        task, graph = parallel_case("all-reduce")
        baseline = build(task, graph, 4, kind="xor_tree", resource_weight=0, packets=1)
        report = replay(task, graph, baseline)
        result, info = propose_joint(task, graph, baseline, report, path_dominance=0)
        checked = replay(task, graph, result)
        self.assertTrue(checked["valid"], checked["errors"])
        self.assertAlmostEqual(checked["total_time_us"], info["candidate_time_us"])

    def test_invalid_report_or_budget_is_not_accepted_as_trusted_input(self):
        task, graph, baseline, report = path_limited_case()
        with self.assertRaises(ValueError):
            propose_joint(task, graph, baseline, {**report, "valid": False})
        for changes in ({"max_batch_updates": True}, {"max_extra_packets": -1},
                        {"max_path_scores": 1.5}, {"wall_safety_s": math.inf},
                        {"min_relative_gain": math.nan}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                propose_joint(task, graph, baseline, report, **changes)

    def test_fixed_work_repeats_have_identical_proposals(self):
        task, graph, baseline, report = path_limited_case()
        results = [propose_joint(task, graph, baseline, report) for _ in range(3)]
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(results[1][0], results[2][0])
        self.assertEqual(results[0][1]["work"], results[2][1]["work"])


if __name__ == "__main__":
    unittest.main()
