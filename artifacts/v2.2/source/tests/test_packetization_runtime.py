"""Production packet grouping is pure, bounded and independently replayable."""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from src.packetization import MAX_EXTRA_PACKETS, repacketize
from src.replay import replay
from tests.test_packetization import atomic_rows, fixture, loads


class RuntimePacketizationTests(unittest.TestCase):
    def check_case(self, operator="all-gather", *, capacity=None, budget=2048, threshold=0.002):
        task, topology, graph, answer = fixture(operator, shared_capacity=capacity)
        baseline = replay(task, graph, answer)
        self.assertTrue(baseline["valid"], baseline["errors"])
        original = deepcopy((task, topology, answer, baseline))
        with patch("src.replay.replay", side_effect=AssertionError("packetizer called Replay")):
            candidate, evidence = repacketize(task, graph, answer, baseline,
                                              max_extra_packets=budget, min_relative_gain=threshold)
        final = replay(task, graph, candidate)
        self.assertTrue(final["valid"], final["errors"])
        self.assertEqual((task, topology, answer, baseline), original)
        self.assertEqual(atomic_rows(candidate), atomic_rows(answer))
        self.assertEqual(loads(candidate, task, graph), loads(answer, task, graph))
        self.assertAlmostEqual(evidence["predicted_after_us"], final["total_time_us"])
        self.assertFalse(evidence["replay_performed"])
        actual_extra = sum(len(row["transfers"]) for row in candidate["rounds"]) - sum(
            len(row["transfers"]) for row in answer["rounds"])
        self.assertEqual(actual_extra, evidence["actual_extra_packets"])
        self.assertEqual(actual_extra, evidence["after_transfer_count"] - evidence["before_transfer_count"])
        self.assertLessEqual(actual_extra, budget)
        return task, graph, answer, candidate, baseline, final, evidence

    def test_ag_and_ar_improve_with_all_atoms_and_expression_boundaries_preserved(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                _, _, answer, candidate, before, after, evidence = self.check_case(operator)
                self.assertTrue(evidence["applied"] and evidence["requires_final_replay"])
                self.assertLess(after["total_time_us"], before["total_time_us"])
                self.assertEqual(candidate["round_count"], answer["round_count"])
                self.assertEqual(candidate["split"], answer["split"])
                if operator == "all-reduce":
                    actual = [token for t in candidate["rounds"][1]["transfers"] for token in t["chunk_ranges"]]
                    self.assertEqual(actual, answer["rounds"][1]["transfers"][0]["chunk_ranges"])
                    self.assertTrue(all("+" in token for token in actual))

    def test_duplicate_resource_references_and_overlapping_pools_count_each_path_edge(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, answer, candidate, before, after, evidence = self.check_case(operator, capacity="100GB/s")
                self.assertTrue(evidence["applied"])
                self.assertLess(after["total_time_us"], before["total_time_us"])
                for row, (_, resource_load) in zip(answer["rounds"], loads(candidate, task, graph)):
                    payload = sum(len(t["chunk_ranges"]) for t in row["transfers"]) * 8192
                    self.assertEqual(resource_load, {"overlap-a": 2 * payload, "overlap-b": 2 * payload})

    def test_threshold_measures_total_gain_and_returns_exact_original_below_it(self):
        for threshold, expected in ((0, True), (0.002, True), (0.8, False), (1, False)):
            with self.subTest(threshold=threshold):
                _, _, answer, candidate, _, _, evidence = self.check_case(threshold=threshold)
                self.assertEqual(evidence["applied"], expected)
                if not expected:
                    self.assertIs(candidate, answer)
                    self.assertEqual(evidence["actual_extra_packets"], 0)
                    self.assertEqual(evidence["predicted_after_us"], evidence["before_us"])
                    self.assertLess(evidence["proposed_relative_gain"], threshold)
                    self.assertEqual(evidence["reason"], "below_min_relative_gain")

    def test_budget_and_output_limit_count_actual_new_transfers(self):
        for budget in (0, 1, 2, 3):
            with self.subTest(budget=budget):
                _, _, answer, candidate, _, _, evidence = self.check_case(budget=budget)
                self.assertEqual(evidence["applied"], budget >= 2)
                if budget < 2:
                    self.assertIs(candidate, answer)
        task, _, graph, answer = fixture()
        report = replay(task, graph, answer)
        with patch("src.packetization.MAX_OUTPUT_TRANSFERS", 3):
            candidate, evidence = repacketize(task, graph, answer, report)
        self.assertIs(candidate, answer)
        self.assertEqual(evidence["effective_extra_budget"], 1)
        with patch("src.packetization.MAX_OUTPUT_TRANSFERS", 2), \
                patch("src.packetization._PathCosts", side_effect=AssertionError("oversize optional work")):
            candidate, evidence = repacketize(task, graph, answer, report)
        self.assertIs(candidate, answer)
        self.assertEqual(evidence["reason"], "output_size_limit")

    def test_zero_budget_and_single_atoms_skip_path_costs(self):
        task, _, graph, answer = fixture()
        report = replay(task, graph, answer)
        with patch("src.packetization._PathCosts", side_effect=AssertionError("zero budget optional work")):
            candidate, evidence = repacketize(task, graph, answer, report, max_extra_packets=0)
        self.assertIs(candidate, answer)
        self.assertEqual(evidence["reason"], "zero_budget")
        for row in answer["rounds"]:
            row["transfers"] = [{**t, "chunk_ranges": [atom]}
                                for t in row["transfers"] for atom in t["chunk_ranges"]]
        report = replay(task, graph, answer)
        with patch("src.packetization._PathCosts", side_effect=AssertionError("indivisible optional work")):
            candidate, evidence = repacketize(task, graph, answer, report)
        self.assertIs(candidate, answer)
        self.assertEqual(evidence["reason"], "no_splittable_transfer")

    def test_resource_dominated_round_skips_load_rebuild_and_returns_identity(self):
        task, _, graph, answer = fixture(shared_capacity="0.1GB/s")
        report = replay(task, graph, answer)
        with patch("src.packetization._RoundLoad", side_effect=AssertionError("resource floor already dominates")):
            candidate, evidence = repacketize(task, graph, answer, report)
        self.assertIs(candidate, answer)
        self.assertEqual(evidence["rounds_skipped_by_replay"], 1)
        self.assertFalse(evidence["applied"])

    def test_predicted_metrics_and_empty_round_alignment_match_full_replay(self):
        task, _, graph, answer = fixture("all-reduce")
        for row in answer["rounds"]:
            row["round"] += 1
        answer["rounds"].insert(0, {"round": 1, "transfers": []})
        answer["round_count"] += 1
        answer["metrics"] = {"total_time": 999, "round_times": [], "tag": "kept"}
        original = deepcopy(answer)
        baseline = replay(task, graph, answer)
        candidate, evidence = repacketize(task, graph, answer, baseline)
        checked = replay(task, graph, candidate)
        self.assertTrue(checked["valid"], checked["errors"])
        self.assertAlmostEqual(candidate["metrics"]["total_time"] * 1e6, checked["total_time_us"])
        self.assertEqual(candidate["metrics"]["round_times"], [v / 1e6 for v in checked["round_times_us"]])
        self.assertEqual(candidate["metrics"]["tag"], "kept")
        self.assertEqual(answer, original)
        self.assertEqual(evidence["actual_extra_packets"], 2)

    def test_invalid_budget_threshold_and_nonfinite_report_are_rejected(self):
        task, _, graph, answer = fixture()
        baseline = replay(task, graph, answer)
        original = deepcopy((task, answer, baseline))
        for budget in (-1, True, None, "2", 1.5, math.inf, MAX_EXTRA_PACKETS + 1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                repacketize(task, graph, answer, baseline, max_extra_packets=budget)
        for threshold in (-0.1, True, None, "0.1", math.nan, math.inf, 1.01, 10 ** 400):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                repacketize(task, graph, answer, baseline, min_relative_gain=threshold)
        for report in ([], {"valid": False},
                       {"valid": True, "round_times_us": [math.inf], "total_time_us": math.inf},
                       {"valid": True, "round_times_us": [10 ** 400], "total_time_us": 10 ** 400},
                       {"valid": True, "round_times_us": [True], "total_time_us": 1},
                       {"valid": True, "round_times_us": [], "total_time_us": 0}):
            with self.subTest(report=report), self.assertRaises(ValueError):
                repacketize(task, graph, answer, report)
        self.assertEqual((task, answer, baseline), original)

    def test_internal_candidate_without_report_is_costed_not_claimed_validated(self):
        task, _, graph, answer = fixture()
        original = deepcopy(answer)
        actual = replay(task, graph, answer)
        with_report, _ = repacketize(task, graph, answer, actual)
        candidate, evidence = repacketize(task, graph, answer, None)
        self.assertEqual(candidate, with_report)
        self.assertEqual(evidence["baseline_cost_source"], "path_resource_model")
        self.assertFalse(evidence["replay_performed"])
        self.assertTrue(replay(task, graph, candidate)["valid"])
        self.assertEqual(answer, original)

    def test_nonfinite_chunk_or_path_cost_is_rejected_without_mutation(self):
        task, _, graph, answer = fixture()
        baseline = replay(task, graph, answer)
        task["data_size"]["per_rank_bytes"] = 10 ** 400
        original = deepcopy((task, answer, baseline))
        with self.assertRaisesRegex(ValueError, "finite"):
            repacketize(task, graph, answer, baseline)
        self.assertEqual((task, answer, baseline), original)
        task["data_size"]["per_rank_bytes"] = 32768
        with patch("src.packetization._PathCosts.describe", return_value=((), {}, math.inf, 1)):
            with self.assertRaisesRegex(ValueError, "finite"):
                repacketize(task, graph, answer, baseline)


if __name__ == "__main__":
    unittest.main()
