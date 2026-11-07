"""Model atom overlays preserve counts while retaining independent control gains."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.selection import _model_changes, choose_model_candidate, round_actions
from tests.test_route_balance import parallel_case


def reroute(solution, ri, indices):
    result = deepcopy(solution)
    for ti in indices:
        transfer = result["rounds"][ri]["transfers"][ti]
        transfer["path"] = [transfer["src"], "alternate", transfer["dst"]]
    return result


def paths(round_):
    return {(row["action"], row["src"], row["dst"], token): tuple(row["path"])
            for row in round_["transfers"] for token in row["chunk_ranges"]}


class AtomicOverlayTests(unittest.TestCase):
    def fixture(self, operator="all-gather"):
        task, graph = parallel_case(operator)
        baseline = build(task, graph, 4, kind="reverse_doubling" if operator == "all-gather" else "xor_tree",
                         resource_weight=0)
        # AR round 2 carries complete expressions with multiple contributors.
        ri = 1
        size = len(baseline["rounds"][ri]["transfers"])
        control = reroute(baseline, ri, range(size // 2))
        edited = reroute(baseline, ri, range(size // 2, size))
        return task, graph, baseline, control, edited, ri

    def checked(self, task, graph, solution):
        report = replay(task, graph, solution)
        self.assertTrue(report["valid"], report)
        return report

    def select(self, task, graph, baseline, control, edited, rounds):
        return choose_model_candidate(task, graph, control, self.checked(task, graph, control),
                                      edited, self.checked(task, graph, edited), rounds, baseline=baseline)

    def assert_manifest(self, evidence, baseline, control, edited, result):
        actual_positions = []
        count = 0
        for row in evidence["model_atomic_manifest"]:
            ri, ti = row["round"] - 1, row["transfer_index"]
            packet = result["rounds"][ri]["transfers"][ti]
            self.assertEqual(packet["path"], row["model_path"])
            self.assertEqual([packet["chunk_ranges"][ci] for ci in row["chunk_indices"]], row["chunks"])
            before, after, ctrl = paths(baseline["rounds"][ri]), paths(edited["rounds"][ri]), paths(control["rounds"][ri])
            for token, ci in zip(row["chunks"], row["chunk_indices"]):
                key = row["action"], row["src"], row["dst"], token
                self.assertEqual(before[key], tuple(row["model_before_path"]))
                self.assertEqual(after[key], tuple(row["model_path"]))
                self.assertEqual(ctrl[key], tuple(row["control_before_path"]))
                self.assertNotEqual(before[key], after[key])
                actual_positions.append([ri, ti, ci])
                count += 1
        self.assertEqual(evidence["model_atomic_count"], count)
        self.assertEqual(evidence["selected_atomic_positions"], sorted(actual_positions))
        self.assertEqual(evidence["selected_transfers"],
                         [list(pair) for pair in sorted({tuple(pos[:2]) for pos in actual_positions})])

    def test_same_round_complementary_routes_beat_both_with_reordered_packets(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline, control, edited, ri = self.fixture(operator)
                control["rounds"][ri]["transfers"].reverse()
                if operator == "all-gather":
                    # Same atoms now have different control packet indices and
                    # token positions; seed transfer indices cannot be reused.
                    packets = []
                    for row in control["rounds"][ri]["transfers"]:
                        tokens = list(reversed(row["chunk_ranges"]))
                        packets.extend([{**row, "chunk_ranges": tokens[:1]},
                                        {**row, "chunk_ranges": tokens[1:]}])
                    control["rounds"][ri]["transfers"] = packets
                saved = deepcopy((baseline, control, edited))
                result, report, evidence = self.select(task, graph, baseline, control, edited, [ri + 1])
                self.assertEqual(evidence["combination"], "atomic_overlay")
                self.assertLess(report["total_time_us"], self.checked(task, graph, control)["total_time_us"])
                self.assertLess(report["total_time_us"], self.checked(task, graph, edited)["total_time_us"])
                self.assertEqual(report, self.checked(task, graph, result))
                self.assertEqual(evidence["without_overlay_time_us"], self.checked(task, graph, control)["total_time_us"])
                for number, round_ in enumerate(result["rounds"]):
                    self.assertEqual(round_actions(round_), round_actions(control["rounds"][number]))
                changes = _model_changes(baseline["rounds"][ri], edited["rounds"][ri], control["rounds"][ri])
                final, ctrl = paths(result["rounds"][ri]), paths(control["rounds"][ri])
                for key in final:
                    self.assertEqual(final[key], changes[key][1] if key in changes else ctrl[key])
                self.assert_manifest(evidence, baseline, control, edited, result)
                self.assertEqual((baseline, control, edited), saved)

    def test_partial_packet_overlay_keeps_model_count_and_unmoved_control_path(self):
        task, graph, baseline, control, edited, ri = self.fixture()
        # Model changes half the atoms in the second half of baseline packets.
        for ti in range(len(edited["rounds"][ri]["transfers"]) // 2, len(edited["rounds"][ri]["transfers"])):
            row = edited["rounds"][ri]["transfers"][ti]
            row["chunk_ranges"] = row["chunk_ranges"][:len(row["chunk_ranges"]) // 2]
        # Build legal model packets from complete source packets, never drop data.
        packets = []
        half = len(baseline["rounds"][ri]["transfers"]) // 2
        for ti, original in enumerate(baseline["rounds"][ri]["transfers"]):
            if ti < half:
                packets.append(original)
            else:
                moved = edited["rounds"][ri]["transfers"][ti]
                packets.extend([moved, {**original, "chunk_ranges": original["chunk_ranges"][len(moved["chunk_ranges"]):]}])
        edited["rounds"][ri]["transfers"] = packets
        result, report, evidence = self.select(task, graph, baseline, control, edited, [ri + 1])
        self.assertEqual(evidence["combination"], "atomic_overlay")
        changes = _model_changes(baseline["rounds"][ri], edited["rounds"][ri], control["rounds"][ri])
        self.assertEqual(evidence["model_atomic_count"], len(changes))
        for key, final_path in paths(result["rounds"][ri]).items():
            self.assertEqual(final_path, changes[key][1] if key in changes else paths(control["rounds"][ri])[key])
        self.assert_manifest(evidence, baseline, control, edited, result)

    def test_already_routed_atoms_count_as_satisfied_without_inventing_new_moves(self):
        task, graph, baseline, control, edited, ri = self.fixture()
        size = len(control["rounds"][ri]["transfers"])
        control = reroute(control, ri, range(size * 3 // 4))
        result, _, evidence = self.select(task, graph, baseline, control, edited, [ri + 1])
        self.assertEqual(evidence["combination"], "atomic_overlay")
        self.assertGreater(evidence["model_atomic_count"], evidence["control_changed_atoms"])
        self.assertEqual(evidence["control_changed_atoms"], evidence["overlay_control_changed_atoms"])
        self.assert_manifest(evidence, baseline, control, edited, result)

    def test_all_model_targets_already_in_control_skip_partition_and_replay(self):
        task, graph, baseline, control, edited, ri = self.fixture()
        control = reroute(control, ri, range(len(control["rounds"][ri]["transfers"])))
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        with patch("src.selection.replay", side_effect=AssertionError("noop must not replay")):
            result, report, evidence = choose_model_candidate(task, graph, control, cr, edited, er,
                                                              [ri + 1], baseline=baseline)
        self.assertIs(result, control)
        self.assertIs(report, cr)
        self.assertFalse(evidence["selected"])
        self.assertEqual(evidence["overlay_rejections"]["no_changed_atoms"], 1)

    def test_joint_overlay_overloading_shared_pool_is_pruned_not_count_adjusted(self):
        task, graph, baseline, control, edited, ri = self.fixture()
        graph.resources["alternate-pool"] = 3e9
        for edge in graph.edges:
            if "alternate" in edge:
                graph.edge_resources[edge] = ("alternate-pool",)
        cr, er, br = (self.checked(task, graph, row) for row in (control, edited, baseline))
        self.assertLess(er["total_time_us"], br["total_time_us"])
        with patch("src.selection.replay", side_effect=AssertionError("cost-pruned overlay must not replay")):
            result, report, evidence = choose_model_candidate(task, graph, control, cr, edited, er,
                                                              [ri + 1], baseline=baseline)
        self.assertIs(result, control)
        self.assertIs(report, cr)
        self.assertEqual(evidence["overlay_rejections"]["not_better_round"], 1)
        self.assertNotIn("model_atomic_manifest", evidence)

    def test_duplicate_actions_are_not_deduplicated_or_ar_expressions_split(self):
        transfer = {"action": "reduce", "src": "r0", "dst": "r2", "path": ["r0", "s0", "r2"],
                    "chunk_ranges": ["r0_c0+r1_c0"]}
        old = {"transfers": [deepcopy(transfer), deepcopy(transfer)]}
        new = deepcopy(old); new["transfers"][0]["path"] = ["r0", "s1", "r2"]
        self.assertEqual(round_actions(old), round_actions(new))
        self.assertIsNone(_model_changes(old, new, old), "duplicate occurrence identity is ambiguous")
        split = {"transfers": [{**transfer, "chunk_ranges": ["r0_c0", "r1_c0"]}]}
        self.assertIsNone(_model_changes({"transfers": [transfer]}, split, split))

    def test_changed_baseline_layout_or_atom_program_disables_overlay(self):
        for change in ("split", "initial", "atom"):
            task, graph, baseline, control, edited, ri = self.fixture()
            if change == "split":
                baseline["split"] = 1
            elif change == "initial":
                baseline["initial_possession"] = {}
            else:
                baseline["rounds"][ri]["transfers"][0]["chunk_ranges"][0] = "unknown"
            result, report, evidence = self.select(task, graph, baseline, control, edited, [ri + 1])
            self.assertNotEqual(evidence["combination"], "atomic_overlay")
            self.assertEqual(report, self.checked(task, graph, result))
            self.assertTrue(evidence["overlay_rejections"])

    def test_model_same_path_noop_does_not_create_a_candidate(self):
        task, graph, baseline, control, _, ri = self.fixture()
        cr, er = self.checked(task, graph, control), self.checked(task, graph, baseline)
        with patch("src.selection.replay", side_effect=AssertionError("no path delta must not replay")):
            result, report, evidence = choose_model_candidate(task, graph, control, cr, baseline, er,
                                                              [ri + 1], baseline=baseline)
        self.assertIs(result, control)
        self.assertIs(report, cr)
        self.assertEqual(evidence["overlay_rejections"]["no_model_path_changes"], 1)

    def test_real_different_split_candidates_are_compared_whole_without_migration(self):
        task, graph, baseline, _, edited, ri = self.fixture()
        control = build(task, graph, 1, kind="reverse_doubling", resource_weight=0)
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        with patch("src.selection.replay", side_effect=AssertionError("different splits cannot overlay")):
            result, report, evidence = choose_model_candidate(task, graph, control, cr, edited, er,
                                                              [ri + 1], baseline=baseline)
        self.assertNotEqual(evidence["combination"], "atomic_overlay")
        self.assertEqual(evidence["overlay_rejections"]["incompatible_layout"], 1)
        self.assertEqual(report, self.checked(task, graph, result))
        self.assertEqual(evidence["without_overlay_time_us"], report["total_time_us"])

    def test_replayed_cost_overrules_a_promising_overlay_estimate(self):
        task, graph, baseline, control, edited, ri = self.fixture()
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        # A hypothetical disagreement with the cost prefilter must not let its
        # estimate override the independent validator's returned cost.
        with patch("src.selection.replay", return_value=deepcopy(cr)):
            result, report, evidence = choose_model_candidate(task, graph, control, cr, edited, er,
                                                              [ri + 1], baseline=baseline)
        self.assertIs(result, control)
        self.assertIs(report, cr)
        self.assertFalse(evidence["selected"])
        self.assertEqual(evidence["overlay_rejections"]["replay_not_better"], 1)
        self.assertNotIn("model_atomic_manifest", evidence)

    def test_overlay_gain_threshold_boundary_and_default_zero(self):
        task, graph, baseline, control, edited, ri = self.fixture()
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        args = task, graph, control, cr, edited, er, [ri + 1]
        expected, expected_report, default = choose_model_candidate(*args, baseline=baseline)
        self.assertEqual(default["combination"], "atomic_overlay")
        self.assertEqual(default["overlay_min_relative_gain"], 0.0)
        ratio = default["overlay_predicted_relative_gain"]
        self.assertGreater(ratio, 0)
        boundary, boundary_report, evidence = choose_model_candidate(
            *args, baseline=baseline, overlay_min_relative_gain=ratio)
        self.assertEqual(boundary, expected)
        self.assertEqual(boundary_report, expected_report)
        self.assertEqual(evidence["combination"], "atomic_overlay")
        with patch("src.selection.replay", side_effect=AssertionError("threshold-pruned overlay must not replay")):
            result, report, rejected = choose_model_candidate(
                *args, baseline=baseline, overlay_min_relative_gain=ratio + 1e-12)
        self.assertIs(result, control)
        self.assertIs(report, cr)
        self.assertEqual(rejected["overlay_rejections"]["below_min_relative_gain"], 1)
        self.assertEqual(rejected["overlay_predicted_time_us"], default["overlay_predicted_time_us"])
        self.assertEqual(rejected["without_overlay_time_us"] - rejected["overlay_predicted_time_us"],
                         rejected["overlay_predicted_gain_us"])

    def test_overlay_threshold_does_not_discard_a_valid_whole_model_winner(self):
        task, graph, baseline, _, edited, ri = self.fixture()
        control = deepcopy(baseline)
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        result, report, evidence = choose_model_candidate(task, graph, control, cr, edited, er,
            [ri + 1], baseline=baseline, overlay_min_relative_gain=1.0)
        self.assertIs(result, edited)
        self.assertIs(report, er)
        self.assertEqual(evidence["combination"], "whole_model")
        self.assertEqual(evidence["without_overlay_time_us"], er["total_time_us"])

    def test_invalid_overlay_threshold_is_rejected(self):
        task, graph, baseline, control, edited, ri = self.fixture()
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        for threshold in (-1, float("nan"), float("inf"), True, "0.1", None):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                choose_model_candidate(task, graph, control, cr, edited, er, [ri + 1],
                                       baseline=baseline, overlay_min_relative_gain=threshold)

    def test_failed_overlay_replay_preserves_the_actual_compatible_round_winner(self):
        task, graph, baseline, _, _, _ = self.fixture()
        n = len(baseline["rounds"][0]["transfers"])
        control = reroute(baseline, 0, range(n // 2))
        control = reroute(control, 2, range(len(control["rounds"][2]["transfers"])))
        edited = reroute(baseline, 0, range(n // 2, n))
        edited = reroute(edited, 1, range(len(edited["rounds"][1]["transfers"])))
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        original, original_report, original_evidence = choose_model_candidate(task, graph, control, cr, edited, er, [1, 2])
        self.assertEqual(original_evidence["combination"], "compatible_rounds")
        threshold_result, threshold_report, threshold_evidence = choose_model_candidate(
            task, graph, control, cr, edited, er, [1, 2], baseline=baseline, overlay_min_relative_gain=1.0)
        self.assertEqual(threshold_result, original)
        self.assertEqual(threshold_report, original_report)
        self.assertEqual(threshold_evidence["combination"], "compatible_rounds")
        self.assertEqual(threshold_evidence["overlay_rejections"]["below_min_relative_gain"], 1)
        calls = []

        def fail_overlay(task, graph, candidate):
            calls.append(candidate)
            return (replay(task, graph, candidate) if len(calls) == 1 else
                    {"valid": False, "errors": ["synthetic overlay failure"], "total_time_us": -1})

        with patch("src.selection.replay", side_effect=fail_overlay):
            result, report, evidence = choose_model_candidate(task, graph, control, cr, edited, er,
                                                              [1, 2], baseline=baseline)
        self.assertEqual(len(calls), 2, "one compatible candidate and one potential overlay winner")
        self.assertEqual(result, original)
        self.assertEqual(report, original_report)
        self.assertEqual(report, self.checked(task, graph, result))
        self.assertEqual(evidence["without_overlay_time_us"], original_report["total_time_us"])
        self.assertEqual(evidence["combination"], "compatible_rounds")
        self.assertEqual(evidence["overlay_rejections"]["replay_failed"], 1)
        self.assertNotIn("model_atomic_manifest", evidence)


if __name__ == "__main__":
    unittest.main()
