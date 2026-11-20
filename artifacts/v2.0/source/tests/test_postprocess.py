"""The capacity chain must validate once and retain its genuine incumbent."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.collectives import build
from src.packetization import repacketize
from src.postprocess import polish, compare_with_control
from src.replay import replay
from src.topology import Topology
from tests.test_solver import make_case
from tests.test_route_balance import parallel_case


class PostprocessTests(unittest.TestCase):
    def fixture(self):
        task, topology = make_case(4)
        graph = Topology.from_data(topology)
        solution = build(task, graph, 4, packets=1)
        report = replay(task, graph, solution)
        self.assertTrue(report["valid"])
        return task, graph, solution, report

    def test_real_packet_candidate_has_one_final_replay_and_correct_metrics(self):
        task, graph, solution, report = self.fixture()
        original = deepcopy(solution)
        with patch("src.postprocess.replay", wraps=replay) as validator:
            final, checked, evidence = polish(task, graph, solution, report)
        self.assertTrue(evidence["accepted"])
        self.assertEqual(validator.call_count, 1)
        self.assertEqual(evidence["replay_calls"], 1)
        self.assertLess(checked["total_time_us"], report["total_time_us"])
        self.assertEqual(checked, replay(task, graph, final))
        self.assertAlmostEqual(final["metrics"]["total_time"] * 1e6, checked["total_time_us"])
        self.assertEqual(solution, original)

    def test_material_cheap_gain_skips_routing_search(self):
        task, graph, solution, report = self.fixture()
        with patch("src.postprocess.propose_joint") as joint:
            final, checked, evidence = polish(task, graph, solution, report)
        joint.assert_not_called()
        self.assertTrue(evidence["accepted"])
        self.assertEqual(evidence["joint"]["reason"], "cheap_capacity_gain")
        self.assertEqual(checked, replay(task, graph, final))

    def test_no_op_reuses_report_without_replay(self):
        task, graph, solution, report = self.fixture()
        with patch("src.postprocess.propose_joint", return_value=(solution, {})), \
                patch("src.postprocess.repacketize", return_value=(solution, {})), \
                patch("src.postprocess.replay") as validator:
            final, checked, evidence = polish(task, graph, solution, report)
        self.assertIs(final, solution)
        self.assertIs(checked, report)
        validator.assert_not_called()
        self.assertEqual(evidence["replay_calls"], 0)

    def test_changed_route_cannot_reuse_stale_report_for_packet_pass(self):
        task, graph = parallel_case()
        solution = build(task, graph, 4, resource_weight=0, packets=1)
        report = replay(task, graph, solution)
        routed = deepcopy(solution)
        for row in routed["rounds"]:
            for transfer in row["transfers"]:
                transfer["path"] = [transfer["src"], "alternate", transfer["dst"]]
        with patch("src.postprocess.propose_joint", return_value=(routed, {})), \
                patch("src.postprocess.repacketize", side_effect=lambda t, g, s, r, **kw: (s, {"predicted_after_us": r["total_time_us"]}) if r is not None else repacketize(t, g, s, r, **kw)) as packets, \
                patch("src.postprocess.replay", wraps=replay) as validator:
            final, checked, evidence = polish(task, graph, solution, report)
        self.assertIsNone(packets.call_args.args[3])
        self.assertEqual(validator.call_count, 1)
        self.assertTrue(evidence["accepted"])
        self.assertEqual(checked, replay(task, graph, final))

    def test_invalid_proposal_and_exception_keep_original_actual_report(self):
        task, graph, solution, report = self.fixture()
        broken = deepcopy(solution)
        broken["rounds"][0]["transfers"].pop()
        for raises in (False, True):
            joint = {"side_effect": ValueError("test")} if raises else {"return_value": (broken, {})}
            with self.subTest(raises=raises), patch("src.postprocess.propose_joint", **joint), \
                    patch("src.postprocess.repacketize", return_value=(broken, {})):
                final, checked, evidence = polish(task, graph, solution, report)
            self.assertIs(final, solution)
            self.assertIs(checked, report)
            self.assertFalse(evidence["accepted"])

    def test_previous_full_winner_survives_a_worse_new_overlay(self):
        task, graph, solution, report = self.fixture()
        old = deepcopy(solution)
        strong = deepcopy(solution)
        old_report = {**report, "total_time_us": 10.0}
        strong_report = {**report, "total_time_us": 15.0}
        selected = {"selected": True, "combination": "atomic_overlay", "selected_rounds": [1]}
        unchanged = lambda _t, _g, s, r, **kw: (s, r, {"accepted": False})
        with patch("src.selection.choose_model_candidate", side_effect=[
                (old, old_report, selected), (strong, strong_report, selected)]), \
                patch("src.postprocess.polish", side_effect=unchanged):
            final, checked, evidence = compare_with_control(
                task, graph, solution, solution, report, strong, strong_report,
                old, old_report, [1])
        self.assertIs(final, old)
        self.assertIs(checked, old_report)
        self.assertEqual(evidence["capacity_comparison"]["legacy_winner_us"], 10.0)
        self.assertEqual(evidence["control_time_us"], 15.0)


if __name__ == "__main__":
    unittest.main()
