"""Independent safety boundaries for competition against a strong control."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.selection import choose_model_candidate, round_actions
from tests.test_route_balance import parallel_case


def routed(solution, indices):
    result = deepcopy(solution)
    for index in indices:
        for transfer in result["rounds"][index]["transfers"]:
            transfer["path"] = [transfer["src"], "alternate", transfer["dst"]]
    return result


class SelectionInvariantTests(unittest.TestCase):
    def fixture(self, operator="all-gather", split=4):
        task, graph = parallel_case(operator)
        kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
        seed = build(task, graph, split, kind=kind, resource_weight=0)
        return task, graph, seed

    def checked(self, task, graph, solution):
        report = replay(task, graph, solution)
        self.assertTrue(report["valid"], report)
        return report

    def test_complementary_rounds_beat_both_with_empty_rounds_and_different_packets(self):
        for operator in ("all-gather", "all-reduce"):
            task, graph, seed = self.fixture(operator)
            seed["rounds"].insert(0, {"round": 0, "transfers": []})
            for number, round_ in enumerate(seed["rounds"], 1):
                round_["round"] = number
            seed["round_count"] = len(seed["rounds"])
            control = routed(seed, [1])
            edited = routed(seed, [len(seed["rounds"]) - 1])
            # Chunk partitioning can differ while each whole AR expression and
            # each AG chunk remains an indivisible atomic action.
            packets = []
            for transfer in edited["rounds"][-1]["transfers"]:
                packets.extend({**transfer, "chunk_ranges": [token]} for token in transfer["chunk_ranges"])
            edited["rounds"][-1]["transfers"] = packets
            before_control, before_model = deepcopy(control), deepcopy(edited)
            cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
            chosen, checked, evidence = choose_model_candidate(task, graph, control, cr, edited, er,
                                                               [len(seed["rounds"])])
            self.assertEqual(evidence["combination"], "compatible_rounds")
            self.assertLess(checked["total_time_us"], min(cr["total_time_us"], er["total_time_us"]))
            self.assertEqual(checked, self.checked(task, graph, chosen))
            self.assertEqual(chosen["rounds"][1], control["rounds"][1])
            self.assertEqual(chosen["rounds"][-1], edited["rounds"][-1])
            self.assertEqual(control, before_control)
            self.assertEqual(edited, before_model)

    def test_different_split_compares_whole_candidates_without_crossing_states(self):
        task, graph, control = self.fixture(split=1)
        edited = build(task, graph, 4, kind="reverse_doubling", resource_weight=0)
        edited = routed(edited, range(len(edited["rounds"])))
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        self.assertLess(er["total_time_us"], cr["total_time_us"])
        with patch("src.selection.replay", side_effect=AssertionError("incompatible states cannot merge")):
            chosen, checked, evidence = choose_model_candidate(task, graph, control, cr, edited, er,
                                                               list(range(1, len(edited["rounds"]) + 1)))
        self.assertIs(chosen, edited)
        self.assertIs(checked, er)
        self.assertEqual(evidence["combination"], "whole_model")

    def test_rank_order_action_difference_prevents_round_replacement(self):
        task, graph, control = self.fixture()
        order = list(graph.ranks)
        order[0], order[1] = order[1], order[0]
        edited = build(task, graph, 4, kind="reverse_doubling", resource_weight=0, order=order)
        edited = routed(edited, range(len(edited["rounds"])))
        # At least one action-incompatible round cannot be spliced even when
        # its initial/final ownership and split match exactly.
        incompatible = [i + 1 for i, (left, right) in enumerate(zip(control["rounds"], edited["rounds"]))
                        if round_actions(left) != round_actions(right)]
        self.assertTrue(incompatible)
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        with patch("src.selection.replay", side_effect=AssertionError("different actions cannot merge")):
            chosen, checked, evidence = choose_model_candidate(task, graph, control, cr, edited, er, incompatible)
        self.assertIs(chosen, edited)
        self.assertIs(checked, er)
        self.assertEqual(evidence["combination"], "whole_model")

    def test_stronger_unrestricted_control_is_never_sacrificed_for_model_attribution(self):
        task, graph, seed = self.fixture()
        control = routed(seed, range(len(seed["rounds"])))
        edited = routed(seed, [0])
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        self.assertLess(er["total_time_us"], self.checked(task, graph, seed)["total_time_us"])
        chosen, checked, evidence = choose_model_candidate(task, graph, control, cr, edited, er, [1])
        self.assertIs(chosen, control)
        self.assertIs(checked, cr)
        self.assertFalse(evidence["selected"])
        self.assertEqual(evidence["selected_rounds"], [])

    def test_failed_combination_keeps_report_of_actual_valid_winner(self):
        task, graph, seed = self.fixture()
        control = routed(seed, [0])
        edited = routed(seed, [len(seed["rounds"]) - 1])
        cr, er = self.checked(task, graph, control), self.checked(task, graph, edited)
        with patch("src.selection.replay", return_value={"valid": False, "errors": ["synthetic"],
                                                         "total_time_us": -1}):
            chosen, checked, evidence = choose_model_candidate(task, graph, control, cr, edited, er,
                                                               [len(seed["rounds"])])
        self.assertEqual(checked, self.checked(task, graph, chosen))
        self.assertTrue(evidence["errors"])
        self.assertNotEqual(checked["total_time_us"], -1)

    def test_atom_action_comparison_retains_expression_and_duplicate_boundaries(self):
        template = {"action": "reduce", "src": "r0", "dst": "r2"}
        packed = {"transfers": [{**template, "chunk_ranges": ["r0_c0+r1_c0"]}]}
        split = {"transfers": [{**template, "chunk_ranges": ["r0_c0", "r1_c0"]}]}
        doubled = {"transfers": [{**template, "chunk_ranges": ["r0_c0+r1_c0"]}] * 2}
        self.assertNotEqual(round_actions(packed), round_actions(split))
        self.assertNotEqual(round_actions(packed), round_actions(doubled))


if __name__ == "__main__":
    unittest.main()
