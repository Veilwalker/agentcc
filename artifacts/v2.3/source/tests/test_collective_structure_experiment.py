"""Independent semantics checks for the offline structure prototype."""

import copy
import unittest

from src.replay import replay
from src.topology import Topology
from tests.test_solver import make_case
from tests.collective_structure_experiment import (
    build_structure, candidate_specs, default_roots, dimension_order, equivalent_round_permutation,
    traffic_rounds,
)


class StructureExperimentTests(unittest.TestCase):
    def test_bit_dimensions_change_data_program_not_just_round_order(self):
        for operator in ("all-gather", "all-reduce"):
            task, data = make_case(8, operator)
            graph = Topology.from_data(data)
            before = copy.deepcopy((task, data))
            ordinary = build_structure(task, graph, 4)
            reordered = build_structure(task, graph, 4, dimensions=[2, 0, 1])
            self.assertTrue(replay(task, graph, reordered)["valid"])
            self.assertFalse(equivalent_round_permutation(ordinary, reordered))
            self.assertEqual((task, data), before)

    def test_whole_ar_expression_bijection_reaches_exact_requested_roots(self):
        task, data = make_case(8, "all-reduce")
        # Directed asymmetric terminal capacities exercise physical rerouting.
        data["edges"] = [{"src": f"r{i}", "dst": "S0", "bw": f"{i + 1}GB/s"}
                         for i in range(8)] + [
                            {"src": "S0", "dst": f"r{i}", "bw": f"{9 - i}GB/s"}
                            for i in range(8)]
        graph = Topology.from_data(data)
        roots = [7, 2, 6, 1]
        dims = [1, 2, 0]
        answer = build_structure(task, graph, 4, dimensions=dims, roots=roots)
        report = replay(task, graph, answer)
        self.assertTrue(report["valid"], report["errors"])
        actual_roots = {}
        for transfer in answer["rounds"][2]["transfers"]:
            for expression in transfer["chunk_ranges"]:
                terms = expression.split("+")
                self.assertEqual(len(terms), 4)
                self.assertEqual(len(set(terms)), 4)
                slot = int(terms[0].rsplit("_c", 1)[1])
                self.assertTrue(all(term.endswith(f"_c{slot}") for term in terms))
                actual_roots[slot] = transfer["dst"]
        order = dimension_order(graph.ranks, dims)
        self.assertEqual(actual_roots, {slot: order[root] for slot, root in enumerate(roots)})

    def test_equal_root_multiset_is_only_slot_relabeling_for_cost(self):
        task, data = make_case(8, "all-reduce")
        graph = Topology.from_data(data)
        roots = default_roots(8, 4)
        ordinary = build_structure(task, graph, 4, roots=roots)
        relabeled = build_structure(task, graph, 4, roots=roots[::-1])
        self.assertTrue(replay(task, graph, relabeled)["valid"])
        self.assertEqual(traffic_rounds(ordinary), traffic_rounds(relabeled))
        self.assertAlmostEqual(replay(task, graph, ordinary)["total_time_us"],
                               replay(task, graph, relabeled)["total_time_us"])

    def test_duplicate_dimensions_and_wrong_root_types_are_rejected(self):
        task, data = make_case(8, "all-reduce")
        graph = Topology.from_data(data)
        for dims in ([0, 0, 2], [0, 1], [True, 1, 2]):
            with self.assertRaises(ValueError):
                build_structure(task, graph, 4, dimensions=dims)
        for roots in ([1, 2], [0, 1, 2, 8], [True, 1, 2, 3]):
            with self.assertRaises(ValueError):
                build_structure(task, graph, 4, roots=roots)

    def test_catalogue_reads_physical_capacities_without_using_instance_name(self):
        task, data = make_case(8)
        data["switches"] = []
        data["edges"] = [{"src": f"r{i}", "dst": f"r{j}",
                          "bw": "100GB/s" if i ^ j == 1 else "1GB/s"}
                         for i in range(8) for j in range(8) if i != j]
        original = candidate_specs(task, Topology.from_data(data), 1)
        task["instance_id"] = "unrelated-name"
        task["task_id"] = "another-task-name"
        self.assertEqual(original, candidate_specs(task, Topology.from_data(data), 1))
        for edge in data["edges"]:
            edge["bw"] = "100GB/s" if int(edge["src"][1:]) ^ int(edge["dst"][1:]) == 4 else "1GB/s"
        changed = candidate_specs(task, Topology.from_data(data), 1)
        self.assertNotEqual(original[1], changed[1])
        self.assertEqual(min(range(3), key=original[1].__getitem__), 0)
        self.assertEqual(min(range(3), key=changed[1].__getitem__), 2)


if __name__ == "__main__":
    unittest.main()
