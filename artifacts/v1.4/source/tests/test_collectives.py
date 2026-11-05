"""Independent replay checks for locality-aware collective candidates."""

import copy
import unittest

from src.collectives import build, locality_order
from src.replay import replay
from src.topology import Topology
from tests.test_solver import make_case


class CollectiveTests(unittest.TestCase):
    def test_reverse_doubling_both_chunk_counts(self):
        for n in [1, 2, 4, 8, 16]:
            for split in [1, 4]:
                with self.subTest(n=n, split=split):
                    task, topology = make_case(n)
                    graph = Topology.from_data(topology)
                    answer = build(task, graph, split)
                    report = replay(task, graph, answer)
                    self.assertTrue(report["valid"], report["errors"])
                    self.assertEqual(answer["round_count"], (n - 1).bit_length())

    def test_locality_order_respects_actual_local_links(self):
        task, topology = make_case(8)
        expected_groups = [{"r0", "r2", "r4", "r6"}, {"r1", "r3", "r5", "r7"}]
        for group in expected_groups:
            for src in group:
                for dst in group:
                    if src != dst:
                        topology["edges"].append({"src": src, "dst": dst, "bw": "100GB/s"})
        graph = Topology.from_data(topology)
        original = copy.deepcopy(topology)
        order = locality_order(graph)
        self.assertIn(set(order[:4]), expected_groups)
        self.assertIn(set(order[4:]), expected_groups)
        self.assertEqual(topology, original)
        self.assertTrue(replay(task, graph, build(task, graph))["valid"])

    def test_rank_order_is_validated(self):
        task, topology = make_case(4)
        with self.assertRaises(ValueError):
            build(task, Topology.from_data(topology), order=["r0"] * 4)

    def test_xor_tree_disjoint_contributions_and_complete_broadcast(self):
        for n in [1, 2, 4, 8, 16, 256]:
            for split in [1, 4]:
                with self.subTest(n=n, split=split):
                    task, topology = make_case(n, "all-reduce")
                    graph = Topology.from_data(topology)
                    answer = build(task, graph, split, kind="xor_tree")
                    report = replay(task, graph, answer)
                    self.assertTrue(report["valid"], report["errors"])
                    self.assertEqual(answer["round_count"], (n - 1).bit_length() * 2)

    def test_round_limit_is_enforced(self):
        task, topology = make_case(8)
        task["max_nonempty_rounds"] = 2
        with self.assertRaises(ValueError):
            build(task, Topology.from_data(topology))

    def test_each_original_block_crosses_a_domain_cut_only_once(self):
        task, topology = make_case(8)
        topology["switches"] = ["left", "right"]
        topology["edges"] = [
            {"src": f"r{i}", "dst": "left" if i < 4 else "right",
             "bw": "100GB/s", "bidirectional": True} for i in range(8)]
        topology["edges"].append({"src": "left", "dst": "right", "bw": "1GB/s", "bidirectional": True})
        graph = Topology.from_data(topology)
        answer = build(task, graph)
        report = replay(task, graph, answer)
        self.assertTrue(report["valid"], report["errors"])
        cut_bytes = {("left", "right"): 0, ("right", "left"): 0}
        for round_ in answer["rounds"]:
            for transfer in round_["transfers"]:
                for edge in zip(transfer["path"], transfer["path"][1:]):
                    if edge in cut_bytes:
                        cut_bytes[edge] += len(transfer["chunk_ranges"]) * task["data_size"]["per_rank_bytes"]
        self.assertEqual(set(cut_bytes.values()), {4 * task["data_size"]["per_rank_bytes"]})

    def test_xor_roots_use_both_local_link_directions(self):
        task, topology = make_case(8, "all-reduce")
        graph = Topology.from_data(topology)
        answer = build(task, graph, 4, kind="xor_tree", order=graph.ranks)
        pairs = {(transfer["src"], transfer["dst"]) for transfer in answer["rounds"][0]["transfers"]}
        self.assertIn(("r0", "r1"), pairs)
        self.assertIn(("r1", "r0"), pairs)

    def test_packetization_preserves_semantics_and_reduces_path_serialization(self):
        for operator, kind in [("all-gather", "reverse_doubling"), ("all-reduce", "xor_tree")]:
            with self.subTest(operator=operator):
                task, topology = make_case(8, operator)
                graph = Topology.from_data(topology)
                split = 1 if operator == "all-gather" else 4
                ordinary = build(task, graph, split, kind=kind)
                packetized = build(task, graph, split, kind=kind, packets=4)
                expected = replay(task, graph, ordinary)
                actual = replay(task, graph, packetized)
                self.assertTrue(actual["valid"], actual["errors"])
                self.assertLess(actual["total_time_us"], expected["total_time_us"])
                self.assertEqual(sum(len(t["chunk_ranges"]) for r in ordinary["rounds"] for t in r["transfers"]),
                                 sum(len(t["chunk_ranges"]) for r in packetized["rounds"] for t in r["transfers"]))

    def test_routing_representative_does_not_change_payload_semantics(self):
        task, topology = make_case(16, "all-reduce")
        graph = Topology.from_data(topology)
        for split in [1, 4]:
            answer = build(task, graph, split, kind="xor_tree", routing_bytes=4096)
            report = replay(task, graph, answer)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(answer["split"], split)


if __name__ == "__main__":
    unittest.main()
