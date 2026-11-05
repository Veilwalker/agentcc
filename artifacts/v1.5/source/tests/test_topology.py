import unittest

from src.topology import Topology, bandwidth


class TopologyTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(bandwidth("8Gbps"), bandwidth("1GB/s"))
        self.assertEqual(bandwidth("8Gbit/s"), 1e9)
        for value in [0, -1, True, float("nan"), "unlimited"]:
            with self.assertRaises(ValueError):
                bandwidth(value)

    def test_rank_cannot_be_transit(self):
        graph = Topology.from_data({"ranks": ["r0", "r1", "r2"], "edges": [
            {"src": "r0", "dst": "r1", "bw": 1},
            {"src": "r1", "dst": "r2", "bw": 1}]})
        with self.assertRaises(ValueError):
            graph.path("r0", "r2")

    def test_duplicate_bandwidth_and_resources(self):
        graph = Topology.from_data({"ranks": ["r0", "r1"], "edges": [
            {"src": "r0", "dst": "r1", "bw": 2, "bidirectional": True, "resource_id": "pool"},
            {"src": "r0", "dst": "r1", "bw": 3}], "resources": [
            {"id": "pool", "capacity": 4},
            {"id": "out", "capacity": 5, "edges": [{"src": "r0", "dst": "r1"}]}]})
        self.assertEqual(graph.edges["r0", "r1"], 5)
        self.assertEqual(graph.edges["r1", "r0"], 2)
        self.assertEqual(graph.edge_resources["r0", "r1"], ("out", "pool"))
        self.assertEqual(graph.edge_resources["r1", "r0"], ("pool",))

    def test_shared_capacity_changes_route(self):
        edges = [{"src": src, "dst": dst, "bw": bw} for src, dst, bw in [
            ("r0", "a", 100), ("a", "r1", 100), ("r0", "b", 50), ("b", "r1", 50)]]
        graph = Topology.from_data({"ranks": ["r0", "r1"], "switches": ["a", "b"],
            "edges": edges, "resources": [{"id": "tight", "capacity": 1,
            "edges": [{"src": "r0", "dst": "a"}]}]})
        self.assertEqual(graph.path("r0", "r1", resource_weight=0), ["r0", "a", "r1"])
        self.assertEqual(graph.path("r0", "r1", resource_weight=1), ["r0", "b", "r1"])

    def test_resource_does_not_create_edge(self):
        with self.assertRaises(ValueError):
            Topology.from_data({"ranks": ["r0", "r1"], "edges": [], "resources": [
                {"id": "invalid", "capacity": 1, "edges": [{"src": "r0", "dst": "r1"}]}]})


if __name__ == "__main__":
    unittest.main()
