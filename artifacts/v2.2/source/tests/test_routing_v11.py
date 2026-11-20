"""Routing checks against exhaustive simple-path search on small directed graphs."""

import hashlib
import math
import random
import unittest
from unittest.mock import patch

from src.topology import Topology


def route_score(graph, source, route, size, resource_weight):
    cost = 0.0
    tie = 0
    for src, dst in zip(route, route[1:]):
        shared = sum(1 / graph.resources[r] for r in graph.edge_resources[src, dst])
        cost += size * 1e6 * (1 / graph.edges[src, dst] + resource_weight * shared)
        if dst in graph.switches:
            cost += graph.switch_overhead_us
        tie += int.from_bytes(hashlib.sha256((source + "\0" + dst).encode()).digest()[:4], "big")
    return cost, tie


def exhaustive_score(graph, source, destination, size, resource_weight):
    best = (math.inf, 0)
    pending = [[source]]
    while pending:
        route = pending.pop()
        node = route[-1]
        if node == destination:
            best = min(best, route_score(graph, source, route, size, resource_weight))
            continue
        for neighbor, _, _ in graph.adjacency[node]:
            if neighbor in route:
                continue
            if neighbor != destination and neighbor not in graph.switches:
                continue
            pending.append(route + [neighbor])
    return best


class RoutingV11Tests(unittest.TestCase):
    def test_matches_exhaustive_directed_paths(self):
        randomizer = random.Random(731)
        nodes = ["r0", "r1", "r2", "a", "b", "c"]
        for graph_index in range(16):
            edges = []
            resource_members = []
            for source in nodes:
                for destination in nodes:
                    if source == destination or randomizer.random() > 0.35:
                        continue
                    edges.append({"src": source, "dst": destination,
                                  "bw": randomizer.choice([1e6, 2e6, 8e6])})
                    if randomizer.random() < 0.35:
                        resource_members.append({"src": source, "dst": destination})
            graph = Topology.from_data({"ranks": nodes[:3], "switches": nodes[3:],
                "edges": edges, "switch_processing_overhead_us": 0.3,
                "resources": [{"id": "shared", "capacity": 1e6, "edges": resource_members}]})
            for source in nodes[:3]:
                for destination in nodes[:3]:
                    if source == destination:
                        continue
                    for size in (1, 1024):
                        for weight in (0, 0.5, 2):
                            expected = exhaustive_score(graph, source, destination, size, weight)
                            with self.subTest(graph=graph_index, src=source, dst=destination,
                                              size=size, resource_weight=weight):
                                if math.isinf(expected[0]):
                                    with self.assertRaises(ValueError):
                                        graph.path(source, destination, size, weight)
                                    continue
                                route = graph.path(source, destination, size, weight)
                                actual = route_score(graph, source, route, size, weight)
                                self.assertAlmostEqual(actual[0], expected[0], places=8)
                                self.assertEqual(len(route), len(set(route)))
                                self.assertTrue(all(node in graph.switches for node in route[1:-1]))

    def test_payload_sensitive_overhead_is_not_cached_as_equivalent(self):
        graph = Topology.from_data({"ranks": ["r0", "r1"], "switches": ["a", "b", "c"],
            "switch_processing_overhead_us": 10,
            "edges": [{"src": src, "dst": dst, "bw": bandwidth} for src, dst, bandwidth in [
                ("r0", "a", 1e6), ("a", "r1", 1e6), ("r0", "b", 1e9),
                ("b", "c", 1e9), ("c", "r1", 1e9)]]})
        self.assertEqual(graph.path("r0", "r1", 1, 0), ["r0", "a", "r1"])
        self.assertEqual(graph.path("r0", "r1", 1000, 0), ["r0", "b", "c", "r1"])

    def test_equivalent_sizes_reuse_tree_only_without_overhead(self):
        graph = Topology.from_data({"ranks": ["r0", "r1"], "switches": ["s"],
            "switch_processing_overhead_us": 0,
            "edges": [{"src": "r0", "dst": "s", "bw": 1e9},
                      {"src": "s", "dst": "r1", "bw": 1e9}]})
        self.assertEqual(graph.path("r0", "r1", 1, 0), graph.path("r0", "r1", 1e6, 4))
        self.assertEqual(len(graph._paths), 1)

    def test_only_switches_enter_heap(self):
        graph = Topology.from_data({"ranks": ["r0", "r1", "r2"], "switches": ["s"],
            "edges": [{"src": rank, "dst": "s", "bw": 1e9, "bidirectional": True}
                      for rank in ("r0", "r1", "r2")]})
        import heapq
        pushes = []
        original_push = heapq.heappush
        def tracked_push(queue, item):
            pushes.append(item[2])
            return original_push(queue, item)
        # Instrument the Python reference heap; native routing is checked
        # separately against this reference in test_native_routing.py.
        with patch("src.topology.native", None), \
             patch("src.topology.heapq.heappush", side_effect=tracked_push):
            graph.path("r0", "r1")
        self.assertTrue(pushes)
        self.assertTrue(all(index >= graph._rank_count for index in pushes))

    def test_equal_paths_are_deterministic_and_spread_across_sources(self):
        ranks = [f"r{i}" for i in range(32)]
        topology = {"ranks": ranks, "switches": ["a", "b"],
            "edges": [{"src": rank, "dst": switch, "bw": 1e9, "bidirectional": True}
                      for rank in ranks for switch in ("a", "b")]}
        graph = Topology.from_data(topology)
        routes = [graph.path(rank, "r31", 1024, 1) for rank in ranks[:-1]]
        self.assertEqual({route[1] for route in routes}, {"a", "b"})
        second = Topology.from_data(topology)
        self.assertEqual(routes, [second.path(rank, "r31", 1024, 1) for rank in ranks[:-1]])
        for source, route in zip(ranks, routes):
            actual = route_score(graph, source, route, 1024, 1)
            self.assertEqual(actual, exhaustive_score(graph, source, "r31", 1024, 1))

    def test_direct_edge_is_fast_only_when_provably_cheapest(self):
        topology = {"ranks": ["r0", "r1"], "switches": ["s"], "edges": [
            {"src": "r0", "dst": "r1", "bw": 1e6},
            {"src": "r0", "dst": "s", "bw": 1e9},
            {"src": "s", "dst": "r1", "bw": 1e9}]}
        graph = Topology.from_data(topology)
        self.assertEqual(graph.path("r0", "r1", 1024), ["r0", "s", "r1"])
        topology["edges"][0]["bw"] = 1e12
        graph = Topology.from_data(topology)
        self.assertEqual(graph.path("r0", "r1", 1024), ["r0", "r1"])
        self.assertFalse(graph._paths)


if __name__ == "__main__":
    unittest.main()
