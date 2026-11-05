"""Independent raw-input parsing and Bellman-Ford routing cross-checks."""

from collections import defaultdict
import math
import random
import unittest

from src.topology import Topology


def reference_graph(data):
    """Small numeric-only oracle independent of Topology's derived structures."""
    capacities = defaultdict(float)
    resource_sets = defaultdict(set)
    pools = {pool["id"]: pool["capacity"] for pool in data.get("resources", [])}
    for edge in data["edges"]:
        directions = [(edge["src"], edge["dst"])]
        if edge.get("bidirectional"):
            directions.append((edge["dst"], edge["src"]))
        for direction in directions:
            capacities[direction] += edge["bw"]
            if "resource_id" in edge:
                resource_sets[direction].add(edge["resource_id"])
    for pool in data.get("resources", []):
        for member in pool.get("edges", []):
            resource_sets[member["src"], member["dst"]].add(pool["id"])
            if member.get("bidirectional"):
                resource_sets[member["dst"], member["src"]].add(pool["id"])
    return dict(capacities), {edge: resource_sets[edge] for edge in capacities}, pools


def reference_distance(data, reference, source, destination, payload, penalty):
    capacities, resource_sets, pools = reference
    nodes = {source, destination, *data["switches"]}
    weighted = []
    for (start, end), capacity in capacities.items():
        if start not in nodes or end not in nodes or start == destination:
            continue
        cost = payload * 1e6 / capacity
        cost += payload * 1e6 * penalty * sum(1 / pools[name] for name in resource_sets[start, end])
        if end in data["switches"]:
            cost += data["switch_processing_overhead_us"]
        weighted.append((start, end, cost))
    distances = dict.fromkeys(nodes, math.inf)
    distances[source] = 0
    for _ in range(len(nodes) - 1):
        changed = False
        for start, end, cost in weighted:
            proposed = distances[start] + cost
            if proposed < distances[end]:
                distances[end] = proposed
                changed = True
        if not changed:
            break
    return distances[destination], {(start, end): cost for start, end, cost in weighted}


def generated_topology(randomizer, index):
    # Node names intentionally do not encode roles or rank ordering.
    ranks = ["z-rank", "a-rank", "mid"]
    randomizer.shuffle(ranks)
    switches = ["r0", "a0", "m0", "s0"]
    randomizer.shuffle(switches)
    nodes = ranks + switches
    pools = [{"id": f"pool{i}", "capacity": randomizer.choice([1e6, 4e6, 16e6]), "edges": []}
             for i in range(3)]
    edges = []
    present = set()
    for source in nodes:
        for destination in nodes:
            if source == destination or randomizer.random() >= 0.28:
                continue
            edge = {"src": source, "dst": destination, "bw": randomizer.choice([1e6, 2e6, 8e6]),
                    "bidirectional": randomizer.random() < 0.3}
            if randomizer.random() < 0.3:
                edge["resource_id"] = randomizer.choice(pools)["id"]
            edges.append(edge)
            present.add((source, destination))
            if edge["bidirectional"]:
                present.add((destination, source))
            if randomizer.random() < 0.15:
                edges.append({"src": source, "dst": destination, "bw": 3e6})
    for pool in pools:
        for source, destination in sorted(present):
            if randomizer.random() >= 0.22:
                continue
            reference = {"src": source, "dst": destination}
            if (destination, source) in present and randomizer.random() < 0.2:
                reference["bidirectional"] = True
            pool["edges"].append(reference)
            if randomizer.random() < 0.1:
                pool["edges"].append(dict(reference))
    return {"ranks": ranks, "switches": switches, "edges": edges, "resources": pools,
            "switch_processing_overhead_us": [0, 0.3, 5][index % 3]}


class TopologyStressTests(unittest.TestCase):
    def test_random_raw_inputs_and_bellman_ford(self):
        randomizer = random.Random(20261103)
        for index in range(150):
            data = generated_topology(randomizer, index)
            reference = reference_graph(data)
            graph = Topology.from_data(data)
            graph._path_cache_limit = 5  # Exercise LRU eviction while varying payload and weights.
            self.assertEqual(graph.edges, reference[0])
            self.assertEqual({edge: set(ids) for edge, ids in graph.edge_resources.items()}, reference[1])
            self.assertEqual(graph.resources, reference[2])
            for source in data["ranks"]:
                for destination in data["ranks"]:
                    if source == destination:
                        continue
                    for payload in (1, 4096, 1048576):
                        for penalty in (0, 0.5, 2, 4):
                            expected, weighted_edges = reference_distance(
                                data, reference, source, destination, payload, penalty)
                            with self.subTest(graph=index, src=source, dst=destination,
                                              payload=payload, penalty=penalty):
                                if math.isinf(expected):
                                    with self.assertRaises(ValueError):
                                        graph.path(source, destination, payload, penalty)
                                    continue
                                path = graph.path(source, destination, payload, penalty)
                                actual = sum(weighted_edges[start, end] for start, end in zip(path, path[1:]))
                                self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8),
                                                (actual, expected, path))
                                self.assertEqual(path[0], source)
                                self.assertEqual(path[-1], destination)
                                self.assertEqual(len(path), len(set(path)))
                                self.assertTrue(all(node in data["switches"] for node in path[1:-1]))
            self.assertLessEqual(len(graph._paths), 5)
            self.assertLessEqual(len(graph._weighted_adjacency), 8)
            self.assertEqual(set(graph._weighted_adjacency), set(graph._rank_exit_bounds))

    def test_source_trees_survive_weight_cache_eviction(self):
        data = {"ranks": ["r0", "r1", "r2"], "switches": ["s"],
                "edges": [{"src": rank, "dst": "s", "bw": 1e9, "bidirectional": True}
                          for rank in ("r0", "r1", "r2")]}
        graph = Topology.from_data(data)
        first = graph.path("r0", "r1", 1, 0)
        for payload in range(2, 15):
            graph.path("r0", "r1", payload, 0)
        self.assertEqual(graph.path("r0", "r1", 1, 0), first)
        self.assertEqual(graph.path("r2", "r1", 1, 0), ["r2", "s", "r1"])

    def test_zero_or_nonfinite_capacity_is_rejected(self):
        for capacity in (0, -1, True, math.nan, math.inf):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    Topology.from_data({"ranks": ["r0", "r1"], "edges": [
                        {"src": "r0", "dst": "r1", "bw": capacity}]})
                with self.assertRaises(ValueError):
                    Topology.from_data({"ranks": ["r0", "r1"], "resources": [
                        {"id": "invalid", "capacity": capacity}], "edges": []})

    def test_duplicate_capacity_overflow_and_cost_overflow_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "merged edge bandwidth"):
            Topology.from_data({"ranks": ["r0", "r1"], "edges": [
                {"src": "r0", "dst": "r1", "bw": 1e308},
                {"src": "r0", "dst": "r1", "bw": 1e308}]})
        graph = Topology.from_data({"ranks": ["r0", "r1"], "edges": [
            {"src": "r0", "dst": "r1", "bw": 1e-320}]})
        with self.assertRaisesRegex(ValueError, "finite numeric range"):
            graph.path("r0", "r1", 1e308)
        self.assertFalse(graph._weighted_adjacency)


if __name__ == "__main__":
    unittest.main()
