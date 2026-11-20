"""Differential routing checks, including tie order and terminal ranks."""
import random
import unittest
from unittest.mock import patch

from src import topology
from src.capacity_backend import native
from src.topology import Topology


@unittest.skipUnless(native is not None and hasattr(native, 'shortest_tree'), 'build routing extension')
class NativeRoutingTests(unittest.TestCase):
    def test_random_directed_shared_graphs_match_reference(self):
        rng = random.Random(1107)
        for iteration in range(12):
            ranks = ['r'+str(i) for i in range(8)]
            switches = ['s'+str(i) for i in range(10)]
            edges = []
            for a in ranks + switches:
                for b in ranks + switches:
                    if a != b and (rng.random() < .20 or a in ranks and b == 's0'
                                   or a == 's0' and b in ranks):
                        edges.append({'src': a, 'dst': b, 'bw': rng.choice([1e8, 2e8, 4e8])})
            data = {'ranks': ranks, 'switches': switches, 'edges': edges,
                    'switch_processing_overhead_us': rng.choice([0, .3]),
                    'resources': [{'id': 'shared', 'capacity': 3e8,
                                   'edges': [{'src': e['src'], 'dst': e['dst']} for e in edges[::3]]}]}
            fast, reference = Topology.from_data(data), Topology.from_data(data)
            for size, weight in [(1, 0), (4096, 4), (4096, 1), (1, 0)]:
                for a in ranks:
                    for b in ranks:
                        if a == b:
                            continue
                        actual = fast.path(a, b, size, weight)
                        with patch.object(topology, 'native', None):
                            expected = reference.path(a, b, size, weight)
                        self.assertEqual(actual, expected, (iteration, a, b, size, weight))
                        self.assertTrue(all(n in switches for n in actual[1:-1]))

    def test_bad_graphs_and_priority_dimensions_raise(self):
        for value in [[], [[(2, 1.)]], [[(0, -1.)]], [[(0, float('nan'))]]]:
            with self.assertRaises(ValueError):
                native.compile_adjacency(value)
        graph = native.compile_adjacency([[(1, 1.)], []])
        for source, ranks, priorities in [(2, 1, [0, 1]), (0, 0, [0, 1]), (0, 1, [0]), (0, 1, [0, 2**33])]:
            with self.assertRaises(ValueError):
                native.shortest_tree(graph, source, ranks, priorities)


if __name__ == '__main__':
    unittest.main()
