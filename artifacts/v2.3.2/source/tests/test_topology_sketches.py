"""Ownership, accessibility and numerical contracts for the 2.0 communication sketches."""
import unittest
from unittest.mock import patch

from src import collectives
from src.collectives import _initial
from src.fabric_routes import RouteCatalog
from src.group_sketch import AxisExchange, direct_axes
from src.refined_search import RefinedSearchSpace
from src.replay import replay
from src.stage_program import StageSpace
from src.topology import Topology
from tests.test_solver import make_case


class TopologySketchTests(unittest.TestCase):
    def test_two_local_axes_exchange_disjoint_whole_atoms(self):
        task, data = make_case(16)
        for i in range(16):
            for j in range(i):
                if i // 4 == j // 4 or i % 4 == j % 4:
                    data['edges'].append({'src': f'r{i}', 'dst': f'r{j}', 'bw': '4GB/s',
                                          'bidirectional': True})
        graph = Topology.from_data(data)
        space = StageSpace(task, graph, 4)
        self.assertEqual(direct_axes(space), (4, 4))
        for partition in (1, 2, 3):
            with self.subTest(partition=partition):
                sketch = AxisExchange(space, 4, 4, partition)
                answer = _initial(task, list(graph.ranks), 4)
                answer.update(round_count=2, rounds=[{'round': i, 'transfers': row}
                              for i, row in enumerate(sketch.compile(), 1)])
                result = replay(task, graph, answer)
                self.assertTrue(result['valid'], result['errors'])
                self.assertAlmostEqual(sketch.cost, result['total_time_us'], places=6)
        graph.edges.pop(('r0', 'r1'))
        self.assertIsNone(direct_axes(space))

    def test_phase_bridge_keeps_all_contributors_at_each_new_root(self):
        task, data = make_case(16, 'all-reduce')
        space = RefinedSearchSpace(task, Topology.from_data(data), 4)
        self.assertIn((2, 4), space.profiles[0].meetings)
        answer, predicted = space.compile([0, [1, 1, 4, 1, 1]])
        self.assertEqual(answer['round_count'], 5)
        self.assertEqual({t['action'] for t in answer['rounds'][2]['transfers']}, {'reduce'})
        result = replay(task, data, answer)
        self.assertTrue(result['valid'], result['errors'])
        self.assertAlmostEqual(predicted, result['total_time_us'], places=6)

    def test_remote_gateway_hidden_by_several_local_ports(self):
        _, data = make_case(2)
        data['switches'] = ['L0', 'L1', 'L2', 'A', 'B', 'X', 'Y']
        data['edges'] = [{'src':a, 'dst':b, 'bw':'4GB/s'} for a,b in
                         [('r0','L0'), ('r0','L1'), ('r0','L2'), ('r0','A'),
                          ('A','X'), ('A','Y'), ('X','B'), ('Y','B'), ('B','r1')]]
        graph = Topology.from_data(data)
        catalog = RouteCatalog(graph, 8192, branch_ports=True)
        self.assertEqual(set(catalog.options('r0','r1')),
                         {('r0','A','X','B','r1'), ('r0','A','Y','B','r1')})
        self.assertEqual(catalog.options('r1','r0'), ())

    @unittest.skipUnless(getattr(collectives.native, 'locality_permutation', None), 'C++ extension required')
    def test_cpp_locality_preserves_reference_ties_and_asymmetry(self):
        _, data = make_case(16)
        for i, edge in enumerate(data['edges']):
            edge['bw'] = (1 + i % 3) * 1e9
        data['edges'].append({'src':'r9','dst':'r3','bw':'6GB/s'})
        graph = Topology.from_data(data)
        current = collectives.locality_order(graph)
        with patch.object(collectives, 'native', None):
            self.assertEqual(current, collectives.locality_order(graph))
