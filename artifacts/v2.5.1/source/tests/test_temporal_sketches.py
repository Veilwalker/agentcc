"""Check the native gather verifier against the independent symbolic reference."""
from copy import deepcopy
import unittest

from src.balanced_stages import BalancedStageSpace
from src.capacity_backend import native
from src.collectives import _initial
from src.group_sketch import GroupBroadcast
from src.replay import certify, replay
from src.topology import Topology
from tests.test_solver import make_case


@unittest.skipUnless(getattr(native, 'verify_gather', None), 'C++ gather verifier required')
class GatherCertificateTests(unittest.TestCase):
    def setUp(self):
        self.task, data = make_case(16)
        edges = [{'src': a, 'dst': b} for rank in data['ranks']
                 for a, b in [(rank, 'S0'), ('S0', rank)]]
        data['resources'] = [{'id': 'pooled', 'capacity': '2GB/s', 'edges': edges},
                             {'id': 'second', 'capacity': '4GB/s', 'edges': edges[:5]}]
        self.graph = Topology.from_data(data)
        ranks = self.graph.ranks
        space = BalancedStageSpace(self.task, self.graph, 2)
        sketch = GroupBroadcast(space, [ranks[i:i + 4] for i in range(0, 16, 4)])
        self.predicted = sketch.cost
        self.answer = _initial(self.task, ranks, 2)
        self.answer['rounds'] = [dict(round=i, transfers=row)
                                 for i, row in enumerate(sketch.compile(), 1)]
        self.answer['round_count'] = len(self.answer['rounds'])

    def test_group_spread_and_shared_pool_occurrences(self):
        for check in (replay, certify):
            result = check(self.task, self.graph, self.answer)
            self.assertTrue(result['valid'], result)
            self.assertAlmostEqual(result['total_time_us'], self.predicted)
        expected = replay(self.task, self.graph, self.answer)['round_times_us']
        measured = certify(self.task, self.graph, self.answer)['round_times_us']
        for x, y in zip(expected, measured, strict=True):
            self.assertAlmostEqual(x, y)

    def test_corrupt_emitted_schedule(self):
        for fault in ('duplicate', 'same_round', 'path', 'action', 'round', 'final', 'incomplete'):
            with self.subTest(fault=fault):
                answer = deepcopy(self.answer)
                first_round = answer['rounds'][0]
                move = first_round['transfers'][0]
                if fault == 'duplicate':
                    first_round['transfers'].append(deepcopy(move))
                elif fault == 'same_round':
                    rank = next(r for r in self.graph.ranks if r not in (move['src'], move['dst']))
                    first_round['transfers'].append(dict(action='copy', src=move['dst'], dst=rank,
                        path=[move['dst'], 'S0', rank], chunk_ranges=move['chunk_ranges'][:1]))
                elif fault == 'path':
                    move['path'] = [move['src'], move['dst']]
                elif fault == 'action':
                    move['action'] = 'reduce'
                elif fault == 'round':
                    first_round['round'] = True
                elif fault == 'final':
                    answer['final_chunks'].pop()
                elif fault == 'incomplete':
                    answer['rounds'][-1]['transfers'].pop()
                self.assertFalse(replay(self.task, self.graph, answer)['valid'])
                self.assertFalse(certify(self.task, self.graph, answer)['valid'])
