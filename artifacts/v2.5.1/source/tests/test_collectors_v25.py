"""Check the novel forest construction and its route-to-packet conversion."""
import unittest
from types import SimpleNamespace

from src.bridge_forest import LocalCover
from src.reduction_parents import emit_batches


class ReductionPacketsTests(unittest.TestCase):
    def test_mixed_actions_share_the_path_without_mixing_their_tokens(self):
        path = ('rank0', 'leaf', 'rank1')
        costs = SimpleNamespace(paths={path: ((), 2.0, 0.25, path)})
        batches = {(path, False): [0, 2, 4], (path, True): [1, 3]}
        # Four bytes per atom, two microseconds latency, two atoms of room.
        transfers = emit_batches(batches, 4.0, 4, costs)
        self.assertEqual([(m['action'], m['chunk_ranges']) for m in transfers], [
            ('reduce', ['rank0_c0', 'rank0_c2']),
            ('reduce', ['rank0_c4']),
            ('copy', ['c1_sum', 'c3_sum'])])
        self.assertTrue(all(m['path'] == list(path) for m in transfers))


class LocalCoverTests(unittest.TestCase):
    @staticmethod
    def cover(ports):
        edges = {}
        for rank, switches in ports.items():
            for switch in switches:
                edges[rank, switch] = edges[switch, rank] = 1
        forest = SimpleNamespace(ports=ports, graph=SimpleNamespace(edges=edges))
        return LocalCover(forest, list(ports))

    def test_bridges_connect_a_rank_without_a_port_shared_with_the_root(self):
        cover = self.cover({'root': ['a'], 'relay': ['a', 'b'], 'leaf': ['b'], 'tail': ['b']})
        root, bridges, parents = cover.representatives(0)
        self.assertEqual(root, 'root')
        self.assertEqual(bridges, ['relay'])
        self.assertEqual(parents, {'leaf': ('relay',), 'tail': ('relay',)})

    def test_unreachable_domain_falls_back(self):
        cover = self.cover({'root': ['a'], 'relay': ['a'], 'isolated': ['b']})
        self.assertIsNone(cover.representatives(0))
