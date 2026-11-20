"""Regression tests for ready ownership, shared pools and alternate senders."""
import unittest

from src.capacity_backend import native


@unittest.skipUnless(native is not None and hasattr(native, 'fill_slack'), 'requires native slack kernel')
class RoundSlackTests(unittest.TestCase):
    def run_plan(self, paths, moves, owners, ceilings):
        resources = max(k for terms in paths for k, _ in terms) + 1
        routes = [(tuple(terms), 0., 1., ()) for terms in paths]
        return native.fill_slack(4,len(owners),resources,1.,-1,routes,moves,owners,ceilings,False)

    def test_free_path_can_fill_a_round_without_extending_it(self):
        dests,_,rows = self.run_plan([[(0,1.)],[(1,1.)]],
            [(0,0,3,1,True,(1,)),(1,1,0,2,True,(0,))], [0,3], [1.,1.])
        self.assertEqual(list(dests), [0,0])
        self.assertEqual([r[0] for r in rows], [1.,0.])

    def test_same_round_arrival_is_not_available_to_send(self):
        dests,_,_ = self.run_plan([[(0,1.)],[(1,1.)]],
            [(0,0,0,1,True,(0,)),(1,1,1,2,True,(0,))], [0], [1.,1.])
        self.assertEqual(list(dests), [0,1])

    def test_full_shared_pool_blocks_idle_edge(self):
        dests,_,_ = self.run_plan([[(0,1.),(2,2.)],[(1,1.),(2,1.)]],
            [(0,0,3,1,True,(1,)),(1,1,0,2,True,(0,))], [0,3], [2.,1.])
        self.assertEqual(list(dests), [0,1])

    def test_completed_slot_cannot_cross_its_reduce_barrier(self):
        dests,_,_ = self.run_plan([[(0,1.)],[(1,1.)]],
            [(0,0,0,1,False,(0,)),(1,1,1,2,True,(0,))], [-1], [1.,1.])
        self.assertEqual(list(dests), [0,1])

    def test_existing_alternative_sender_has_the_atom_at_round_start(self):
        dests,paths,_ = self.run_plan([[(0,1.)],[(1,1.)],[(2,1.)]],
            [(0,0,0,1,True,(0,)),(1,1,1,2,True,(0,)),(2,2,0,2,True,(1,))],
            [0,0], [2.,1.,1.])
        self.assertEqual(dests[1],0)
        self.assertEqual(memoryview(paths).cast('i')[1],2)
