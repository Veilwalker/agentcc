"""Check physical tours, launch pricing and the speculative selection boundary."""
import unittest
from types import SimpleNamespace

from src import domain_cycles, domain_reduce, domain_stripes, flow_search
from src.capacity_backend import native


@unittest.skipUnless(native is not None, 'native scheduling kernels required')
class DomainScheduleTests(unittest.TestCase):
    def test_cycles_visit_each_domain_and_reconcile_shared_pool(self):
        domains, width = 5, 3
        choices = []
        for a in range(domains):
            for b in range(domains):
                if a == b:
                    choices.append([])
                else:
                    choices.append([(((0, 1.),), 1., 1., (a,b)),
                                    (((1, 2.),), 1., 2., (a,b))])
        peak, lanes = native.domain_cycles(domains,width,2,4.,choices)
        loads = [0.,0.]
        for tour in lanes:
            self.assertEqual({source for source,_,_ in tour}, set(range(domains)))
            for i, (source,dest,option) in enumerate(tour):
                self.assertEqual(dest,tour[(i+1) % domains][0])
                self.assertNotEqual(source,dest)
                loads[option] += (1.,2.)[option]*4.
        self.assertEqual(peak,max(loads))

    def test_paced_launch_matches_independently_accumulated_work(self):
        routes=[(((0,1.),(1,2.)),1.,1.,('a','x','b')),(((1,3.),),2.,2.,('b','y','c'))]
        offsets,times=native.paced_launch(2,2,2,3,1.,routes,[(0,0,(0,1)),(1,1,(0,1))],2)
        self.assertEqual(len(offsets),4)
        demand=[[0.,0.] for _ in times]
        latency=[0.]*len(times)
        serial=[0.]*len(times)
        for start in offsets:
            for age,(terms,delay,inverse,_) in enumerate(routes):
                moment=start+age
                for resource,rate in terms:
                    demand[moment][resource]+=rate
                latency[moment]=max(latency[moment],delay)
                serial[moment]=max(serial[moment],delay+inverse)
        self.assertEqual(times,[max(s,l+max(row)) for s,l,row in zip(serial,latency,demand)])

    def test_pacing_rejects_excessive_replication_and_depth(self):
        for depth,copies in [(17,1),(3,9),(2,0)]:
            with self.subTest(depth=depth,copies=copies):
                with self.assertRaises(ValueError):
                    native.paced_launch(depth,1,2,2,1.,[],[],copies)

    def test_missing_reverse_port_or_insufficient_rounds_preserves_incumbent(self):
        leaves=['s9','s3','s7','s1']
        groups=[[f'r{2*k}',f'r{2*k+1}'] for k in range(4)]
        ranks=sum(groups,[])
        adj={r:[(leaf,1,0)] for leaf,group in zip(leaves,groups) for r in group}
        edges={(leaf,r):1. for leaf,group in zip(leaves,groups) for r in group}
        graph=SimpleNamespace(ranks=ranks,switches=set(leaves),adjacency=adj,edges=edges)
        parent,report={},dict(total_time_us=10000.)
        for operator,build in [('all-gather',domain_cycles.improve),('all-reduce',domain_reduce.improve)]:
            plan,checked,_=build(dict(operator=operator,max_nonempty_rounds=2),graph,parent,report,None,[1,2])
            self.assertIs(plan,parent)
            self.assertIs(checked,report)
        edges.pop((leaves[0],ranks[0]))
        self.assertIsNone(domain_cycles._layout(graph))

    def test_faster_prepared_action_wins_and_attribution_tracks_it(self):
        state=flow_search.FlowSearch({},None,None)
        state.slack_data=object()
        def candidate(cost):
            return {'id':cost},{'total_time_us':cost},{'accepted':True,'rounds':[]}
        state.slack_options={False:candidate(80.),True:candidate(70.)}
        probes=[dict(key='uplink',row=['P0',50,20,0])]
        _,checked,audit=state.apply_focus({}, {}, {}, probes, [dict(focus_id=0,repair_policy='E')])
        self.assertEqual(checked['total_time_us'],70.)
        self.assertEqual(audit['round_slack']['applied_policy'],'R')
        self.assertFalse(audit['round_slack']['model_preferred_winner'])

    def test_shared_port_between_distinct_dual_port_domains_is_not_a_private_domain(self):
        ranks=['alpha','beta']
        graph=SimpleNamespace(ranks=ranks,switches={'x','y','z'},
            adjacency={'alpha':[('x',1,0),('y',1,0)],'beta':[('y',1,0),('z',1,0)]},
            edges={(p,r):1. for p in ('x','y','z') for r in ranks})
        self.assertIsNone(domain_stripes.layout(graph))

    def test_deferred_tree_competes_after_model_repair(self):
        state=flow_search.FlowSearch({},None,None)
        state.deferred_tree=({'id':'tree'},{'total_time_us':70.})
        selected,report,audit=state.apply_focus({'id':'old'},{'total_time_us':80.},{},[],[])
        self.assertEqual(selected['id'],'tree')
        self.assertEqual(audit['final_us'],report['total_time_us'])
        self.assertIsNone(state.deferred_tree)
