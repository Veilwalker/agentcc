"""Independent capacity arithmetic, holder ownership and generated recurrence tests."""
import math
import random
import unittest

from src.capacity_backend import native
from src.code_policy import assemble_frontier
from src.fabric_routes import PythonCapacityAssignment
from src.generated_service import execute_program
from src.program_vm import Program, ProgramError
from src.refined_search import RefinedSearchSpace
from src.stage_program import StageSpace
from src.replay import replay
from src.topology import Topology
from tests.test_generated_solver import SINGLE_REFERENCE
from tests.test_solver import make_case

# A test-only recurrence; production receives this expression from the model.
EXPRESSION = 'min((v+t,p+[w]) for v,p,w,t in a)'


def duration(packets, selections, resources):
    loads, overhead, serial = [0.0]*resources, 0.0, 0.0
    for (options, size, packet), index in zip(packets, selections):
        terms, latency, inverse, _ = options[index]
        for resource, factor in terms:
            loads[resource] += size*factor
        overhead = max(overhead, latency)
        serial = max(serial, latency + packet*inverse)
    return max(serial, overhead+max(loads, default=0.0))


@unittest.skipIf(native is None or not hasattr(native, 'optimize_round'), 'native optimizer required')
class RoundOptimizationTests(unittest.TestCase):
    def test_independent_phases_preserve_exact_contributors(self):
        task, data = make_case(16, 'all-reduce')
        space = StageSpace(task, Topology.from_data(data), 4, max_width=4)
        for first in range(4):
            for width in range(1, 5 - first):
                for atoms in (False, True):
                    both = space._movements(first, width, atoms=atoms)
                    for phase in (0, 1):
                        one = space._movements(first, width, atoms=atoms, phase=phase)
                        self.assertEqual(one[phase], both[phase])
                        self.assertFalse(one[1 - phase])

    def test_shared_route_options_preserve_packet_sizes_and_overflow_checks(self):
        options = ((((0, .5),), 1., .5, ('a', 'b')),
                   (((1, .25),), 2., .25, ('a', 'c', 'b')))
        packets = [(options, size, 1.) for size in (8., 16., 32., 64.)]
        selected, _, actual, _ = native.optimize_round(2, packets, 2)
        self.assertAlmostEqual(actual, duration(packets, selected, 2))
        # The second packet reuses already parsed options. It must still be
        # checked with its own size, including numerical overflow.
        huge = ((((0, 1e300),), 0., 1., ('a', 'b')),)
        with self.assertRaises(ValueError):
            native.optimize_round(1, [(huge, 1., 1.), (huge, 1e300, 1.)], 2)

    def test_random_rounds_match_reference_and_never_regress(self):
        rng = random.Random(19010)
        costs = type('Costs', (), {'indices': list(range(25))})()
        for _ in range(80):
            packets = []
            greedy = PythonCapacityAssignment(costs)
            for row in range(rng.randint(1, 50)):
                options = tuple((tuple((i, rng.choice([.1,.25,1.,2.]))
                                      for i in rng.sample(range(25), rng.randint(1,5))),
                                 rng.choice([0.,1.,.3]), rng.choice([.1,.2,.4]), (row,j))
                                for j in range(rng.randint(1,8)))
                size, packet = rng.randint(1,1000), rng.randint(1,10)
                packets.append((options,size,packet))
                greedy.choose(options,size,packet)
            selected,before,after,moved = native.optimize_round(25,packets,2)
            self.assertAlmostEqual(before,greedy.time_us,places=8)
            self.assertAlmostEqual(after,duration(packets,selected,25),places=8)
            self.assertLessEqual(after,before+1e-8)
            self.assertEqual((selected,before,after,moved),native.optimize_round(25,packets,2))

    def test_rejects_invalid_inputs_without_leaking_state(self):
        options = ((((0,1.),),0.,1.,('a','b')),)
        good = [(options,10.,1.)]
        expected = native.optimize_round(1,good,2)
        invalid = [([],1.,1.), (options,-1.,1.), (options,math.nan,1.),
                   (((((3,1.),),0.,1.,('a','b')),),10.,1.),
                   (((((0,1.),(0,2.)),0.,1.,('a','b')),),10.,1.)]
        for row in invalid:
            with self.assertRaises((TypeError,ValueError)):
                native.optimize_round(1,[row],2)
            self.assertEqual(expected,native.optimize_round(1,good,2))

    def test_current_holder_and_independent_reduce_copy_are_replayable(self):
        for op in ['all-gather','all-reduce']:
            task,data=make_case(16,op)
            data['resources']=[{'id':'shared','capacity':1e8,'edges':data['edges'][::3]}]
            for i,e in enumerate(data['edges']):
                e['bw']=(i%3+1)*1e9
            graph=Topology.from_data(data)
            space=RefinedSearchSpace(task,graph,4)
            result=execute_program(SINGLE_REFERENCE,space)
            self.assertTrue(result['valid_program'],result)
            answer,predicted=space.compile(result['output'])
            report=replay(task,graph,answer)
            self.assertTrue(report['valid'],report['errors'])
            self.assertAlmostEqual(predicted,report['total_time_us'],places=6)
            self.assertAlmostEqual(predicted,space.audit_bound(),places=6)


class FrontierProgramTests(unittest.TestCase):
    def test_model_expression_reads_actual_costs_and_handles_wide_transitions(self):
        source=assemble_frontier(EXPRESSION)
        self.assertEqual(Program(source).run([[],[3],[2,1]],2),[2])
        self.assertEqual(Program(source).run([[],[3],[2,9]],2),[1,1])
        table=[[]]+[[9.]*min(5,i) for i in range(1,7)]
        table[5][-1]=1.
        self.assertEqual(Program(source).run(table,6),[5,1])

    def test_formatting_does_not_change_arithmetic_or_accept_fixed_answers(self):
        self.assertEqual(assemble_frontier('```python\n'+EXPRESSION+'\n```'),assemble_frontier(EXPRESSION))
        self.assertEqual(assemble_frontier('min((v+t,p+[w])\n for v,p,w,t in a)'),assemble_frontier(EXPRESSION))
        self.assertEqual(assemble_frontier('[2,3,3]'),'')
        self.assertEqual(assemble_frontier('d[i]=min(a)'), '')
        with self.assertRaises(ProgramError):
            Program(assemble_frontier('open(a)')).run([[],[1]],1)
        source=assemble_frontier('min((v-t,p+[w]) for v,p,w,t in a)')
        self.assertIn('v - t',source)  # Incorrect arithmetic is not silently repaired.
