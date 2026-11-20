import copy
import unittest

from src.capacity_backend import native
from src.code_policy import layout_response, relay_expression
from src.collectives import build
from src.replay import certify, replay
from src.topology import Topology
from tests.test_solver import make_case


class RelayPrograms(unittest.TestCase):
    def test_layout_is_executed_on_slot_and_domain_coordinates(self):
        spacing, text = layout_response('(3, (i + 3*g) % n)')
        self.assertEqual(spacing, 3)
        program = relay_expression(text)
        self.assertEqual([program((4, g, 8)) for g in range(3)], [4, 7, 2])

    def test_rejects_unbounded_or_non_placement_programs(self):
        for source in ('(True, i)', '(5, 0)', '(2, i ** n)', '(3, f(i))', '(20, i % n)'):
            self.assertIsNone(layout_response(source))
        with self.assertRaises(ZeroDivisionError):
            relay_expression('i // g')((1, 0, 8))


@unittest.skipUnless(hasattr(native, 'verify_reduce'), 'requires compiled reduction replay')
class ReductionCertificates(unittest.TestCase):
    def setUp(self):
        self.task, data = make_case(16, 'all-reduce')
        self.graph = Topology.from_data(data)
        self.answer = build(self.task, self.graph, 4, kind='xor_tree')

    def check(self, answer, expected):
        reference = replay(self.task, self.graph, answer)
        actual = certify(self.task, self.graph, answer)
        self.assertEqual(actual['valid'], expected, actual)
        self.assertEqual(reference['valid'], expected)
        if expected:
            for left, right in zip(reference['round_times_us'], actual['round_times_us'], strict=True):
                self.assertAlmostEqual(left, right, places=7)

    def test_independent_cost_and_state(self):
        self.check(self.answer, True)

    def test_broken_rounds_and_contributions(self):
        for error in ('same-round', 'duplicate', 'missing', 'partial-copy', 'unknown', 'repeated-slot'):
            with self.subTest(error=error):
                a = copy.deepcopy(self.answer)
                first = a['rounds'][0]['transfers']
                if error == 'duplicate':
                    first.append(copy.deepcopy(first[0]))
                elif error == 'same-round':
                    first.extend(a['rounds'][1]['transfers'])
                    a['rounds'][1]['transfers'] = []
                elif error == 'missing':
                    a['rounds'][-1]['transfers'].pop()
                elif error == 'partial-copy':
                    first[0]['action'] = 'copy'
                elif error == 'unknown':
                    first[0]['chunk_ranges'][0] = 'unknown_c0'
                else:
                    first[0]['chunk_ranges'] *= 2
                self.check(a, False)
