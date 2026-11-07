"""Communication semantics, capacity accounting and model-program acceptance."""
import copy
import unittest
from unittest.mock import patch

from src.fabric_routes import RouteCatalog, CapacityCosts, CapacityAssignment
from src.stage_program import StageSpace, validate_widths
from src.stage_plan import prepare, execute, compile_bands
from src.collectives import build
from src.replay import replay
from src.route_balance import _PathCosts, _RoundLoad
from src.topology import Topology
from src.config import Settings
from src.llm import edit_stages, _parse_edit_json
from src.synthesis_service import solve_input
from tests.test_solver import make_case


def partitions(n):
    if not n:
        yield []
    for width in range(1, min(n, 3) + 1):
        for tail in partitions(n - width):
            yield [width, *tail]


class StageProgramTests(unittest.TestCase):
    def test_route_cache_shares_switch_cores_without_sharing_endpoint_capacities(self):
        topology = {'ranks': ['a', 'b', 'c'], 'switches': ['s', 't'],
            'edges': [{'src': a, 'dst': b, 'bw': capacity} for a, b, capacity in
                      [('a', 's', 1e9), ('c', 's', 2e9), ('s', 't', 3e9), ('t', 'b', 4e9)]]}
        catalog = RouteCatalog(Topology.from_data(topology), 100)
        first = catalog.options('a', 'b')[0]
        second = catalog.options('c', 'b')[0]
        self.assertEqual(len(catalog.fabric_paths), 1)
        self.assertEqual(first, ('a', 's', 't', 'b'))
        self.assertEqual(second, ('c', 's', 't', 'b'))
        costs = catalog.capacity_costs
        self.assertGreater(costs.compile(first)[2], costs.compile(second)[2])
        self.assertEqual(catalog.options('b', 'a'), ())

    def test_stage_transport_parses_exact_array_while_legacy_edits_reject_it(self):
        problem = {'bands': [{'keep': 20, 'all': 10}], 'op': 'AG', 'task': [4, 100, 1, 8], 'e': [], 'p': []}
        settings = Settings('http://unused', '', 'test')
        for content in ('["all"]', '{"bands":["all"]}', '```json\n["all"]\n```'):
            with patch('src.llm._chat_completion', return_value=(content, {'finish_reason': 'stop'})):
                proposal, metadata = edit_stages({}, problem, settings=settings)
            self.assertTrue(metadata['proposal_json_valid'])
            self.assertIn(proposal, (['all'], {'bands': ['all']}))
        with self.assertRaises(ValueError):
            _parse_edit_json('["all"]')
        for content in ('[NaN]', '{"bands":[],"bands":[]}', '["all"] trailing text'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                _parse_edit_json(content, allow_array=True)

    def test_every_small_partition_matches_replay_for_both_collectives(self):
        for n in (2, 4, 8, 16):
            for operator in ('all-gather', 'all-reduce'):
                task, topology = make_case(n, operator)
                graph = Topology.from_data(topology)
                for split in (1, 4):
                    space = StageSpace(task, graph, split)
                    for widths in partitions(space.dimensions):
                        with self.subTest(n=n, operator=operator, split=split, widths=widths):
                            answer, expected = space.compile(widths)
                            report = replay(task, graph, answer)
                            self.assertTrue(report['valid'], report['errors'])
                            self.assertAlmostEqual(expected, report['total_time_us'], places=7)

    def test_invalid_stage_instructions_are_not_repaired(self):
        for widths in ([True, 2], [0, 3], [4], [1], None, '3', [1.0, 2]):
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                validate_widths(widths, 3)

    def test_named_edits_and_singleton_alias_preserve_exact_program(self):
        task, topology = make_case(8)
        space = StageSpace(task, Topology.from_data(topology), 1)
        bands = (((1, 1, 1), (2, 1), (1, 2), (3,)),)
        for code, name, expected in ((1, 'left', [2, 1]), (2, 'right', [1, 2]), (3, 'all', [3])):
            for instruction in ([code], [[code]], [name]):
                self.assertEqual(compile_bands(space, bands, instruction), expected)
        self.assertEqual(compile_bands(space, bands, ['keep']), [])
        for instruction in (['unknown'], [True], [[1, 2]], [[True]], [4], [], [1.0]):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                compile_bands(space, bands, instruction)

    def test_preview_does_not_allocate_atom_expressions_or_materialized_rounds(self):
        task, topology = make_case(16, 'all-reduce')
        space = StageSpace(task, Topology.from_data(topology), 4)
        plans, copies, predicted = space.transition(0, 3)
        self.assertFalse(space.materialized)
        self.assertTrue(all(type(row[4]) is int for row in plans + copies))
        solution, cost = space.compile([3, 1])
        report = replay(task, topology, solution)
        self.assertTrue(report['valid'], report['errors'])
        self.assertAlmostEqual(report['total_time_us'], cost)
        self.assertGreaterEqual(cost, predicted)

    def test_preview_and_packet_coalescing_count_repeated_overlapping_pools(self):
        for operator in ('all-gather', 'all-reduce'):
            task, topology = make_case(16, operator)
            topology['resources'] = [
                {'id': 'both', 'capacity': 2e8, 'edges': topology['edges']},
                {'id': 'overlap', 'capacity': 1e8, 'edges': topology['edges'][::2]}]
            graph = Topology.from_data(topology)
            space = StageSpace(task, graph, 4)
            for widths in partitions(4):
                solution, expected = space.compile(widths)
                checked = replay(task, graph, solution)
                self.assertTrue(checked['valid'], checked['errors'])
                self.assertAlmostEqual(expected, checked['total_time_us'])

    def test_order_must_be_exact_rank_permutation_and_round_bound_is_enforced(self):
        task, topology = make_case(8, 'all-reduce')
        graph = Topology.from_data(topology)
        with self.assertRaises(ValueError):
            StageSpace(task, graph, 1, order=list(graph.ranks) * 2)
        task['max_nonempty_rounds'] = 2
        space = StageSpace(task, graph, 1)
        with self.assertRaises(ValueError):
            space.compile([1, 1, 1])
        answer, _ = space.compile([3])
        self.assertTrue(replay(task, graph, answer)['valid'])

    def test_catalog_discovers_multihop_alternatives_without_rank_relays(self):
        topology = {'ranks': ['a', 'b', 'c'], 'switches': ['s', 'u', 't', 'v'],
            'edges': [{'src': a, 'dst': b, 'bw': 1e9} for a, b in
                      [('a', 's'), ('s', 't'), ('t', 'b'), ('a', 'u'), ('u', 'v'),
                       ('v', 'b'), ('s', 'c'), ('c', 'v')]]}
        graph = Topology.from_data(topology)
        choices = RouteCatalog(graph, 100).options('a', 'b')
        self.assertEqual(set(choices), {('a', 's', 't', 'b'), ('a', 'u', 'v', 'b')})
        self.assertEqual(RouteCatalog(graph, 100).options('b', 'a'), ())

    def test_secondary_switch_branches_are_actual_directed_edges(self):
        topology = {'ranks': ['a', 'b'], 'switches': ['s', 'x', 'y', 't'],
            'edges': [{'src': a, 'dst': b, 'bw': 1e9} for a, b in
                      [('a', 's'), ('s', 'x'), ('s', 'y'), ('x', 't'), ('y', 't'), ('t', 'b')]]}
        graph = Topology.from_data(topology)
        choices = RouteCatalog(graph, 100, branch_ports=True).options('a', 'b')
        self.assertEqual(set(choices), {('a', 's', 'x', 't', 'b'), ('a', 's', 'y', 't', 'b')})
        self.assertEqual(RouteCatalog(graph, 100, branch_ports=True).options('b', 'a'), ())

    def test_structural_budget_exhaustion_keeps_live_retiming_context(self):
        task, topology = make_case(8)
        graph = Topology.from_data(topology)
        seed = build(task, graph, order=list(graph.ranks))
        report = replay(task, graph, seed)
        with patch.object(RouteCatalog, 'options', side_effect=ValueError('budget exhausted')):
            problem, context, evidence = prepare(task, graph, seed, report)
        self.assertEqual(context.mode, 'retime')
        self.assertIn('edits', problem)
        self.assertIsNone(context.space)
        self.assertIn('budget', evidence['stage_unavailable_reason'])

    def test_capacity_incidence_counts_overlapping_pools_and_repeated_use(self):
        task, topology = make_case(4)
        topology['resources'] = [
            {'id': 'both', 'capacity': 2e8, 'edges': [{'src': 'r0', 'dst': 'S0'}, {'src': 'S0', 'dst': 'r1'}]},
            {'id': 'ingress', 'capacity': 4e8, 'edges': [{'src': 'r0', 'dst': 'S0'}]}]
        graph = Topology.from_data(topology)
        costs = CapacityCosts(graph)
        load, reference = CapacityAssignment(costs), _RoundLoad(_PathCosts(graph))
        for path, count in [(('r0', 'S0', 'r1'), 4000), (('r2', 'S0', 'r1'), 8000)]:
            selected = load.choose([costs.compile(path)], count)
            reference.add(path, count)
            self.assertEqual(selected, path)
            self.assertAlmostEqual(load.time_us, reference.time_us)

    def test_compiled_packet_bound_is_real_packet_size(self):
        task, topology = make_case(4)
        graph = Topology.from_data(topology)
        costs = CapacityCosts(graph)
        load, reference = CapacityAssignment(costs), _RoundLoad(_PathCosts(graph))
        path = ('r0', 'S0', 'r1')
        load.choose([costs.compile(path)], 4000, 1000)
        for _ in range(4):
            reference.add(path, 1000)
        self.assertAlmostEqual(load.time_us, reference.time_us)

    def test_binding_and_malformed_program_preserve_control(self):
        task, topology = make_case(8)
        graph = Topology.from_data(topology)
        seed = build(task, graph, order=list(graph.ranks))
        report = replay(task, graph, seed)
        problem, context, _ = prepare(task, graph, seed, report)
        zero = [0] * len(context.retiming.rounds)
        before = copy.deepcopy(seed)
        for proposal in ({'counts': zero}, {'counts': zero, 'bands': [True, 2]},
                         {'counts': zero, 'bands': [0], 'extra': 1}):
            candidate, evidence = execute(task, graph, seed, report, context, proposal)
            self.assertEqual(candidate, before)
            self.assertFalse(evidence['selected'])
            self.assertTrue(evidence['errors'])
        changed = copy.deepcopy(task)
        changed['data_size']['per_rank_bytes'] *= 2
        candidate, evidence = execute(changed, graph, seed, report, context, {'counts': zero, 'bands': [0]})
        self.assertEqual(candidate, before)
        self.assertTrue(evidence['errors'])

    def test_service_calls_model_once_and_accepts_only_actual_program(self):
        task, topology = make_case(8)
        seen = []
        def model(task, problem, **kwargs):
            seen.append(problem)
            return ({'bands': [len(row) - 1 for row in problem['bands']]} if 'bands' in problem
                    else {'counts': [0] * len(problem['edits'])}), {'called': True}
        with patch('src.synthesis_service.load_settings', return_value=Settings('http://invalid', 'test', 'test')), \
             patch('src.synthesis_service.edit_stages', side_effect=model):
            result = solve_input({'task': task, 'topology': topology})
        self.assertEqual(len(seen), 1)
        checked = replay(task, topology, result['answer'])
        self.assertTrue(checked['valid'], checked['errors'])
        self.assertLessEqual(checked['total_time_us'], result['diagnostics']['search']['model_edit']['control_time_us'] + 1e-6)


if __name__ == '__main__':
    unittest.main()
