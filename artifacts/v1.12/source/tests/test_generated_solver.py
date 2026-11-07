"""Generated solver contracts, bounded execution, and whole-atom compilation."""
import copy
import json
import random
import unittest
from unittest.mock import patch

from src.program_vm import Program, ProgramError
from src.code_policy import assemble_body
from src.stage_search import SearchSpace
from src.generated_service import execute_program, solve_input
from src.topology import Topology
from src.replay import replay
from tests.test_solver import make_case

# Test-only reference, never supplied as a model reply in live runs.
REFERENCE = '''def solve(c, limit):
 b = (1e99, 0, [])
 for p, t in enumerate(c):
  d = {(0, 0): (0, [])}
  for i, r in enumerate(t):
   for k in range(limit):
    if (i, k) in d:
     v, a = d[i, k]
     for w, x in enumerate(r, start=1):
      q = (i+w, k+1)
      d[q] = min(d.get(q, (1e99, [])), (v+x, a+[w]))
  for k in range(1, limit+1):
   if (len(t), k) in d:
    v, a = d[len(t), k]
    b = min(b, (v, p, a))
 return [b[1], b[2]]
'''

# The production interface runs once per layout and includes a terminal row.
SINGLE_REFERENCE = '''def solve(c, limit):
 d = {(0,0): (0,[])}
 for i,r in enumerate(c):
  for k in range(limit):
   if (i,k) in d:
    v,a = d[i,k]
    for w,x in enumerate(r,1):
     q = (i+w,k+1)
     d[q] = min(d.get(q,(1e99,[])),(v+x,a+[w]))
 return min(d.get((len(c)-1,k),(1e99,[])) for k in range(1,limit+1))[1]
'''


FRONTIER_REFERENCE = 'min((v+t,p+[w]) for v,p,w,t in a)'


BODY_REFERENCE = """for i in range(1,len(c)):
 d[i]=min((d[i-w][0]+v,d[i-w][1]+[w]) for w,v in enumerate(c[i],1))"""


def enumerate_paths(table, position=0, limit=30):
    if position == len(table):
        return [(0.0, [])]
    if not limit:
        return []
    return [(cost + rest, [w] + path)
            for w, cost in enumerate(table[position], 1)
            for rest, path in enumerate_paths(table, position+w, limit-1)]


class GeneratedProgramTests(unittest.TestCase):
    def test_unsupported_key_lambda_reports_actionable_feedback(self):
        from src.code_policy import assemble_frontier
        source = assemble_frontier('min(((v+t,p+[w]) for v,p,w,t in a),key=lambda x:x[0])')
        with self.assertRaisesRegex(ProgramError, 'lambda is not allowed.*without key='):
            Program(source)

    def test_vm_matches_exhaustive_paths_with_round_limits(self):
        rng = random.Random(17)
        for _ in range(60):
            n, limit = rng.randrange(2, 7), rng.randrange(2, 7)
            tables = [[[rng.randrange(1, 20) for _ in range(min(3, n-i))]
                       for i in range(n)] for _ in range(2)]
            expected = min((cost, p, path) for p, table in enumerate(tables)
                           for cost, path in enumerate_paths(table, limit=limit))
            program = Program(REFERENCE)
            self.assertEqual(program.run(tables, limit), [expected[1], expected[2]])

    def test_input_is_immutable_and_limits_bound_the_language(self):
        bad = [
            'import os\ndef solve(c,limit): return c',
            'def solve(c,limit): return c.__class__',
            'def solve(c,limit): return open(".env")',
            'def solve(c,limit): return [0] * 9999999999',
            'def solve(c,limit):\n while True: pass\n return c',
            'def solve(c,limit):\n c[0]=0\n return c',
            'def solve(c,limit):\n a=[]\n a.append(a)\n return c',
            'def solve(c,limit):\n a={}\n a[0]=a\n return c',
        ]
        for code in bad:
            with self.subTest(code=code), self.assertRaises(ProgramError):
                Program(code).run([[[1.0]]], 1)

    def test_directed_shared_capacity_independent_phases(self):
        for op in ('all-gather', 'all-reduce'):
            task, data = make_case(16, op)
            data['resources'] = [{'id': 'shared', 'capacity': 1e8, 'edges': data['edges'][::3]}]
            # Make opposite directions asymmetric without changing reachability.
            for i, edge in enumerate(data['edges']):
                edge['bw'] = (1 + i % 3) * 1e9
            graph = Topology.from_data(data)
            space = SearchSpace(task, graph, 4)
            source = Program(REFERENCE)
            output = source.run(space.costs, space.limit)
            candidate, cost = space.compile(output)
            report = replay(task, graph, candidate)
            self.assertTrue(report['valid'], report['errors'])
            self.assertAlmostEqual(cost, report['total_time_us'], places=6)
            self.assertAlmostEqual(cost, space.audit_bound(), places=6)

    def test_no_source_repair_and_input_binding(self):
        task, data = make_case(8)
        graph = Topology.from_data(data)
        space = SearchSpace(task, graph, 1)
        code = 'def solve(c,limit):\n x=len(c)\n return [0,[0,3]]'
        result = execute_program(code, space)
        self.assertFalse(result['valid_program'])
        self.assertEqual(result['source'], code)
        graph.edges[next(iter(graph.edges))] *= 2
        with self.assertRaises(ValueError):
            space.compile([0, [3]])

    def test_actual_cost_change_changes_generated_solver_decision(self):
        p = Program(REFERENCE)
        self.assertEqual(p.run([[[1, 9], [1]]], 2), [0, [1, 1]])
        self.assertEqual(p.run([[[5, 2], [5]]], 2), [0, [2]])
        self.assertEqual(p.run([[[1, 9], [1]]], 1), [0, [2]])

    def test_unavailable_stage_space_records_no_code_execution(self):
        task, data = make_case(8)
        meta = {'called': True, 'usage': {'total_tokens': 1},
                'program_protocol': 'incoming_v1'}
        with patch('src.generated_service.load_settings'), \
             patch('src.generated_service.propose',
                   return_value=(assemble_body(BODY_REFERENCE), meta)) as model, \
             patch('src.generated_service.SearchSpace',
                   side_effect=ValueError('no legal direct stage route')):
            solved = solve_input({'task': task, 'topology': data})
        self.assertTrue(replay(task, data, solved['answer'])['valid'])
        evidence = solved['diagnostics']['search']['model_edit']
        self.assertEqual(model.call_count, 1)
        self.assertEqual(evidence['stop_reason'], 'capacity_space_unavailable')
        self.assertEqual(evidence['executions'], [])
        self.assertFalse(evidence['selected'])
        self.assertIn('unavailable', evidence['space'])

    def test_bad_program_feedback_then_verified_improvement(self):
        task, data = make_case(8, 'all-reduce')
        bad = 'def solve(c,limit):\n x=len(c)\n return [0,[0]]'
        meta = {'called': True, 'usage': {'total_tokens': 1}, 'duration_ms': 1}
        with patch('src.generated_service.load_settings'), patch('src.generated_service.propose',
                side_effect=[(bad,meta),(SINGLE_REFERENCE,meta)]) as model:
            result = solve_input({'task':task,'topology':data})
        self.assertTrue(result['diagnostics']['local_replay']['valid'])
        self.assertEqual(model.call_count, 2)
        execution = result['diagnostics']['search']['model_edit']['executions']
        self.assertFalse(execution[0]['valid_program'])
        self.assertEqual(execution[0]['decision'], 'invalid_program')
        self.assertTrue(execution[1]['valid_program'])
        self.assertIsNotNone(model.call_args.args[2])

    def test_truncated_frontier_keeps_evidence_but_cannot_execute(self):
        from src.code_policy import propose
        from src.config import Settings
        settings = Settings('http://unused.invalid', 'test-credential', 'test-model')
        for finish, expected in [('stop', True), ('length', False)]:
            metadata = {'finish_reason': finish, 'usage': {'total_tokens': 27}}
            with patch('src.code_policy._chat_completion',
                       return_value=(FRONTIER_REFERENCE, metadata)) as model:
                source, evidence = propose({'step_bound_free': True}, settings)
            self.assertEqual(bool(source), expected)
            self.assertEqual(evidence['response_evidence']['content'], FRONTIER_REFERENCE)
            self.assertLessEqual(model.call_args.args[2].max_tokens, 48)
            if source:
                # Different live costs must still change the generated choice.
                self.assertEqual(Program(source).run([[], [1], [1, 9]], 2), [1, 1])
                self.assertEqual(Program(source).run([[], [5], [5, 2]], 2), [2])

    def test_terminal_row_protocol_and_round_limit(self):
        program = Program(SINGLE_REFERENCE)
        self.assertEqual(program.run([[1, 9], [1], []], 2), [1, 1])
        self.assertEqual(program.run([[1, 9], [1], []], 1), [2])
        task, data = make_case(16, 'all-reduce')
        space = SearchSpace(task, Topology.from_data(data), 4)
        result = execute_program(SINGLE_REFERENCE, space)
        self.assertTrue(result['valid_program'], result)
        self.assertAlmostEqual(result['gap_us'], 0.0)
        incoming = execute_program(assemble_body(BODY_REFERENCE), space, 'incoming_v1')
        self.assertTrue(incoming['valid_program'], incoming)
        self.assertAlmostEqual(incoming['predicted_us'], result['predicted_us'])

    def test_model_transport_failure_preserves_valid_floor_without_retry(self):
        from src.llm import LLMError
        task, data = make_case(8)
        with patch('src.generated_service.load_settings'), patch('src.generated_service.propose',
                side_effect=LLMError('test failure')) as model:
            output = solve_input({'task': task, 'topology': data})
        self.assertEqual(model.call_count, 1)
        self.assertTrue(output['diagnostics']['local_replay']['valid'])
        audit = output['diagnostics']['search']['model_edit']
        self.assertFalse(audit['selected'])
        self.assertEqual(audit['stop_reason'], 'model_transport_failed')
        self.assertEqual(audit['executions'], [])

    def test_single_atom_preview_reuses_plain_route(self):
        task, data = make_case(8)
        space = SearchSpace(task, Topology.from_data(data), 1)
        for profile in space.profiles:
            self.assertIs(profile.selected[0, 1, None], profile.plain)
            self.assertNotIn((0, 1), profile.striped.stages)


if __name__ == '__main__':
    unittest.main()
