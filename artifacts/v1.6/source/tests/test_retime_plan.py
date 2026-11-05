"""Programs retain their declared meaning across rank order and route changes."""
from copy import deepcopy
import itertools
import json
import unittest
from unittest.mock import patch

from src.collectives import build
from src.config import Settings
from src.llm import edit_retiming
from src.replay import replay
from src.retime_plan import prepare, execute
from src.forward_edit import Context, Rewrite, compile_rewrites, copy_states
from src.retime_plan import _moves
from tests.test_forward_edit import fixture, receptions
from tests.test_route_balance import parallel_case


class RetimingTests(unittest.TestCase):
    def test_capacity_preview_matches_independent_replay_of_each_edit(self):
        for operator in ('all-gather', 'all-reduce'):
            task, graph = parallel_case(operator)
            kind = 'reverse_doubling' if operator == 'all-gather' else 'xor_tree'
            base = build(task, graph, 4, kind=kind, resource_weight=0)
            report = replay(task, graph, base)
            problem, bound, _ = prepare(task, graph, base, report)
            states, bit = copy_states(task, graph, base)
            for ri, item in zip(bound.rounds, problem['edits']):
                for amount, key in ((1, 'half'), (2, 'all')):
                    ctx = Context(bound.input_hash, bound.seed_hash,
                                  (Rewrite('advance', _moves(base, graph, states, bit, ri, amount)),),
                                  report['total_time_us'])
                    candidate = compile_rewrites(base, ctx, [0])
                    checked = replay(task, graph, candidate)
                    self.assertTrue(checked['valid'])
                    times = iter(checked['round_times_us'])
                    expanded = [next(times) if row['transfers'] else 0 for row in candidate['rounds']]
                    for predicted, actual in zip(item[key][:2], expanded[ri-1:ri+1]):
                        self.assertEqual(predicted, float(f'{actual:.4g}'))
                    expected_gain = report['total_time_us'] - checked['total_time_us']
                    self.assertAlmostEqual(item[key][2], expected_gain,
                                           delta=max(1e-8, abs(expected_gain) * 0.0005))

    def test_cost_screen_handles_declared_empty_rounds(self):
        task, graph, base = fixture()
        base['rounds'].insert(0, {'round':1, 'transfers':[]})
        for i, row in enumerate(base['rounds']):
            row['round'] = i + 1
        base['round_count'] = len(base['rounds'])
        report = replay(task, graph, base)
        self.assertTrue(report['valid'])
        _, ctx, _ = prepare(task, graph, base, report)
        result, ev = execute(task, graph, base, report, ctx, {'counts':[2]*len(ctx.rounds)})
        self.assertTrue(replay(task, graph, result)['valid'])
        self.assertLessEqual(ev['final_time_us'], report['total_time_us'])

    def test_actual_program_changes_rounds_and_improves_valid_schedule(self):
        task, graph, base = fixture()
        report = replay(task, graph, base)
        problem, context, _ = prepare(task, graph, base, report)
        candidate, ev = execute(task, graph, base, report, context, {'counts':[2]})
        self.assertTrue(ev['accepted'], ev)
        self.assertGreater(ev['advanced_atoms'], 0)
        self.assertEqual(ev['compiled_counts'], [2])
        self.assertEqual(receptions(base), receptions(candidate))
        self.assertTrue(replay(task, graph, candidate)['valid'])
        self.assertEqual(len(problem['edits']), 1)

    def test_all_ternary_programs_on_both_operators_are_nonregressing(self):
        for op in ('all-gather', 'all-reduce'):
            task, graph = parallel_case(op)
            kind = 'reverse_doubling' if op == 'all-gather' else 'xor_tree'
            seed = build(task, graph, 4, kind=kind, resource_weight=0)
            saved = deepcopy(seed)
            report = replay(task, graph, seed)
            _, ctx, _ = prepare(task, graph, seed, report)
            for counts in itertools.product((0,1,2), repeat=len(ctx.rounds)):
                result, ev = execute(task, graph, seed, report, ctx, {'counts':list(counts)})
                checked = replay(task, graph, result)
                self.assertTrue(checked['valid'], checked['errors'])
                self.assertLessEqual(checked['total_time_us'], report['total_time_us'] + 1e-6)
                self.assertEqual(receptions(seed), receptions(result))
            self.assertEqual(seed, saved)

    def test_recompilation_uses_current_packet_contents(self):
        task, graph, seed = fixture()
        initial = replay(task, graph, seed)
        _, ctx, _ = prepare(task, graph, seed, initial)
        control = build(task, graph, 1, order=list(reversed(graph.ranks)))
        report = replay(task, graph, control)
        result, ev = execute(task, graph, control, report, ctx, {'counts':[2]})
        self.assertNotEqual(ev['reason'] if 'reason' in ev else '', 'rejected')
        self.assertEqual(receptions(control), receptions(result))
        self.assertTrue(replay(task, graph, result)['valid'])
        self.assertEqual(ev['compiled_counts'], [2])

    def test_malformed_or_different_instance_never_gets_silently_repaired(self):
        task, graph, base = fixture()
        report = replay(task, graph, base)
        _, ctx, _ = prepare(task, graph, base, report)
        for value in (None, {}, {'counts':[]}, {'counts':[True]}, {'counts':[3]},
                      {'counts':[1.0]}, {'counts':[1,2]}, {'counts':[2], 'apply':[0]}):
            result, ev = execute(task, graph, base, report, ctx, value)
            self.assertIs(result, base)
            self.assertTrue(ev['errors'])
            self.assertFalse(ev['accepted'])
        task['max_nonempty_rounds'] += 1
        self.assertTrue(execute(task, graph, base, report, ctx, {'counts':[2]})[1]['errors'])

    def test_capacity_change_changes_real_prompt(self):
        task, graph, base = fixture()
        problem, ctx, _ = prepare(task, graph, base, replay(task, graph, base))
        graph.edges['r0','r1'] *= 2
        changed, bound, _ = prepare(task, graph, base, replay(task, graph, base))
        self.assertNotEqual(problem, changed)
        self.assertNotEqual(ctx.input_hash, bound.input_hash)

    def test_short_reply_is_recorded_without_substituting_counts(self):
        settings = Settings('http://example.test/v1/chat/completions', 'test-key', 'test-model')
        raw = '{"counts":[1,0,2]}'
        with patch('src.llm._chat_completion', return_value=(raw, {'finish_reason':'stop'})) as call:
            proposal, meta = edit_retiming({}, {'edits':[]}, settings=settings)
        self.assertEqual(proposal, {'counts':[1,0,2]})
        self.assertEqual(meta['response_evidence']['content'], raw)
        self.assertEqual(call.call_args.kwargs['temperature'], 0)
        self.assertLessEqual(call.call_args.args[2].max_tokens, 64)
        for text, finish in [('{"counts":[NaN]}', 'stop'), ('{"counts":[],"counts":[2]}','stop'),
                             ('{"counts":[2]}','length'), ('test-key', 'stop')]:
            with patch('src.llm._chat_completion', return_value=(text, {'finish_reason':finish})):
                proposal, meta = edit_retiming({}, {'edits':[]}, settings=settings)
            self.assertIsNone(proposal)
            self.assertNotIn('test-key', json.dumps(meta))
