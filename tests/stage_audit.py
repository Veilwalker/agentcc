"""Offline replay, causal ablation and input perturbations for stage edits.

Saved replies and synthetic replies are labelled separately from real API
measurements. This tool never reads credentials or contacts the model service.
"""
import argparse
from copy import deepcopy
import json
from unittest.mock import patch

from src.config import Settings
from src.forward_edit import _body, _digest
from src.replay import replay
from src.stage_plan import execute
from src.synthesis_service import solve_input
from tests.compare_versions import apply_limits, load_samples, within_project, write_json
from tests.retiming_audit import hashes
from tests.test_regressions_v11 import perturb_topology


SETTINGS = Settings('http://unused.invalid', '', 'synthetic-offline')


def fixed(run, output):
    summary = json.loads((run / 'summary.json').read_text())
    samples = {sample.label: sample for sample in load_samples(all_public=True)}
    result = {'actual_network_calls': 0, 'actual_model_calls': 0, 'actual_tokens': 0,
              'source_before': hashes(), 'rows': [], 'scope': 'saved real replies, offline causal replay'}
    rows = [row for row in summary['rows'] if row['version'] == '1.6' and row['repeat'] == 1]
    if len(rows) != 14:
        raise ValueError('fourteen recorded first-repeat replies are required')
    for row in rows:
        sample = samples[row['sample']]
        saved = json.loads((run / row['response_file']).read_text())['output']
        proposal = saved['diagnostics']['llm']['response_proposal']
        captured = {}

        def observed(task, graph, control, report, context, reply):
            captured.update(task=task, graph=graph, control=control, report=report, context=context)
            return execute(task, graph, control, report, context, reply)

        with patch('src.synthesis_service.load_settings', return_value=SETTINGS), \
             patch('src.synthesis_service.edit_stages', return_value=(proposal, {'synthetic': True, 'usage': {}})), \
             patch('src.synthesis_service.execute', side_effect=observed), \
             patch('src.llm._chat_completion', side_effect=AssertionError('offline transport')):
            solved = solve_input({'task': sample.task, 'topology': sample.topology})
        task, graph, control, report, context = (captured[key] for key in ('task', 'graph', 'control', 'report', 'context'))
        first, evidence = execute(task, graph, control, report, context, proposal)
        repeated, _ = execute(task, graph, control, report, context, proposal)
        zero = ['keep'] * len(context.bands) if context.mode == 'bands' else {'counts': [0] * len(context.retiming.rounds)}
        bad = ['not-an-edit'] * len(context.bands) if context.mode == 'bands' else {'counts': [9] * len(context.retiming.rounds)}
        zero_answer, _ = execute(task, graph, control, report, context, zero)
        bad_answer, rejection = execute(task, graph, control, report, context, bad)
        oracle = None
        if context.mode == 'bands':
            position, indices = 0, []
            for patterns in context.bands:
                costs = []
                for pattern in patterns:
                    offset, cost = position, 0.0
                    for width in pattern:
                        cost += context.space.transition(offset, width)[2]
                        offset += width
                    costs.append(cost)
                indices.append(min(range(len(costs)), key=costs.__getitem__))
                position += sum(patterns[0])
            best, _ = execute(task, graph, control, report, context, indices)
            free_cost, free_widths = context.space.oracle()
            free_answer, _ = context.space.compile(free_widths)
            free_check = replay(task, graph, free_answer)
            oracle = {'scope': 'offline only, not model output', 'band_indices': indices,
                      'band_best_us': replay(task, graph, best)['total_time_us'],
                      'unrestricted_stage_us': free_cost, 'unrestricted_widths': free_widths,
                      'unrestricted_valid': free_check['valid']}
        record = {'sample': sample.label, 'mode': context.mode, 'proposal': proposal,
                  'control_us': report['total_time_us'], 'final_us': evidence['final_time_us'],
                  'model_selected': evidence['selected'], 'valid': replay(task, graph, first)['valid'],
                  'same_reply_same_answer': _digest(_body(first)) == _digest(_body(repeated)),
                  'matches_recorded': _digest(_body(first)) == _digest(_body(saved['answer'])),
                  'matches_service': _digest(_body(first)) == _digest(_body(solved['answer'])),
                  'zero_unchanged': zero_answer is control,
                  'invalid_unchanged': bad_answer is control and bool(rejection['errors']), 'oracle': oracle}
        result['rows'].append(record)
        write_json(output, result)
        print(json.dumps(record), flush=True)
    result['source_after'] = hashes()
    checks = ('valid', 'same_reply_same_answer', 'matches_recorded', 'matches_service', 'zero_unchanged', 'invalid_unchanged')
    result['valid'] = result['source_before'] == result['source_after'] and all(all(row[k] for k in checks) for row in result['rows'])
    write_json(output, result)
    return result['valid']


def mutations(output):
    result = {'scope': 'synthetic responses; same task IDs with changed actual graphs',
              'actual_network_calls': 0, 'actual_model_calls': 0, 'actual_tokens': 0,
              'source_before': hashes(), 'rows': []}
    for sample in load_samples(all_public=True):
        prompts = set()
        for index, seed in enumerate((211, 307, 401)):
            topology, changes = perturb_topology(sample.topology, seed, preserve_rank_connectivity=True)
            original, seen = deepcopy(topology), []

            def complete(system, body, settings, **kwargs):
                problem = json.loads(body)
                seen.append(_digest(problem))
                if 'bands' in problem:
                    proposal = [list(row)[(index + i) % len(row)] for i, row in enumerate(problem['bands'])]
                else:
                    proposal = {'counts': [(index + i) % 3 for i in range(len(problem['edits']))]}
                return json.dumps(proposal), {'synthetic': True, 'finish_reason': 'stop', 'usage': {}}

            with patch('src.synthesis_service.load_settings', return_value=SETTINGS), \
                 patch('src.llm._chat_completion', side_effect=complete), \
                 patch('src.llm.build_opener', side_effect=AssertionError('offline transport')):
                solved = solve_input({'task': sample.task, 'topology': topology})
            evidence = solved['diagnostics']['search']['model_edit']
            record = {'sample': sample.label, 'seed': seed, 'changes': changes,
                      'valid': replay(sample.task, topology, solved['answer'])['valid'],
                      'nonregressing': evidence['final_time_us'] <= evidence['control_time_us'] + 1e-6,
                      'input_unchanged': topology == original, 'one_synthetic_reply': len(seen) == 1,
                      'prompt_sha256': seen[0] if seen else None}
            prompts.update(seen)
            result['rows'].append(record)
            write_json(output, result)
            print(json.dumps(record), flush=True)
        if len(prompts) != 3:
            raise AssertionError('changed actual graph did not change model context')
    result['source_after'] = hashes()
    checks = ('valid', 'nonregressing', 'input_unchanged', 'one_synthetic_reply')
    result['valid'] = result['source_before'] == result['source_after'] and all(all(row[k] for k in checks) for row in result['rows'])
    write_json(output, result)
    return result['valid']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fixed', 'mutations'))
    parser.add_argument('--run')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = within_project(args.output)
    if output.exists():
        raise ValueError('use a new output file')
    apply_limits()
    passed = fixed(within_project(args.run), output) if args.mode == 'fixed' else mutations(output)
    raise SystemExit(0 if passed else 1)
