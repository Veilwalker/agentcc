"""Offline model-code replay, module ablation and actual-topology perturbations.

Uses recorded model source, never credentials or a network request. The numeric
optimum in an ablation is an explicitly labelled offline reference, not an LLM
answer. Every run writes a new report and keeps its source fingerprints.
"""
import argparse
from copy import deepcopy
import json
from unittest.mock import patch

from src.forward_edit import _body, _digest
from src.generated_service import execute_program, fixed_band_guard, solve_input
from src.replay import replay
from src.schedule_edit import _input_digest
from src.refined_search import RefinedSearchSpace as SearchSpace
from src.topology import Topology
from tests.compare_versions import apply_limits, load_samples, within_project, write_json
from tests.retiming_audit import hashes
from tests.test_regressions_v11 import perturb_topology


def recorded(run, label):
    summary = json.loads((run / 'summary.json').read_text())
    version = next(row['version'] for row in summary['versions'] if row['current'])
    path = run / ('v' + version) / (label.replace('/', '-') + '-r01.json')
    output = json.loads(path.read_text())['output']
    execution = output['diagnostics']['search']['model_edit']['executions'][-1]
    return output, execution['source'], execution['program_protocol']


def optimum(table, limit):
    """Offline reference only; independent of the production model interpreter."""
    states = {(0, 0): 0.0}
    for i, row in enumerate(table):
        for steps in range(limit):
            if (i, steps) in states:
                for width, cost in enumerate(row, 1):
                    key = i + width, steps + 1
                    states[key] = min(states.get(key, float('inf')), states[i, steps] + cost)
    return min(states.get((len(table), steps), float('inf')) for steps in range(1, limit + 1))


def fixed(samples, run, report, output):
    for sample in samples:
        saved, code, protocol = recorded(run, sample.label)
        captured = []

        def build(*args, **kwargs):
            space = SearchSpace(*args, **kwargs)
            captured.append(space)
            return space

        meta = {'recorded_replay': True, 'usage': {}, 'program_protocol': protocol}
        with patch('src.generated_service.load_settings'), \
             patch('src.generated_service.propose', return_value=(code, meta)), \
             patch('src.generated_service.SearchSpace', side_effect=build), \
             patch('src.llm.build_opener', side_effect=AssertionError('offline transport')):
            solved = solve_input({'task': sample.task, 'topology': sample.topology})
        space = captured[0]
        first = execute_program(code, space, protocol)
        second = execute_program(code, space, protocol)
        candidate, predicted = space.compile(first['output'])
        repeated, _ = space.compile(second['output'])
        checked = replay(sample.task, sample.topology, candidate)
        guard = fixed_band_guard(space)
        guard_check = replay(sample.task, sample.topology, guard[0]) if guard else None
        plain = []
        for profile in space.profiles:
            s, d = profile.plain, space.physical_dimensions
            if space.independent_ar:
                table = [[s.transition(i,w) and s.phase_times[i,w][0]
                          for w in range(1, min(3,d-i)+1)] for i in range(d)]
                table += [[s.transition(d-i-w,w) and s.phase_times[d-i-w,w][1]
                           for w in range(1,min(3,d-i)+1)] for i in range(d)]
            else:
                table = [[s.transition(i,w)[2] for w in range(1,min(3,d-i)+1)] for i in range(d)]
            plain.append(optimum(table, space.limit))
        audit = solved['diagnostics']['search']['model_edit']
        row = {'sample': sample.label, 'valid': checked['valid'],
               'predicted_matches_replay': abs(predicted-checked['total_time_us']) < 1e-4,
               'same_program_same_compiled_schedule': _digest(_body(candidate)) == _digest(_body(repeated)),
               'same_recorded_answer': _digest(_body(saved['answer'])) == _digest(_body(solved['answer'])),
               'control_with_guard_us': audit['control_time_us'], 'model_candidate_us': predicted,
               'final_us': audit['final_time_us'], 'model_selected': audit['selected'],
               'guard_us': guard_check['total_time_us'] if guard_check else None,
               'plain_global_offline_optimum_us': min(plain),
               'refined_global_offline_optimum_us': space.audit_bound(),
               'guard_valid': guard_check['valid'] if guard_check else True}
        report['rows'].append(row)
        write_json(output, report)
        print(json.dumps(row), flush=True)
    keys = ('valid', 'predicted_matches_replay', 'same_program_same_compiled_schedule', 'guard_valid')
    return all(all(row[key] for key in keys) for row in report['rows'])


def mutations(samples, run, report, output):
    for sample in samples:
        saved, code, protocol = recorded(run, sample.label)
        baseline = _input_digest(sample.task, Topology.from_data(sample.topology))
        observed = set()
        for seed in (211, 307, 401):
            topology, changes = perturb_topology(sample.topology, seed, preserve_rank_connectivity=True)
            before = deepcopy(topology)
            input_hash = _input_digest(sample.task, Topology.from_data(topology))
            meta = {'recorded_replay': True, 'usage': {}, 'program_protocol': protocol}
            with patch('src.generated_service.load_settings'), \
                 patch('src.generated_service.propose', return_value=(code,meta)) as proposed, \
                 patch('src.llm.build_opener', side_effect=AssertionError('offline transport')):
                solved = solve_input({'task': sample.task, 'topology': topology})
            audit = solved['diagnostics']['search']['model_edit']
            space = audit['space']
            available = 'unavailable' not in space
            observed.add(input_hash)
            row = {'sample': sample.label, 'seed': seed, 'changes': changes,
                   'valid': replay(sample.task, topology, solved['answer'])['valid'],
                   'nonregressing_against_control': audit['final_time_us'] <= audit['control_time_us'] + 1e-6,
                   'input_unchanged': before == topology,
                   'actual_input_changed': input_hash != baseline,
                   'input_sha256': input_hash,
                   'space_available': available,
                   'space_input_matches': space.get('input_sha256') == input_hash if available else None,
                   'fallback_reason': space.get('unavailable'),
                   'stop_reason': audit['stop_reason'],
                   'model_context_sha256': _digest(proposed.call_args_list[0].args[0]),
                   'cost_table_sha256': _digest(space['cost_tables_us']) if available else None,
                   'program_executed': bool(audit['executions']),
                   'program_valid': all(e['valid_program'] for e in audit['executions']) if audit['executions'] else None,
                   'final_us': audit['final_time_us']}
            report['rows'].append(row)
            write_json(output, report)
            print(json.dumps(row), flush=True)
        if len(observed) != 3:
            raise AssertionError('input fingerprints did not change after topology perturbations')
    keys = ('valid', 'nonregressing_against_control', 'input_unchanged', 'actual_input_changed')
    report['generated_space_cases'] = sum(row['space_available'] for row in report['rows'])
    report['explicit_fallback_cases'] = len(report['rows']) - report['generated_space_cases']
    return all(all(row[key] for key in keys) and (
        row['space_input_matches'] and row['program_executed'] and row['program_valid']
        if row['space_available'] else
        bool(row['fallback_reason']) and row['stop_reason'] == 'capacity_space_unavailable'
        and not row['program_executed']) for row in report['rows'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fixed', 'mutations'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, output = within_project(args.run), within_project(args.output)
    if output.exists():
        raise ValueError('use a new output file')
    apply_limits()
    report = {'mode': args.mode, 'scope': 'recorded model code, offline replay',
              'actual_model_calls': 0, 'actual_tokens': 0, 'source_before': hashes(), 'rows': []}
    valid = (fixed if args.mode == 'fixed' else mutations)(load_samples(all_public=True), run, report, output)
    report['source_after'] = hashes()
    report['valid'] = valid and report['source_before'] == report['source_after']
    write_json(output, report)
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
