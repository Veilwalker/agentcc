"""Offline causal and perturbation checks for the retiming program.

No API calls or credentials. Real saved model replies are compared with zero
edits, repeated execution, and same-catalog greedy/exhaustive controls. Synthetic
mutation replies test semantics, not model intelligence or official scores.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import time
from unittest.mock import patch

from src.config import Settings
from src.forward_edit import _body, _digest, _round_loads, Context, Rewrite, compile_rewrites
from src.forward_service import solve_input
from src.replay import replay
from src.retime_plan import prepare, execute, _moves
from src.forward_edit import copy_states
from src.route_balance import _PathCosts
from src.solver import solve_schedule, validate_task
from src.topology import Topology
from tests.compare_versions import apply_limits, load_samples, within_project, PROJECT, write_json
from tests.test_regressions_v11 import perturb_topology


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROJECT/'src').glob('*.py'))}


def exact_catalog(task, graph, control, report, bound):
    states, atom_bit = copy_states(task, graph, control)
    cached = {(i, n): Rewrite('advance', _moves(control, graph, states, atom_bit, ri, n))
              for i, ri in enumerate(bound.rounds) for n in (1,2)}
    solutions = {}
    def cost(counts):
        key = tuple(counts)
        if key not in solutions:
            context = Context(bound.input_hash, _digest(_body(control)),
                tuple(cached[i,n] for i,n in enumerate(counts) if n), report['total_time_us'])
            candidate = compile_rewrites(control, context, range(len(context.rewrites)))
            solutions[key] = sum(x.time_us for x in _round_loads(graph, candidate,
                task['data_size']['per_rank_bytes']//control['split'], _PathCosts(graph)))
        return solutions[key]
    counts = [0] * len(bound.rounds)
    for i in range(len(counts)):
        options = [counts[:i]+[n]+counts[i+1:] for n in (0,1,2)]
        counts = min(options, key=lambda c:(cost(c), c))
    greedy = {'counts':counts,'time_us':cost(counts)}
    all_counts = list(itertools.product((0,1,2), repeat=len(counts)))
    best = min(all_counts, key=lambda c:(cost(c), c))
    result, evidence = execute(task, graph, control, report, bound, {'counts':list(best)})
    checked = replay(task, graph, result)
    return greedy, {'counts':list(best),'time_us':checked['total_time_us'],
                    'valid':checked['valid'], 'enumerated':len(all_counts),
                    'prediction_matches_replay':abs(cost(best)-checked['total_time_us'])<1e-5}


def fixed(run, output, version='1.5'):
    summary = json.loads((run/'summary.json').read_text())
    samples = {s.label:s for s in load_samples(all_public=True)}
    result = {'scope':__doc__, 'actual_network_calls':0, 'source_before':hashes(), 'rows':[]}
    rows = [r for r in summary['rows'] if r['version']==version and r['repeat']==1]
    if len(rows) != 14:
        raise ValueError('fixed audit requires fourteen first-repeat records for the selected version')
    for row in rows:
        sample = samples[row['sample']]
        t,g = sample.task,Topology.from_data(sample.topology)
        allowed = validate_task(t,g)
        seed = solve_schedule(t,g,seed_only=True,seed_split=allowed[0] if t['operator']=='all-gather' else allowed[-1],seed_weight=4)
        _,bound,_ = prepare(t,g,seed,seed['strategy']['validation_report'])
        control = solve_input({'task':t,'topology':sample.topology},use_llm=False)['answer']
        report = replay(t,g,control)
        stored = json.loads((run/row['response_file']).read_text())['output']
        proposal = stored['diagnostics']['llm']['response_proposal']
        first,ev = execute(t,g,control,report,bound,proposal)
        repeat,again = execute(t,g,control,report,bound,proposal)
        zero,_ = execute(t,g,control,report,bound,{'counts':[0]*len(bound.rounds)})
        bad,rejection = execute(t,g,control,report,bound,{'counts':[9]*len(bound.rounds)})
        greedy,oracle = exact_catalog(t,g,control,report,bound)
        record = {'sample':sample.label,'proposal':proposal,'control_us':report['total_time_us'],
                  'actual_us':replay(t,g,first)['total_time_us'], 'accepted':ev['accepted'],
                  'valid':replay(t,g,first)['valid'], 'repeat_same':_digest(_body(first))==_digest(_body(repeat)),
                  'matches_recorded':_digest(_body(first))==_digest(_body(stored['answer'])),
                  'zero_unchanged':zero is control, 'bad_unchanged':bad is control and bool(rejection['errors']),
                  'greedy':greedy,'oracle':oracle}
        result['rows'].append(record);write_json(output,result);print(json.dumps(record),flush=True)
    result['source_after']=hashes()
    result['valid']=result['source_before']==result['source_after'] and len(rows)==14 and all(
        all(r[k] for k in ('valid','repeat_same','matches_recorded','zero_unchanged','bad_unchanged'))
        and r['oracle']['valid'] and r['oracle']['prediction_matches_replay'] for r in result['rows'])
    write_json(output,result)
    return result['valid']


def mutations(output):
    settings = Settings('http://unused.test/v1/chat/completions','synthetic-key','synthetic-model')
    result={'scope':__doc__, 'actual_network_calls':0,'source_before':hashes(),'rows':[]}
    for sample in load_samples(all_public=True):
        prompts=set()
        for i,seed in enumerate((211,307,401)):
            topology,changes=perturb_topology(sample.topology,seed,preserve_rank_connectivity=True)
            original=deepcopy(topology)
            seen=[]
            def complete(system,body,settings,**kwargs):
                p=json.loads(body);counts=[(2,1,0)[(i+j)%3] for j in range(len(p['edits']))]
                seen.append(_digest(p))
                return json.dumps({'counts':counts}),{'synthetic':True,'finish_reason':'stop','usage':{}}
            with patch('src.forward_service.load_settings',return_value=settings),patch('src.llm._chat_completion',side_effect=complete),patch('src.llm.build_opener',side_effect=AssertionError('offline network')):
                solved=solve_input({'task':sample.task,'topology':topology})
            d=solved['diagnostics'];ev=d['search']['model_edit']
            record={'sample':sample.label,'seed':seed,'changes':changes,'valid':replay(sample.task,topology,solved['answer'])['valid'],
                    'nonregressing':ev['final_time_us']<=ev['control_time_us']+1e-6,'input_unchanged':original==topology,
                    'one_synthetic_completion':len(seen)==1,'prompt_sha256':seen[0] if seen else None}
            prompts.update(seen);result['rows'].append(record);write_json(output,result);print(json.dumps(record),flush=True)
        if len(prompts)!=3:
            raise AssertionError('material topology perturbations did not change prompt')
    result['source_after']=hashes();result['valid']=result['source_before']==result['source_after'] and all(
        all(r[k] for k in ('valid','nonregressing','input_unchanged','one_synthetic_completion')) for r in result['rows'])
    write_json(output,result);return result['valid']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['fixed','mutations'])
    parser.add_argument('--run')
    parser.add_argument('--version', default='1.5')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=within_project(args.output)
    if output.exists():raise ValueError('use a new output path')
    apply_limits()
    passed=fixed(within_project(args.run),output,args.version) if args.mode=='fixed' else mutations(output)
    raise SystemExit(0 if passed else 1)
