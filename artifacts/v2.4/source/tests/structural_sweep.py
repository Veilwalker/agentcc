"""Offline structural hypotheses; no API calls or production dispatch by names.

Compare wider/longer ready-copy retiming and mixed-radix communication stages.
This is an exploratory search, not a production candidate-selection algorithm.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import math
import time

from src.collectives import _initial, _round, locality_order
from src.forward_edit import Move, Rewrite, Context, copy_states, compile_rewrites, _round_loads
from src.packetization import repacketize
from src.replay import replay
from src.route_balance import _PathCosts
from src.topology import Topology
from tests.compare_versions import load_samples, within_project, write_json, apply_limits


def radix(task, graph, split, order, widths, weight=4.0):
    """Fuse adjacent XOR dimensions, directly delivering every disjoint block."""
    n = len(order)
    if n & (n - 1) or sum(widths) != (n - 1).bit_length():
        raise ValueError('widths must partition all rank dimensions')
    solution = _initial(task, list(graph.ranks), split)
    size = task['data_size']['per_rank_bytes'] // split
    if task['operator'] == 'all-gather':
        held = [[r] for r in order]
        remaining = (n - 1).bit_length()
        for width in widths:
            remaining -= width
            masks = [j << remaining for j in range(1, 1 << width)]
            movements = {}
            for i, src in enumerate(order):
                atoms = [c for owner in held[i] for c in solution['initial_possession'][owner]]
                for mask in masks:
                    movements['copy', src, order[i ^ mask]] = atoms
            _round(solution, movements, graph, size, weight, math.inf, 2)
            held = [sum((held[i ^ mask] for mask in [0, *masks]), []) for i in range(n)]
    else:
        rounds = [defaultdict(list) for _ in range(2 * len(widths))]
        stride = n // split + 1
        stride += stride % 2 == 0
        for slot in range(split):
            root = slot * stride % n
            perm = [order[i ^ root] for i in range(n)]
            distance = 1
            for stage, width in enumerate(widths):
                factor = 1 << width
                for start in range(0, n, distance * factor):
                    dst = perm[start]
                    for j in range(1, factor):
                        first = start + j * distance
                        src = perm[first]
                        token = '+'.join(f'{r}_c{slot}' for r in perm[first:first + distance])
                        rounds[stage]['reduce', src, dst].append(token)
                        rounds[-stage - 1]['copy', dst, src].append(f'c{slot}_sum')
                distance *= factor
        for movements in rounds:
            _round(solution, movements, graph, size, weight, math.inf, 2)
    return solution


def retime(task, graph, control, window, lag, amount):
    states, bit = copy_states(task, graph, control)
    index = {r:i for i,r in enumerate(graph.ranks)}
    rounds = [i for i,row in enumerate(control['rounds']) if i >= lag and any(t['action']=='copy' for t in row['transfers'])][-window:]
    moves = []
    for ri in rounds:
        for ti,t in enumerate(control['rounds'][ri]['transfers']):
            if t['action'] != 'copy':continue
            mask = states[ri-lag][index[t['src']]]
            ready = [a for a in t['chunk_ranges'] if bit(a)&mask]
            atoms = tuple(ready[:(len(ready)*amount+1)//2])
            if atoms:moves.append(Move(ri,ti,atoms,t['src'],ri-lag,tuple(t['path'])))
    return compile_rewrites(control,Context('offline','offline',(Rewrite('advance',tuple(moves)),),0),[0])


def run(output, families=None):
    results={'scope':__doc__,'network_calls':0,'rows':[]}
    base=Path('artifacts/v1.5/runs/revision-04/verified-sdk')
    summary=json.loads((base/'summary.json').read_text())
    for sample in load_samples(families,all_public=families is None):
        graph=Topology.from_data(sample.topology);task=sample.task
        record=next(r for r in summary['rows'] if r['version']=='1.5.4' and r['sample']==sample.label and r['repeat']==1)
        control=json.loads((base/record['response_file']).read_text())['output']['answer']
        checked=replay(task,graph,control);assert checked['valid']
        best=checked['total_time_us']; costs=_PathCosts(graph); size=task['data_size']['per_rank_bytes']//control['split']
        dimensions=(len(graph.ranks)-1).bit_length(); patterns=[]
        for first in range(dimensions-1):patterns.append([1]*first+[2]+[1]*(dimensions-first-2))
        patterns += [[2]*(dimensions//2)+[1]*(dimensions%2)]
        options=[]
        for window in (3,5,8):
            for lag in (1,2,3):
                for amount in (1,2):options.append(('retime',dict(window=window,lag=lag,amount=amount)))
        for order in ('input','topology'):
            for widths in patterns:options.append(('radix',dict(order=order,widths=widths)))
        for kind,params in options:
            start=time.process_time()
            candidate=(retime(task,graph,control,**params) if kind=='retime' else radix(task,graph,control['split'],list(graph.ranks) if params['order']=='input' else locality_order(graph),params['widths']))
            predicted=sum(x.time_us for x in _round_loads(graph,candidate,size,costs))
            candidate,packets=repacketize(task,graph,candidate,None,max_extra_packets=1024)
            predicted=packets['predicted_after_us']
            # Validate every candidate: these constructors are new hypotheses.
            report=replay(task,graph,candidate)
            row={'sample':sample.label,'kind':kind,'params':params,'control_us':checked['total_time_us'],'time_us':report['total_time_us'],'valid':report['valid'],'errors':report['errors'][:2],'predicted_matches':abs(predicted-report['total_time_us'])<1e-5,'cpu_s':time.process_time()-start,'rounds':len(candidate['rounds']),'transfers':sum(len(r['transfers']) for r in candidate['rounds'])}
            results['rows'].append(row)
            if row['valid'] and row['time_us']<best:
                best=row['time_us'];print(json.dumps(row),flush=True)
        write_json(output,results)
    results['valid']=all(r['valid'] and r['predicted_matches'] for r in results['rows']);write_json(output,results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--samples',nargs='+');a=p.parse_args()
    output=within_project(a.output)
    if output.exists():raise ValueError('use a fresh output')
    apply_limits();run(output,a.samples)
