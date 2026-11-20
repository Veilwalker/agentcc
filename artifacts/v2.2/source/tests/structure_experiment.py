"""Offline cyclic-stage ablation; no model calls and no estimated official score.

Each cyclic stage has the same terminal ownership as a dense exchange, but
forwards blocks through rank endpoints over distinct rounds. Capacity previews
reuse one routing plan because every step has the same endpoints and byte sizes.
"""
import argparse
import json
from pathlib import Path
import time

from src.balanced_stages import BalancedStageSpace
from src.collectives import _initial
from src.refined_search import RefinedSearchSpace
from src.replay import replay
from src.topology import Topology
from tests.compare_versions import apply_limits, load_samples


class CyclicStage(BalancedStageSpace):
    def preview(self, first, width, direction=1, gray=False):
        shift, factor = self.dimensions-first-width, 1 << width
        digits = [i ^ (i >> 1) if gray else i for i in range(factor)]
        next_digit = {digits[i]: digits[(i+direction) % factor] for i in range(factor)}
        moves = {}
        for i, rank in enumerate(self.order):
            digit = (i >> shift) & (factor-1)
            dst = i ^ ((digit ^ next_digit[digit]) << shift)
            moves['copy', rank, self.order[dst]] = (1 << first)*self.split
        plans, duration = self._route_counts(moves)
        return (factor-1)*duration, (plans, digits, direction, first, width)

    def materialize(self, state):
        plans, digits, direction, first, width = state
        shift, factor = self.dimensions-first-width, 1 << width
        positions = {rank:i for i,rank in enumerate(self.order)}
        inverse = {digit:i for i,digit in enumerate(digits)}
        rounds = []
        for step in range(factor-1):
            moves = {}
            for action,source,destination,*_ in plans:
                i = positions[source]
                digit = (i >> shift) & (factor-1)
                origin_digit = digits[(inverse[digit]-step*direction) % factor]
                origin = i ^ ((digit ^ origin_digit) << shift)
                owners = [self.order[origin ^ (j << (self.dimensions-first))]
                          for j in range(1 << first)]
                moves[action,source,destination] = [f'{r}_c{s}' for r in owners for s in range(self.split)]
            rounds.append(self._materialize(plans,moves))
        return rounds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('retain previous experiment output')
    apply_limits()
    rows=[]
    for sample in load_samples(None,all_public=True):
        if sample.task['operator'] != 'all-gather':
            continue
        graph=Topology.from_data(sample.topology)
        start=time.process_time()
        space=RefinedSearchSpace(sample.task,graph,1)
        baseline_cpu=time.process_time()-start
        start=time.process_time()
        candidates=[]
        for index,profile in enumerate(space.profiles):
            plain=profile.plain
            cyclic=CyclicStage(sample.task,graph,plain.split,order=plain.order,catalog=plain.catalog)
            best={plain.dimensions:(0.,[])}
            for first in reversed(range(plain.dimensions)):
                choices=[]
                for width,old in enumerate(space.costs[index][first],1):
                    rest,path=best[first+width]
                    choices.append((old+rest,[('dense',first,width,None)]+path))
                    if width==1:
                        continue
                    for direction,gray in [(1,False),(-1,False),(1,True)]:
                        cost,state=cyclic.preview(first,width,direction,gray)
                        choices.append((cost+rest,[('cyclic',first,width,state)]+path))
                best[first]=min(choices,key=lambda x:x[0])
            cost,sequence=best[0]
            answer=_initial(sample.task,list(graph.ranks),plain.split)
            rounds=[]
            for kind,first,width,state in sequence:
                if kind=='cyclic':
                    rounds.extend(cyclic.materialize(state))
                else:
                    chosen=profile.selected[first,width,None]
                    plans=chosen.transition(first,width)[0]
                    movements=chosen._movements(first,width,atoms=True)[0]
                    rounds.append(chosen._materialize(plans,movements))
            answer['rounds']=[{'round':i+1,'transfers':x} for i,x in enumerate(rounds)]
            answer['round_count']=len(rounds)
            checked=replay(sample.task,graph,answer)
            assert checked['valid'],checked['errors']
            assert abs(checked['total_time_us']-cost)<1e-4
            candidates.append({'cost_us':cost,'profile':index,'rounds':len(rounds),
                               'sequence':[(k,f,w) for k,f,w,_ in sequence]})
        row={'sample':sample.label,'baseline_us':space.audit_bound(),
             'baseline_cpu_s':baseline_cpu,'extra_cpu_s':time.process_time()-start,
             'candidates':candidates,'best':min(candidates,key=lambda x:x['cost_us'])}
        rows.append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
