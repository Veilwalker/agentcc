"""Replace a reduce/broadcast star by two cooperating regional collectors."""
from collections import defaultdict
from math import floor

from .capacity_backend import native
from .reduction_parents import regenerate
from .replay import certify


def improve(task, graph, parent, checked, catalog):
    rows, split = parent['rounds'], parent['split']
    if task['operator'] != 'all-reduce' or split > 64 or native is None:
        return parent, checked, None
    center = max((i for i,row in enumerate(rows)
                  if any(m['action']=='reduce' for m in row['transfers'])),default=-1)
    if not 0 <= center < len(rows)-1 or len(rows) >= task['max_nonempty_rounds']:
        return parent, checked, None
    reduction, broadcast = rows[center:center+2]
    if len(reduction['transfers']) + len(broadcast['transfers']) > 3000:
        return parent, checked, None
    incoming, outgoing = [defaultdict(list) for _ in range(split)], [defaultdict(list) for _ in range(split)]
    for row, action, sets in ((reduction,'reduce',incoming),(broadcast,'copy',outgoing)):
        for move in row['transfers']:
            if move['action'] != action:
                return parent, checked, None
            for token in move['chunk_ranges']:
                if action=='copy':
                    slot=int(token[1:-4]);sets[slot][move['src']].append(move['dst'])
                else:
                    slot=int(token.rsplit('_c',1)[1]);sets[slot][move['dst']].append(move['src'])
    slots = []
    for before, after in zip(incoming,outgoing):
        if len(before)!=1 or before.keys()!=after.keys():
            return parent, checked, None
        hub, = before
        senders, receivers = before[hub], after[hub]
        alternatives = sorted(set(senders).intersection(receivers), key=catalog.rank_index.get)
        if not alternatives or len(senders)>24 or len(receivers)>48:
            return parent, checked, None
        slots.append((hub,senders,receivers,alternatives))
    size = task['data_size']['per_rank_bytes']//split
    costs, cache = catalog.capacity_costs, {}
    def paths(u,v):
        if (u,v) not in cache:
            cache[u,v] = tuple(costs.compile(p) for p in catalog.options(u,v))
        return cache[u,v]

    ceiling = sum(checked['round_times_us'][center:center+2])
    best = None
    # Two placements use different actual physical receivers. A piece can send
    # its local partial to either collector. Both then exchange disjoint sums.
    for shift in (0,1):
        jobs = [[],[],[]]
        for s,(hub,senders,receivers,alternatives) in enumerate(slots):
            other = alternatives[(s+shift*max(1,len(alternatives)//2))%len(alternatives)]
            for u in senders:
                if u != other:
                    jobs[0].append((paths(u,hub)+paths(u,other),s))
            jobs[1].extend(((paths(hub,other),s),(paths(other,hub),s)))
            for v in receivers:
                if v != other:
                    jobs[2].append((paths(hub,v)+paths(other,v),s))
        recipe, times = [],[]
        for job in jobs:
            if not job:
                recipe.append({});times.append(0.);continue
            if any(not choices or len(choices)>128 for choices,_ in job):
                break
            packets=[(choices,size,size) for choices,_ in job]
            selection,_,duration,_=native.optimize_round(len(costs.indices),packets,2)
            batches=defaultdict(list)
            for (choices,s),k in zip(job,selection):
                batches[choices[k][3]].append(s)
            recipe.append(batches);times.append(duration)
        if len(recipe)==3 and sum(times)<ceiling-.005*checked['total_time_us']:
            best=recipe,times;ceiling=sum(times)
    if best is None:
        return parent,checked,None
    recipe,times=best
    new_rows=list(rows[:center])
    for i,(groups,duration) in enumerate(zip(recipe,times)):
        moves=[]
        for path,ss in groups.items():
            tokens=[f'c{s}_sum' if i==2 else f'{path[0]}_c{s}' for s in ss]
            _,delay,inverse,_=costs.paths[path]
            width=max(1,floor((duration-delay)/(size*inverse)+1e-10))
            for j in range(0,len(tokens),width):
                moves.append(dict(src=path[0],dst=path[-1],path=list(path),
                                  action='copy' if i==2 else 'reduce',chunk_ranges=tokens[j:j+width]))
        if moves:
            new_rows.append(dict(transfers=moves))
    new_rows.extend(rows[center+2:])
    new_rows=regenerate([dict(row,round=i+1) for i,row in enumerate(new_rows)],graph.ranks,split)
    candidate=dict(parent,rounds=new_rows,round_count=len(new_rows))
    report=certify(task,graph,candidate)
    predicted=checked['total_time_us']-sum(checked['round_times_us'][center:center+2])+sum(times)
    if report['valid'] and abs(report['total_time_us']-predicted)<max(1e-4,predicted*1e-9):
        return candidate,report,center+1
    return parent,checked,None
