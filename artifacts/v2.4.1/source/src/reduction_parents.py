"""Identify independently reparentable retirement trees and regenerate expressions."""
from collections import defaultdict


def eligibility(solution, rank_ids):
    rounds=solution['rounds']; split=solution['split']; n=len(rank_ids)
    slots={}
    events=[[] for _ in range(split)]
    for i,row in enumerate(rounds):
        for tr in row['transfers']:
            if tr['action']!='reduce':continue
            for tok in tr['chunk_ranges']:
                if tok not in slots:slots[tok]=int(tok.rsplit('_c',1)[1])
                events[slots[tok]].append((i,rank_ids[tr['src']],rank_ids[tr['dst']]))
    masks={}
    for s,es in enumerate(events):
        if not es:continue
        last=max(r for r,_,_ in es)
        # Each retired subtree sends exactly once. The terminal step is
        # either a single root or a complete directed exchange of partials.
        earlier=[(r,u,v) for r,u,v in es if r<last]
        times={u:r for r,u,v in earlier}
        if len(times)!=len(earlier):continue
        live=set(range(n))-times.keys()
        if any(v in times and times[v]<=r for r,u,v in earlier):continue
        closing=[(u,v) for r,u,v in es if r==last]
        if not closing or any(u not in live or v not in live for u,v in closing):continue
        sinks=live-{u for u,_ in closing}
        if len(sinks)==1 and len(closing)==len(live)-1 and all(v in sinks for u,v in closing):
            times.update((u,last) for u,v in closing)
        elif len(closing)!=len(live)*(len(live)-1) or len(set(closing))!=len(closing):
            continue
        else:
            times.update((u,last) for u in live)
        for r in sorted(set(times.values())):
            mask=sum(1<<u for u in range(n) if times.get(u,last+1)>r)
            if mask:masks[s,r]=mask
    return slots,masks


def regenerate(rows, ranks, split):
    # Reparenting changes which contributions a sender possesses. Materialize
    # the exact round-start sets only after all round flow choices are known.
    state={r:[f'{r}_c{s}' for s in range(split)] for r in ranks}
    slots={}
    result=[]
    for row in rows:
        arrivals=defaultdict(list)
        moves=[]
        for tr in row['transfers']:
            if tr['action']=='copy':
                moves.append(tr)
                continue
            values=[]
            for token in tr['chunk_ranges']:
                if token not in slots:slots[token]=int(token.rsplit('_c',1)[1])
                slot=slots[token]; value=state[tr['src']][slot]
                values.append(value);arrivals[tr['dst'],slot].append(value)
            moves.append(dict(tr,chunk_ranges=values))
        for (r,s),parts in arrivals.items():state[r][s]='+'.join([state[r][s],*parts])
        result.append(dict(row,transfers=moves))
    return result
