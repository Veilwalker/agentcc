"""Pipeline two-level reduction and broadcast trees over directed access domains."""
from collections import defaultdict
from math import floor
from time import monotonic

from . import domain_cycles, mesh_reduce
from .capacity_backend import native
from .collectives import _initial
from .reduction_parents import regenerate
from .replay import certify


def _collectors(order, distance):
    # Minimize the sum of round-trip cost to a nearby collector while keeping
    # the collectors close to the stripe's root. Reduce and copy may use
    # different parents when the physical links are asymmetric.
    root = order[0]
    chosen = [root]
    distances = distance[root]
    # Every center adds one root reduction and one root broadcast. Leave a
    # larger share of peers on the outer branches to reduce root congestion.
    target = max(2,(2*len(order)+4)//5)
    while len(chosen) < target:
        options = []
        for d in order:
            if d in chosen:
                continue
            closer = list(map(min,distances,distance[d]))
            options.append((sum(closer) + .3*distance[root][d],d,closer))
        _,new,distances = min(options,key=lambda item:item[0])
        chosen.append(new)
    return chosen


def _baseline(task, graph, plan, report, catalog, allowed):
    if native is None or task['operator'] != 'all-reduce' or len(graph.ranks)>256:
        return plan,report,None
    layout = domain_cycles._layout(graph)
    if layout is None:
        return mesh_reduce.improve(task, graph, plan, report, catalog, allowed)
    leaves,groups = layout
    count,width = len(groups),len(groups[0])
    choices = [s for s in allowed if width<=s<=64 and s%width==0]
    if count < 4 or not choices:
        return plan,report,None
    split = max(choices)
    streams = min(2,split//width)
    if 5+streams > task['max_nonempty_rounds']:
        return plan,report,None
    before = report['total_time_us']
    tick = monotonic()
    atom = task['data_size']['per_rank_bytes']//split
    costs = catalog.capacity_costs
    local,_,_ = domain_cycles._local_load(graph,leaves,groups,costs,atom)
    # Any rank in this construction sends and receives S - S/K atoms as
    # a leaf and (S/K)*(K-1) as a local stripe representative.
    bound = max(local.values())*(split+(split//width)*(width-2))
    if bound >= before*.993:
        return plan,report,dict(endpoint_bound_us=bound)
    cores = domain_cycles._cores(graph,leaves,costs,atom)
    if cores is None:
        return plan,report,None
    _,lanes = native.domain_cycles(count,width,len(costs.indices),atom,cores)
    distances = [[0.]*count for _ in groups]
    for a in range(count):
        for b in range(count):
            if a!=b:
                distances[a][b] = min(delay+atom*sum(rate*(3 if i>=len(graph.edges) else 1)
                                                    for i,rate in terms)
                                      for terms,delay,_,_ in cores[a*count+b])
    both_ways = [[distances[a][b]+distances[b][a] for b in range(count)] for a in range(count)]
    stages = [defaultdict(list) for _ in range(5+streams)]
    for slot in range(split):
        root = (slot%count + slot//count*max(1,count//3))%count
        start = (slot//count)%streams
        holders = [d*width+(slot+3*d)%width for d in range(count)]
        for d in range(count):
            for member in range(d*width,(d+1)*width):
                if member!=holders[d]:
                    stages[start][member,holders[d],False].append(slot)
                    stages[start+5][holders[d],member,True].append(slot)
        order = [a for a,_,_ in lanes[slot%width]]
        ix = order.index(root)
        order = order[ix:]+order[:ix]
        centers = _collectors(order,both_ways)
        incoming = defaultdict(int)
        outgoing = defaultdict(int)
        for c in centers[1:]:
            stages[start+2][holders[c],holders[root],False].append(slot)
            stages[start+3][holders[root],holders[c],True].append(slot)
        for d in order:
            if d in centers:
                continue
            up = min(centers,key=lambda c:distances[d][c]*(1+.5*incoming[c]))
            down = min(centers,key=lambda c:distances[c][d]*(1+.5*outgoing[c]))
            incoming[up]+=1
            outgoing[down]+=1
            stages[start+1][holders[d],holders[up],False].append(slot)
            stages[start+4][holders[down],holders[d],True].append(slot)
    ranks = [r for group in groups for r in group]
    cache = {}
    recipe = []
    predicted = 0.
    for stage in stages:
        if not stage:
            continue
        jobs,items = [],[]
        for (src,dst,copy),slots in stage.items():
            key = src,dst
            if key not in cache:
                a,b = src//width,dst//width
                middle = [[leaves[a]]] if a==b else [opt[3] for opt in cores[a*count+b]]
                cache[key] = tuple(costs.compile((ranks[src],*p,ranks[dst])) for p in middle)
            paths = cache[key]
            if len(paths)==1:
                jobs.append((paths,len(slots)*atom,atom))
                items.append((paths,copy,slots))
            else:
                for s in slots:
                    jobs.append((paths,atom,atom))
                    items.append((paths,copy,(s,)))
        selected,_,duration,_ = native.optimize_round(len(costs.indices),jobs,2,costs.round_cache)
        by_path = defaultdict(list)
        for (paths,copy,slots),choice in zip(items,selected):
            by_path[paths[choice][3],copy].extend(slots)
        recipe.append((duration,by_path))
        predicted += duration
    audit = dict(domains=count,cohorts=streams,predicted_us=predicted)
    if predicted >= before*.993:
        return plan,report,audit
    rows = []
    for duration,batch in recipe:
        moves = []
        for (path,copy),slots in batch.items():
            tokens = [f'c{s}_sum' if copy else f'{path[0]}_c{s}' for s in slots]
            _,delay,serial,_ = costs.compile(path)
            packet = max(1,floor((duration-delay)/(atom*serial)+1e-8))
            for i in range(0,len(tokens),packet):
                moves.append(dict(src=path[0],dst=path[-1],action='copy' if copy else 'reduce',
                                  path=list(path),chunk_ranges=tokens[i:i+packet]))
        rows.append(dict(round=len(rows)+1,transfers=moves))
    candidate = _initial(task,graph.ranks,split)
    candidate.update(rounds=regenerate(rows,graph.ranks,split),round_count=len(rows))
    checked = certify(task,graph,candidate)
    audit.update(actual_us=checked['total_time_us'],valid=checked['valid'],elapsed_s=round(monotonic()-tick,6))
    if checked['valid'] and checked['total_time_us'] < before*.993:
        return candidate,checked,audit
    return plan,report,audit

def improve(task, graph, plan, report, catalog, allowed):
    best, checked, detail = _baseline(task, graph, plan, report, catalog, allowed)
    if native is None or task['operator'] != 'all-reduce' or not 64 <= len(graph.ranks) <= 256:
        return best, checked, detail
    layout = domain_cycles._layout(graph)
    if layout is None:
        return best, checked, detail
    leaves, groups = layout
    width = len(groups[0])
    choices = [s for s in allowed if width <= s <= 64 and s % width == 0]
    shared_access = any(graph.edge_resources[rank, leaf] or graph.edge_resources[leaf, rank]
                        for leaf, group in zip(leaves, groups) for rank in group)
    if not choices or shared_access:
        return best, checked, detail
    from . import regional_collectors
    try:
        option = regional_collectors.improve(task,graph,groups,max(choices),catalog,
                                            checked['total_time_us'],exchange=False)
    except (ValueError, KeyError, IndexError, OverflowError):
        option = None
    if option:
        best,checked,regional = option
        detail = dict(detail or {}, **regional)
    return best,checked,detail
