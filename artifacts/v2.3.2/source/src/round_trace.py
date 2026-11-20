"""Measure every round of a topology-routed baseline before model consultation."""
from collections import Counter
from math import floor


def profile(task, graph, split):
    ranks = graph.ranks
    n = len(ranks)
    if not 64 <= n <= 256 or n & (n-1):
        return []
    levels = n.bit_length()-1
    gather = task['operator'] == 'all-gather'
    if levels*(1 if gather else 2) > task['max_nonempty_rounds']:
        return []
    size = task['data_size']['per_rank_bytes'] // split
    resources = {('edge', edge): size*1e6/bw for edge,bw in graph.edges.items()}
    resources.update({('pool', pool): size*1e6/bw for pool,bw in graph.resources.items()})
    cache = {}
    def route(src, dst):
        key = src,dst
        if key not in cache:
            path = graph.path(ranks[src],ranks[dst],size)
            edges = list(zip(path,path[1:]))
            incidences = Counter(('edge',e) for e in edges)
            incidences.update(('pool',pool) for e in edges for pool in graph.edge_resources[e])
            terms = [(k,resources[k]*v) for k,v in incidences.items()]
            delay = graph.startup_us+(len(path)-2)*graph.switch_overhead_us
            cache[key] = terms, delay, delay+sum(resources['edge',e] for e in edges)
        return cache[key]

    previous = None
    ready = {0:levels}
    result = []
    labels = {}
    for step in range(levels if gather else 2*levels):
        reduce = not gather and step < levels
        level = step if gather or reduce else 2*levels-1-step
        stride = 1 << level
        if gather:
            # A source owns its own prefix block. Half was ready one round earlier.
            moves = ((src,src^stride,split*stride,split*stride//2 if step else 0)
                     for src in range(n))
        elif reduce:
            moves = ((base+stride,base,split,0) for base in range(0,n,2*stride))
        else:
            moves = ((base,base+stride,split,split if ready[base]<step else 0)
                     for base in range(0,n,2*stride))
        loads = Counter()
        delay = serial = 0.
        fit = deliveries = 0
        if previous:
            last_load, last_delay, last_peak, last_time = previous
            room = last_load.copy()
        for src,dst,count,eligible in moves:
            terms, latency, atom_time = route(src,dst)
            delay = max(delay,latency)
            serial = max(serial,atom_time)
            for key,unit in terms:
                loads[key] += count*unit
            deliveries += count
            if previous and eligible and atom_time<=last_time:
                budget=last_time-max(last_delay,latency)
                if last_peak<=budget+1e-9:
                    take=min(eligible,*(max(0,floor((budget-room[k])/unit+1e-9)) for k,unit in terms))
                    if take:
                        fit+=take
                        last_delay=max(last_delay,latency)
                        for k,unit in terms:
                            room[k]+=take*unit
                            last_peak=max(last_peak,room[k])
            if not gather and not reduce:
                ready[dst]=step+1
        hot, peak = max(loads.items(),key=lambda entry:entry[1])
        duration = max(serial,peak+delay)
        idle=1-sum(loads.values())/(len(loads)*(duration-delay))
        tag=labels.setdefault(hot,('P' if hot[0]=='pool' else 'E')+str(len(labels)))
        result.append(dict(key=hot[1], row=[tag,round(duration,1),round(100*idle),round(100*fit/deliveries)]))
        previous = loads,delay,peak,duration
    return result
