"""Shorten pipeline drain time by choosing launch rounds separately for each tree."""
from collections import defaultdict
from math import floor
from time import monotonic

from .capacity_backend import native
from .collectives import _initial
from .replay import certify


def _packets(plan, catalog, rank_index):
    cached = getattr(catalog, 'pipeline_template', None)
    if cached and cached[0] is plan['rounds']:
        stages = (row.items() for row in cached[1])
    else:
        stages = (((tuple(move['path']), move['chunk_ranges']) for move in row['transfers'])
                  for row in plan['rounds'])
    # Every source has the same atom size. One certified multicast tree per
    # source can deliver each of that source's shards on exactly the same arcs.
    token_ids = {f'{rank}_c0': index for rank, index in rank_index.items()}
    first, routes, route_ids, packets = {}, [], {}, []
    depth = 0
    for t, row in enumerate(stages):
        for path, payload in row:
            by_age = defaultdict(list)
            for token in payload:
                owner = token_ids.get(token)
                if owner is not None:
                    age = t-first.setdefault(owner,t)
                    by_age[age].append(owner)
            if not by_age:
                continue
            number = route_ids.get(path)
            if number is None:
                number = len(routes)
                routes.append(catalog.capacity_costs.compile(path))
                route_ids[path] = number
            for age, owners in by_age.items():
                depth = max(depth,age+1)
                packets.append((age,number,tuple(owners)))
    return depth, routes, packets


def _emit(task, graph, split, routes, packets, delays, times):
    candidate = _initial(task,graph.ranks,split)
    size = task['data_size']['per_rank_bytes']//split
    launches = []
    for i, rank in enumerate(graph.ranks):
        by_time = defaultdict(list)
        for slot, token in enumerate(candidate['initial_possession'][rank]):
            by_time[delays[i*split+slot]].append(token)
        launches.append(tuple(by_time.items()))
    if native is not None and hasattr(native, 'emit_paced_launches') and len(graph.ranks) >= 64:
        try:
            rows = native.emit_paced_launches(routes, packets, launches, times, size)
        except (TypeError, ValueError, OverflowError, MemoryError):
            pass
        else:
            candidate.update(rounds=rows, round_count=len(rows))
            return candidate
    batches = [defaultdict(list) for _ in times]
    for age,route,owners in packets:
        for owner in owners:
            for delay,tokens in launches[owner]:
                batches[delay+age][route].extend(tokens)
    for row,limit in zip(batches,times):
        if not row:
            continue
        moves = []
        for route,tokens in row.items():
            _, latency, inverse, path = routes[route]
            width = max(1,floor((limit-latency)/(size*inverse)+1e-10))
            for i in range(0,len(tokens),width):
                moves.append(dict(src=path[0],dst=path[-1],action='copy',path=list(path),
                                  chunk_ranges=tokens[i:i+width]))
        candidate['rounds'].append(dict(round=len(candidate['rounds'])+1,transfers=moves))
    candidate['round_count'] = len(candidate['rounds'])
    return candidate


def improve(task, graph, plan, report, catalog):
    split, n = plan['split'], len(graph.ranks)
    if (task['operator'] != 'all-gather' or not 2 <= split <= 8 or n*split > 2048
            or len(plan['rounds']) > split+4 or n*(n-1)*split > 530000):
        return plan, report, None
    started = monotonic()
    depth,routes,packets = _packets(plan,catalog,catalog.rank_index)
    if not 2 <= depth <= 6:
        return plan, report, None
    # A longer causal chain leaves fewer extra launches inside the round
    # budget. Use a bounded search without expanding the declared shard count.
    lanes = min(12, task['max_nonempty_rounds']-depth+1, max(4,2*split+1-3*max(0,depth-3)))
    if lanes<2:
        return plan, report, None
    delays,times = native.paced_launch(depth,len(catalog.capacity_costs.indices),n,lanes,
        task['data_size']['per_rank_bytes']//split,routes,packets,split)
    predicted = sum(times)
    audit = dict(parent_us=report['total_time_us'], predicted_us=predicted, accepted=False,
                 depth=depth, launch_rounds=lanes, split=split)
    if predicted < report['total_time_us']*.984:
        candidate = _emit(task,graph,split,routes,packets,delays,times)
        checked = certify(task,graph,candidate)
        if checked['valid'] and abs(checked['total_time_us']-predicted)<max(1e-4,predicted*1e-9):
            plan,report = candidate,checked
            audit.update(accepted=True,launch_counts=[delays.count(i) for i in range(lanes)])
    audit['elapsed_s'] = round(monotonic()-started,6)
    return plan,report,audit
