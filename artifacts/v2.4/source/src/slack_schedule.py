"""Reassign ready copies to unused byte capacity across a verified schedule."""
from collections import defaultdict
from math import floor
from time import monotonic

from .capacity_backend import native
from .replay import certify


def prepare(task, graph, plan, report, catalog):
    started = monotonic()
    rows = plan['rounds']
    split = plan['split']
    gather = task['operator'] == 'all-gather'
    slots = len(graph.ranks) * split if gather else split
    costs = catalog.capacity_costs
    if (native is None or len(rows) < 2 or len(rows) > 128 or not report.get('valid')
            or slots*len(graph.ranks) > 530000 or len(costs.indices)*len(rows) > 3000000
            or sum(len(row['transfers']) for row in rows) > 60000):
        return None
    size = task['data_size']['per_rank_bytes'] // split
    codes, owners = {}, []
    if gather:
        for rank, tokens in plan['initial_possession'].items():
            for token in tokens:
                codes[token] = len(owners)
                owners.append(catalog.rank_index[rank])
    else:
        owners = [-1] * split
    routes, route_id, packets, tokens_by_packet = [], {}, [], []
    for t, row in enumerate(rows):
        for move in row['transfers']:
            path = tuple(move['path'])
            p = route_id.get(path)
            if p is None:
                p = len(routes)
                route_id[path] = p
                routes.append(costs.compile(path))
            tokens = move['chunk_ranges']
            if not gather:
                for token in tokens:
                    if token not in codes:
                        codes[token] = int(token[1:-4] if token.endswith('_sum') else token.rsplit('_c', 1)[1])
            ids = tuple(map(codes.__getitem__, tokens))
            copy = move['action'] == 'copy'
            packets.append((t, p, catalog.rank_index[path[0]], catalog.rank_index[path[-1]], copy, ids))
            tokens_by_packet.append(tokens)
    return (slots,size,costs,routes,packets,owners,tokens_by_packet,monotonic()-started)


def improve(task, graph, plan, report, catalog, *, focus=None, prepared=None, prefer_relay=False):
    started = monotonic()
    data = prepared if prepared is not None else prepare(task,graph,plan,report,catalog)
    if data is None:
        return plan, report, None
    rows = plan['rounds']
    slots,size,costs,routes,packets,owners,tokens_by_packet,prep_time = data
    keys = list(costs.indices)
    selected = costs.indices.get(('pool' if isinstance(focus, str) else 'edge', focus), -1)
    before = report['total_time_us']
    assignment, path_bytes, stats = native.fill_slack(len(graph.ranks), slots, len(costs.indices), size, selected,
                                           routes, packets, owners, list(report['round_times_us']), prefer_relay)
    new_paths = memoryview(path_bytes).cast('i')
    predicted = sum(row[0] for row in stats)
    moved = sum(row[4] for row in stats)
    audit = dict(before_us=before, predicted_us=predicted, moved_atoms=moved, accepted=False,
                 prepare_s=round(prep_time,6), elapsed_s=round(monotonic()-started,6), rounds=[
                     dict(round=i+1, before_us=round(report['round_times_us'][i],4),
                          peak_busy_us=round(peak,4), peak_resource=keys[hot],
                          peak_bytes=round(peak*1e-6*(graph.edges[keys[hot][1]] if keys[hot][0]=='edge'
                                                     else graph.resources[keys[hot][1]])),
                          mean_used_resource_slack=round(slack,4),
                          ready_earlier=ready, proposed_out=depart, proposed_in=arrive,alternate_path_atoms=alternate)
                     for i, (_,peak,slack,ready,depart,arrive,alternate,hot) in enumerate(stats)])
    if not moved or predicted >= before*.999:
        return plan, report, audit
    batches = [defaultdict(list) for _ in rows]
    offset = 0
    examples = []
    for (original,p,_,_,copy,_), tokens in zip(packets,tokens_by_packet):
        key = p, 'copy' if copy else 'reduce'
        k = len(tokens)
        dests = assignment[offset:offset+k]
        offset += k
        if dests.count(original) == k:
            batches[original][key].extend(tokens)
        else:
            for t, token, route in zip(dests, tokens, new_paths[offset-k:offset]):
                batches[t][route,key[1]].append(token)
                if t!=original and len(examples)<4:
                    path=routes[route][3]
                    examples.append(dict(from_round=original+1,to_round=t+1,atom=token,src=path[0],dst=path[-1]))
    schedule = []
    for row, (ceiling,*_) in zip(batches, stats):
        moves = []
        for (p, action), tokens in row.items():
            _, delay, serial, path = routes[p]
            width = max(1, floor((ceiling-delay)/(size*serial)+1e-10))
            for i in range(0,len(tokens),width):
                moves.append(dict(src=path[0],dst=path[-1],path=list(path),action=action,
                                  chunk_ranges=tokens[i:i+width]))
        if moves:
            schedule.append(dict(round=len(schedule)+1,transfers=moves))
    candidate = dict(plan, rounds=schedule, round_count=len(schedule))
    checked = certify(task, graph, candidate)
    audit.update(elapsed_s=round(monotonic()-started,6),verified=checked['valid'])
    if checked['valid'] and abs(checked['total_time_us']-predicted) < max(1e-4,predicted*1e-9):
        audit.update(accepted=True,after_us=checked['total_time_us'],examples=examples)
        return candidate, checked, audit
    return plan, report, audit
