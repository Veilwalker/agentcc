"""Relay pending AllGather deliveries across partially overlapping access ports."""
from collections import defaultdict
from math import floor

from .capacity_backend import native
from .group_sketch import access_groups
from .replay import certify


def _attempt(task, graph, parent, checked, catalog, tapered):
    rows = parent['rounds']
    if (native is None or task['operator'] != 'all-gather' or parent['split'] != 1
            or not 6 <= len(rows) <= 20 or not 64 <= len(graph.ranks) <= 256):
        return parent, checked, None
    ports = catalog.starts
    if not all(1 <= len(ports.get(r, ())) <= 3 for r in graph.ranks):
        return parent, checked, None
    groups = access_groups(graph)
    if not 2 <= len(groups) <= 16 or max(map(len, groups)) > 48:
        return parent, checked, None
    # Different access-port sets inside one domain need a local relay to reach
    # every receiver; a single shared switch already provides a one-hop star.
    if all(len({ports[r] for r in ranks}) == 1 for ranks in groups):
        return parent, checked, None
    home = {r: d for d, members in enumerate(groups) for r in members}
    first = 0 if tapered else len(rows) - 4
    tail = rows[first:]
    crosses = []
    for i, row in enumerate(tail):
        cross = [m for m in row['transfers'] if home[m['src']] != home[m['dst']]]
        crosses.append(cross)
    ranks = graph.ranks
    n = len(ranks)
    ids = catalog.rank_index
    labels = [f'{r}_c0' for r in ranks]
    owner_id = {s: i for i, s in enumerate(labels)}
    if tapered:
        missing = bytearray([1]) * (n * n)
        waiting = set()
        for owner, rank in enumerate(ranks):
            missing[owner * n + owner] = 0
            waiting.update((owner, d) for d in range(len(groups)) if home[rank] != d)
    else:
        missing = bytearray(n * n)
        for row in tail:
            for move in row['transfers']:
                dst = ids[move['dst']]
                for token in move['chunk_ranges']:
                    index = owner_id[token] * n + dst
                    if missing[index]:
                        return parent, checked, None
                    missing[index] = 1
        # The already verified final possession and unique deliveries give the
        # exact initial ownership snapshot of this suffix.
        members = [[ids[r] for r in group] for group in groups]
        waiting = {(owner, d) for owner in range(n) for d, group in enumerate(members)
                   if all(missing[owner * n + dst] for dst in group)}
    arriving = {}
    for t, batch in enumerate(crosses):
        for m in batch:
            owner = ids[m['src']]
            key = owner, home[m['dst']]
            if key in waiting and key not in arriving and labels[owner] in m['chunk_ranges']:
                arriving[key] = t, m['path'], owner
    injection_rounds = {t for t, _, _ in arriving.values()}
    if not waiting or arriving.keys() != waiting:
        return parent, checked, None
    if tapered:
        if len(injection_rounds) < 5:
            return parent, checked, None
        depth = min(12, max(5, len(injection_rounds) - 1), task['max_nonempty_rounds'] - 2)
        stages = depth + 2
    elif injection_rounds == {0, 1}:
        stages = len(tail)
    else:
        return parent, checked, None

    size = task['data_size']['per_rank_bytes']
    routes = []
    route_ids = {}
    def register(path):
        key = tuple(path)
        index = route_ids.get(key)
        if index is None:
            index = len(routes)
            routes.append((ids[key[0]], ids[key[-1]], catalog.capacity_costs.compile(key)))
            route_ids[key] = index
        return index
    for group in groups:
        for u in group:
            for v in group:
                if u == v:
                    continue
                if (u, v) in graph.edges:
                    register((u, v))
                for switch, _ in catalog.outgoing[u]:
                    if (switch, v) in graph.edges:
                        register((u, switch, v))
    local_count = len(routes)
    fixed = [(t, register(path), owner) for t, path, owner in arriving.values()]
    choice = native.local_tail(len(catalog.capacity_costs.indices), size, first, stages,
                               [home[r] for r in ranks], bytes(missing), routes, local_count, fixed, tapered)
    if choice is None:
        return parent, checked, None
    moves, times = choice
    predicted = sum(checked['round_times_us'][:first]) + sum(times)
    if predicted >= checked['total_time_us'] * (.975 if tapered else .992):
        return parent, checked, None
    packets = [defaultdict(list) for _ in times]
    for t, path, owner in moves:
        packets[t][path].append(labels[owner])
    schedule = rows[:first]
    for deadline, packet in zip(times, packets):
        transfers = []
        for index, tokens in packet.items():
            _, latency, serial, path = routes[index][2]
            width = max(1, floor((deadline - latency) / (size * serial) + 1e-10))
            for j in range(0, len(tokens), width):
                transfers.append(dict(src=path[0], dst=path[-1], path=list(path),
                                      action='copy', chunk_ranges=tokens[j:j + width]))
        if transfers:
            schedule.append(dict(round=len(schedule) + 1, transfers=transfers))
    candidate = dict(parent, rounds=schedule, round_count=len(schedule))
    verified = certify(task, graph, candidate)
    if verified['valid'] and abs(verified['total_time_us'] - predicted) < 1e-4:
        return candidate, verified, dict(anchors=len(fixed), first_round=first + 1, tapered=tapered)
    return parent, checked, None


def improve(task, graph, parent, checked, catalog):
    result = _attempt(task, graph, parent, checked, catalog, True)
    if result[2]:
        return result
    return _attempt(task, graph, parent, checked, catalog, False)
