"""Reassign whole atoms to legal paths, preserving round-boundary ownership."""
from collections import defaultdict
from math import floor

from . import reduction_parents
from .capacity_backend import native
from .replay import certify


def graph_data(graph, size):
    nodes = [*graph.ranks, *sorted(graph.switches)]
    index = {rank: i for i, rank in enumerate(nodes)}
    pairs = list(graph.edges)
    pools = {pool: len(pairs) + i for i, pool in enumerate(graph.resources)}
    rates = [size * 1e6 / c for c in (*graph.edges.values(), *graph.resources.values())]
    edges = [(index[u], index[v], (i, *(pools[p] for p in graph.edge_resources[u, v])))
             for i, (u, v) in enumerate(pairs)]
    return edges, rates, pairs, len(nodes)


def start_states(rows, initial, rank_bits, gather):
    held = {t: rank_bits[r] for r, tokens in initial.items() for t in tokens} if gather else defaultdict(int)
    snapshots = []
    for row in rows:
        moves = row['transfers']
        if not gather:
            # At the handoff from reduction to broadcast, every certified
            # complete sender in this round is also an eligible relay.
            for move in moves:
                if move['action'] == 'copy':
                    bit = rank_bits[move['src']]
                    for token in move['chunk_ranges']:
                        held[token] |= bit
        snapshots.append(held.copy())
        for move in moves:
            if move['action'] == 'copy':
                bit = rank_bits[move['dst']]
                for token in move['chunk_ranges']:
                    held[token] |= bit
    return snapshots


def _gather_round(moves, held, members, encoded, edge_ids, pairs, costs, size, prefix, suffix):
    # Passing a whole batch lets the native optimizer maintain one flow order
    # and hand back one payload per final route, including multi-atom flows.
    holders = {token: members(mask) for token,mask in held.items()}
    indices, jobs = {}, []
    for move in moves:
        path = tuple(move['path'])
        pid = indices.setdefault(path,len(indices))
        if path not in encoded:
            encoded[path] = tuple(edge_ids[e] for e in zip(path,path[1:]))
        jobs.append((pid,move['chunk_ranges']))
    result = native.adaptive_routes(*prefix,[encoded[p] for p in indices],jobs,*suffix,holders)
    if result is None:
        return None
    ceiling, batches = result
    emitted = []
    for ids,tokens in batches:
        path = (pairs[ids[0]][0], *(pairs[i][1] for i in ids))
        _,latency,inverse,_ = costs.compile(path)
        width = max(1, floor((ceiling-latency)/(size*inverse)+1e-10))
        for i in range(0,len(tokens),width):
            emitted.append(dict(src=path[0],dst=path[-1],path=list(path),action='copy',
                                chunk_ranges=tokens[i:i+width]))
    return ceiling,emitted


def improve(task, graph, solution, report, catalog, *, passes=5, round_fraction=.12,
            focused_round=None, resource=None):
    ranks = graph.ranks
    size = task['data_size']['per_rank_bytes'] // solution['split']
    edges, rates, pairs, nodes = graph_data(graph, size)
    rank_ids = {rank: i for i, rank in enumerate(ranks)}
    rank_bits = {rank: 1 << i for rank, i in rank_ids.items()}
    edge_ids = {pair: i for i, pair in enumerate(pairs)}
    encoded = {}
    gather = task['operator'] == 'all-gather'
    slots, parents = ({}, {}) if gather else reduction_parents.eligibility(solution, rank_ids)
    rows = list(solution['rounds'])
    times = list(report['round_times_us'])
    if len(rows) != len(times) or not times:
        return solution, report, []
    snapshots = start_states(rows if focused_round is None else rows[:focused_round+1], solution['initial_possession'], rank_bits, gather)
    owners = {1 << i: (i,) for i in range(len(ranks))}

    def members(mask):
        if mask not in owners:
            remaining, indices = mask, []
            while remaining:
                bit = remaining & -remaining
                indices.append(bit.bit_length() - 1)
                remaining -= bit
            owners[mask] = tuple(indices)
        return owners[mask]

    peak = max(times)
    order = sorted(range(len(rows)), key=times.__getitem__, reverse=True)
    focus_index = -1
    if focused_round is not None:
        order = [focused_round]
        focus_index = (edge_ids[resource] if isinstance(resource, tuple) else
                       len(pairs) + list(graph.resources).index(resource))
    changed = []
    large_gather = gather and len(ranks) * (len(ranks) - 1) * solution['split'] >= 110000
    compact_gather = large_gather and len(ranks)*(len(ranks)-1)*solution['split'] <= 3000000
    for probe, step in enumerate(order):
        if times[step] < peak * round_fraction or (large_gather and probe >= 2 and not changed):
            break
        if compact_gather and len(rows[step]['transfers']) <= 300000:
            result = _gather_round(rows[step]['transfers'],snapshots[step],members,encoded,
                edge_ids,pairs,catalog.capacity_costs,size,
                (len(ranks),nodes,edges,rates,graph.startup_us,graph.switch_overhead_us),
                (passes,times[step],focus_index))
            if result is not None:
                times[step],emitted = result
                rows[step] = dict(round=step+1,transfers=emitted)
                changed.append(step+1)
            continue
        flows, indices = defaultdict(list), {}
        held = snapshots[step]
        for move in rows[step]['transfers']:
            path = tuple(move['path'])
            pid = indices.setdefault(path, len(indices))
            if path not in encoded:
                encoded[path] = tuple(edge_ids[e] for e in zip(path, path[1:]))
            if move['action'] == 'copy':
                for token in move['chunk_ranges']:
                    flows[pid, held[token], 'copy'].append(token)
            else:
                for token in move['chunk_ranges']:
                    targets = parents.get((slots[token], step))
                    key = (pid, targets, 'reparent') if targets else (pid, rank_bits[move['src']], 'reduce')
                    flows[key].append(token)
        args = [(p, len(ts), members(mask), action == 'reparent')
                for (p, mask, action), ts in flows.items()]
        result = native.adaptive_routes(len(ranks), nodes, edges, rates, graph.startup_us,
                                        graph.switch_overhead_us, [encoded[p] for p in indices],
                                        args, passes, times[step], focus_index)
        if result is None:
            continue
        ceiling, allocations = result
        batches, physical = defaultdict(list), {}
        for ((_, _, action), tokens), portions in zip(flows.items(), allocations):
            offset = 0
            action = 'reduce' if action == 'reparent' else action
            for count, ids in portions:
                if ids not in physical:
                    physical[ids] = (pairs[ids[0]][0], *(pairs[i][1] for i in ids))
                batches[physical[ids], action].extend(tokens[offset:offset + count])
                offset += count
            if offset != len(tokens):
                raise ValueError('flow allocation differs from input')
        emitted = []
        for (path, action), tokens in batches.items():
            _, latency, inverse, _ = catalog.capacity_costs.compile(path)
            width = max(1, floor((ceiling - latency) / (size * inverse) + 1e-10))
            for i in range(0, len(tokens), width):
                emitted.append(dict(src=path[0], dst=path[-1], path=list(path), action=action,
                                    chunk_ranges=tokens[i:i + width]))
        rows[step] = dict(round=step + 1, transfers=emitted)
        times[step] = ceiling
        changed.append(step + 1)
    predicted = sum(times)
    if changed and predicted < (.99999 if focused_round is not None else .995) * report['total_time_us']:
        if parents:
            rows = reduction_parents.regenerate(rows, ranks, solution['split'])
        candidate = dict(solution, rounds=rows)
        checked = certify(task, graph, candidate)
        if checked['valid'] and abs(checked['total_time_us'] - predicted) < max(1e-4, predicted * 1e-9):
            return candidate, checked, sorted(changed)
    return solution, report, []
