"""Move validated transfers between dependency-safe rounds at fixed split."""
from collections import defaultdict
import math

from .capacity_backend import native


def price(task, graph, incumbent, catalog, *, direct=False, bands=4):
    """Build the producer/consumer windows for copies and reduction slots."""
    n, split = len(graph.ranks), incumbent['split']
    rows = incumbent['rounds']
    slots = split if task['operator'] == 'all-reduce' else n * split
    limit = min(task['max_nonempty_rounds'], len(rows) + 2)
    costs = catalog.capacity_costs
    if (native is None or n < 8 or n * slots > 300000 or not rows or limit > 128
            or len(costs.indices) * limit > 4000000
            or sum(len(row['transfers']) for row in rows) > 40000):
        return None
    ar = task['operator'] == 'all-reduce'
    atom_ids = {} if ar else {a: i for i, a in enumerate(incumbent['chunks'])}
    token_cache, path_ids, routes, packets, original = {}, {}, [], [], []
    for i, row in enumerate(rows):
        grouped = {}
        for item in row['transfers']:
            pair = item['src'], item['dst']
            path = pair if direct and pair in graph.edges else tuple(item['path'])
            grouped.setdefault((path, item['action']), []).extend(item['chunk_ranges'])
        for (path, action), atoms in grouped.items():
            if path not in path_ids:
                path_ids[path] = len(routes)
                routes.append(costs.compile(path))
            tokens = tuple(atoms)
            ids = token_cache.get(tokens)
            if ids is None:
                if ar:
                    for a in tokens:
                        if a not in atom_ids:
                            atom_ids[a] = (int(a[1:-4]) if a.endswith('_sum')
                                           else int(a.rsplit('_c', 1)[1]))
                ids = tuple(atom_ids[a] for a in atoms)
                token_cache[tokens] = ids
            p = path_ids[path]
            packets.append((i, p, catalog.rank_index[path[0]], catalog.rank_index[path[-1]], ids))
            original.append((p, action, atoms, ids))
    size = task['data_size']['per_rank_bytes'] // split
    try:
        assignment, ceilings = native.retime_transfers(n, slots, len(costs.indices), len(rows),
            limit, bands, int(ar), size, routes, packets)
    except ValueError:
        return None  # Native incidence cap for optional edits.
    return sum(ceilings), (original, routes, size, bands, assignment, ceilings)


def materialize(incumbent, plan):
    original, routes, size, bands, assignment, ceilings = plan
    batches = [defaultdict(list) for _ in ceilings]
    bucket = {}
    for i, (route, action, atoms, ids) in enumerate(original):
        key = route, action
        # Many transfers reuse an immutable multicast payload. Partition it
        # once, then append an entire lane to its chosen destination round.
        lanes = bucket.get(ids)
        if lanes is None:
            lanes = defaultdict(list)
            for j, atom in enumerate(ids):
                lanes[(atom ^ (atom >> 3)) % bands].append(j)
            bucket[ids] = lanes
        for b, indexes in lanes.items():
            batch = batches[assignment[i*bands+b]][key]
            batch.extend(atoms[j] for j in indexes)
    rows = []
    for batch, ceiling in zip(batches, ceilings):
        if not batch:
            continue
        transfers = []
        for (route, action), atoms in batch.items():
            _, latency, serial, path = routes[route]
            width = max(1, math.floor((ceiling-latency)/(size*serial)+1e-10))
            for i in range(0,len(atoms),width):
                transfers.append(dict(src=path[0],dst=path[-1],path=list(path),
                                      action=action,chunk_ranges=atoms[i:i+width]))
        rows.append(dict(round=len(rows)+1, transfers=transfers))
    return {**incumbent, 'rounds': rows, 'round_count': len(rows)}
