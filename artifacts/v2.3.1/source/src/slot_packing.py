"""Balance independent reduction slots across the incumbent's physical trees."""
from collections import Counter, defaultdict
import math

from .capacity_backend import native


def price(task, graph, solution, catalog):
    rows, split = solution['rounds'], solution['split']
    if not 2 <= split <= 32 or not 1 <= len(rows) <= 64:
        return None
    # Equal access groups already spread the common trees symmetrically.
    ports = [tuple(sorted(s for s, _ in catalog.outgoing[r])) for r in graph.ranks]
    if max(map(len, ports)) > 2 or len(set(Counter(ports).values())) < 2:
        return None
    size = task['data_size']['per_rank_bytes'] // split
    costs, routes, paths, packets = catalog.capacity_costs, [], {}, []
    slot_by_token = {}
    for r, row in enumerate(rows):
        for transfer in row['transfers']:
            path = tuple(transfer['path'])
            if path not in paths:
                paths[path] = len(routes)
                routes.append(costs.compile(path))
            payload = transfer['chunk_ranges']
            for token in payload:
                if token not in slot_by_token:
                    slot_by_token[token] = (int(token[1:-4]) if token.endswith('_sum')
                                            else int(token.rsplit('_c', 1)[1]))
            packets.append((r, paths[path], tuple(slot_by_token[t] for t in payload)))
    try:
        assignment, times = native.choose_slot_trees(len(rows), len(costs.indices), split,
                                                      1, size, routes, packets)
    except ValueError:
        return None
    return sum(times), (assignment, times, slot_by_token, costs, size)


def materialize(solution, recipe):
    assignment, times, slot_by_token, costs, size = recipe
    copies = defaultdict(list)
    for output_slot, template in enumerate(assignment):
        copies[template].append(output_slot)
    replacements = {}
    rows = []
    for row, ceiling in zip(solution['rounds'], times):
        batches = defaultdict(list)
        for item in row['transfers']:
            payload = batches[tuple(item['path']), item['action']]
            for token in item['chunk_ranges']:
                if token not in replacements:
                    targets = copies[slot_by_token[token]]
                    if token.endswith('_sum'):
                        renamed = [f'c{i}_sum' for i in targets]
                    else:
                        ranks = [atom.rsplit('_c', 1)[0] for atom in token.split('+')]
                        renamed = ['+'.join(f'{rank}_c{i}' for rank in ranks) for i in targets]
                    replacements[token] = renamed
                payload.extend(replacements[token])
        transfers = []
        for (path, action), atoms in batches.items():
            _, latency, serial, _ = costs.paths[path]
            width = max(1, math.floor((ceiling-latency)/(size*serial)+1e-10))
            for start in range(0,len(atoms),width):
                transfers.append(dict(src=path[0], dst=path[-1], path=list(path), action=action,
                                      chunk_ranges=atoms[start:start+width]))
        if transfers:
            rows.append(dict(round=len(rows)+1, transfers=transfers))
    return {**solution, 'rounds': rows, 'round_count':len(rows)}


def wave_price(task, graph, solution, catalog):
    """Offset an entire slot's reduction and broadcast together on narrow fabrics."""
    rounds, split = solution['rounds'], solution['split']
    lanes = min(5, task['max_nonempty_rounds'] - len(rounds) + 1)
    if (not 2 <= split <= 32 or not 2 <= len(rounds) <= 32 or lanes < 2
            or len(graph.ranks) < 64 or any(len(catalog.outgoing[r]) != 1 for r in graph.ranks)):
        return None
    costs = catalog.capacity_costs
    size = task['data_size']['per_rank_bytes']//split
    paths, routes, packets, slots = {}, [], [], {}
    for i, row in enumerate(rounds):
        for item in row['transfers']:
            path = tuple(item['path'])
            if path not in paths:
                paths[path] = len(routes)
                routes.append(costs.compile(path))
            for atom in item['chunk_ranges']:
                if atom not in slots:
                    slots[atom] = int(atom[1:-4]) if atom.endswith('_sum') else int(atom.rsplit('_c', 1)[1])
            packets.append((i, paths[path], tuple(slots[a] for a in item['chunk_ranges'])))
    try:
        delays, times = native.choose_tree_offsets(len(rounds), len(costs.indices), split,
                                                   lanes, size, routes, packets)
    except ValueError:
        return None
    return sum(times), (delays, times, slots, costs, size)


def wave_materialize(solution, recipe):
    delays, times, slots, costs, size = recipe
    waves = [defaultdict(list) for _ in times]
    for i, row in enumerate(solution['rounds']):
        for item in row['transfers']:
            key = tuple(item['path']), item['action']
            for token in item['chunk_ranges']:
                waves[i+delays[slots[token]]][key].append(token)
    rows = []
    for wave, ceiling in zip(waves, times):
        transfers = []
        for (path, action), atoms in wave.items():
            _, overhead, serial, _ = costs.paths[path]
            width = max(1, math.floor((ceiling-overhead)/(size*serial)+1e-10))
            for j in range(0, len(atoms), width):
                transfers.append(dict(src=path[0], dst=path[-1], path=list(path), action=action,
                                      chunk_ranges=atoms[j:j+width]))
        if transfers:
            rows.append(dict(round=len(rows)+1, transfers=transfers))
    return dict(solution, rounds=rows, round_count=len(rows))
