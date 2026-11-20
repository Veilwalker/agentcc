"""Collapse two local reduction or broadcast stages across shared access ports."""
from collections import Counter, defaultdict
from math import floor

from .capacity_backend import native
from .replay import certify


def _direct_groups(waves, split, reducing):
    parents = [{} for _ in range(split)]
    action = 'reduce' if reducing else 'copy'
    for wave in waves:
        for move in wave['transfers']:
            if move['action'] != action or len(move['path']) > 3:
                return None
            u, v = move['src'], move['dst']
            if not reducing:
                u, v = v, u
            for label in move['chunk_ranges']:
                slot = int(label.rsplit('_c', 1)[1] if reducing else label[1:-4])
                if u in parents[slot]:
                    return None
                parents[slot][u] = v
    grouped = defaultdict(list)
    for slot, tree in enumerate(parents):
        for child, receiver in tree.items():
            root = tree.get(receiver, receiver)
            if root in tree:
                return None
            pair = (child, root) if reducing else (root, child)
            grouped[pair].append(slot)
    return grouped



def _coalesce_cycles(assignments):
    # If two equal halves of a movement use parallel switches, an even cycle
    # in the sender/receiver bipartite graph can move every pair fully onto
    # one switch. Every physical edge retains exactly its previous byte load.
    networks = defaultdict(list)
    for i, (paths, slots) in enumerate(assignments):
        if (len(paths) == 2 and len(slots) % 2 == 0 and paths[0] != paths[1]
                and all(len(p) == 3 for p in paths)):
            a, b = sorted((paths[0][1], paths[1][1]))
            networks[a, b, len(slots)].append(i)
    for (first, second, _), subset in networks.items():
        links = defaultdict(list)
        for i in subset:
            path = assignments[i][0][0]
            sender, receiver = (0, path[0]), (1, path[-1])
            links[sender].append((receiver, i))
            links[receiver].append((sender, i))
        visited = set()
        for start in links:
            vertices, edges, position = [start], [], {start: 0}
            while vertices:
                adjacent = links[vertices[-1]]
                while adjacent and adjacent[-1][1] in visited:
                    adjacent.pop()
                if not adjacent:
                    del position[vertices.pop()]
                    if edges:
                        edges.pop()
                    continue
                peer, edge = adjacent.pop()
                visited.add(edge)
                if peer not in position:
                    position[peer] = len(vertices)
                    vertices.append(peer)
                    edges.append(edge)
                    continue
                offset = position[peer]
                cycle = edges[offset:] + [edge]
                for j, i in enumerate(cycle):
                    paths, slots = assignments[i]
                    switch = second if j % 2 else first
                    assignments[i] = ([next(p for p in paths if p[1] == switch)], slots)
                for vertex in vertices[offset + 1:]:
                    del position[vertex]
                del vertices[offset + 1:]
                del edges[offset:]

def _combine(graph, catalog, waves, split, size, reducing, threshold):
    grouped = _direct_groups(waves, split, reducing)
    if not grouped:
        return None
    costs = catalog.capacity_costs
    groups = []
    for (sender, receiver), slots in grouped.items():
        paths = [(sender, receiver)] if (sender, receiver) in graph.edges else []
        paths.extend((sender, switch, receiver) for switch, _ in catalog.outgoing[sender]
                     if (switch, receiver) in graph.edges)
        if not paths:
            return None
        groups.append((tuple(costs.compile(path) for path in paths), len(slots)))
    choices, _, duration, _ = native.optimize_groups(
        len(costs.indices), groups, 2, costs.round_cache, 3, size, True)
    if duration >= threshold:
        return None
    choices = iter(choices)

    assignments = []
    for (options, count), slots in zip(groups, grouped.values()):
        lanes = min(3, len(options), count)
        paths = [options[next(choices)][3] for _ in range(lanes)]
        assignments.append((paths, slots))
    _coalesce_cycles(assignments)
    batches = defaultdict(list)
    for paths, slots in assignments:
        lanes, count = len(paths), len(slots)
        for lane, path in enumerate(paths):
            batches[path].extend(slots[count * lane // lanes:count * (lane + 1) // lanes])

    action = 'reduce' if reducing else 'copy'
    transfers = []
    for path, slots in batches.items():
        _, latency, inverse, _ = costs.paths[path]
        # One packet also pays its path serialization and switch latency.
        width = max(1, floor((duration - latency) / (size * inverse) + 1e-10))
        labels = [f'{path[0]}_c{s}' if reducing else f'c{s}_sum' for s in slots]
        for start in range(0, len(labels), width):
            transfers.append(dict(src=path[0], dst=path[-1], action=action,
                                  path=list(path), chunk_ranges=labels[start:start + width]))
    return dict(transfers=transfers), duration


def improve(task, graph, parent, checked, catalog):
    rows = parent['rounds']
    split = parent['split']
    if task['operator'] != 'all-reduce' or native is None or len(rows) < 4 or split > 64:
        return parent, checked, None
    # A local star with two or more access ports can use them concurrently.
    # Bound the trial to small local domains and a modest number of path options.
    groups = Counter(catalog.starts.values())
    if not groups or max(groups.values()) > 32 or not all(2 <= len(p) <= 4 for p in groups):
        return parent, checked, None
    size = task['data_size']['per_rank_bytes'] // split
    result = list(rows)
    times = list(checked['round_times_us'])
    merged = []
    for index, reducing in ((0, True), (len(rows) - 2, False)):
        original = times[index] + times[index + 1]
        replacement = _combine(graph, catalog, rows[index:index + 2], split, size, reducing,
                               original - .001 * checked['total_time_us'])
        if replacement:
            wave, duration = replacement
            result[index:index + 2] = [wave, None]
            times[index:index + 2] = [duration, 0.]
            merged.append(index + 1)
    if merged:
        waves = [dict(wave, round=i + 1) for i, wave in enumerate(r for r in result if r)]
        candidate = dict(parent, rounds=waves, round_count=len(waves))
        verified = certify(task, graph, candidate)
        if (verified['valid'] and verified['total_time_us'] < checked['total_time_us']
                and abs(verified['total_time_us'] - sum(times)) < 1e-4):
            return candidate, verified, merged
    return parent, checked, None
