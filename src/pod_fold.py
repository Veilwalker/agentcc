"""Fold local partials into representatives placed on usable fabric gateways."""
from collections import Counter, defaultdict
from math import isqrt

from .capacity_backend import native
from .collectives import _initial
from .group_sketch import access_groups
from .reduction_parents import emit_batches, regenerate
from .replay import certify


def improve(task, graph, plan, checked, catalog, allowed):
    if (native is None or task['operator'] != 'all-reduce' or task['max_nonempty_rounds'] < 6
            or min(map(len, catalog.outgoing.values()), default=0) < 3):
        return plan, checked, None
    allowed = [s for s in allowed if 1 < s <= 32]
    if not allowed:
        return plan, checked, None
    groups = access_groups(graph)
    if len(groups) != 4 or len({len(domain) for domain in groups}) != 1:
        return plan, checked, None
    width = len(groups[0])
    if not 16 <= width <= 96:
        return plan, checked, None
    direct = any((u, v) in graph.edges for u in groups[0] for v in groups[0] if u != v)
    if not direct and len(plan['rounds']) < 7:
        return plan, checked, None
    split = max(allowed)
    h = min(width, 2 * isqrt(width))
    if width % h:
        return plan, checked, None
    atom = task['data_size']['per_rank_bytes'] // split
    costs = catalog.capacity_costs
    cores = {p: {q for q, _ in peers if (q, p) in graph.edges} for p, peers in catalog.adjacency.items()}
    ports = {r: {p for p, _ in catalog.outgoing[r] if (p, r) in graph.edges} for r in graph.ranks}
    exits = {r: tuple(sorted(p for p in ports[r] if cores[p])) for r in graph.ranks}
    gateway = {}
    for r in graph.ranks:
        options = [(1 / graph.edges[r, p] + 1 / graph.edges[p, q], p) for p in exits[r] for q in cores[p]]
        gateway[r] = min(options, default=(float('inf'), None))
    if any(not options for options in exits.values()) and not any(all(exits[r] for r in group) for group in groups):
        return plan, checked, None
    blocks = []
    for domain in groups:
        # A true direct clique can be grouped with adjacent independent cliques.
        # Without direct adjacency, ranks sharing a gateway form natural blocks.
        buckets = defaultdict(list)
        for r in domain:
            if direct:
                clique = frozenset(v for v, _, _ in graph.adjacency[r] if v in domain and (v, r) in graph.edges)
                # Overlapping incomplete cliques are still legal block orderings.
                key = clique | {r}
            else:
                key = exits[r]
            buckets[key].append(r)
        ordered = [r for bucket in buckets.values() for r in bucket]
        blocks.append([ordered[i:i + h] for i in range(0, width, h)])

    stages = [defaultdict(list) for _ in range(6)]
    used = Counter()
    for slot in range(split):
        hubs = []
        for d, subgroups in enumerate(blocks):
            reps = [sub[(slot + (j + 3) * d) % h] for j, sub in enumerate(subgroups)]

            def score(j):
                r = reps[j]
                penalty = atom * 1e6 * 2 * gateway[r][0]
                return (penalty + .65 * used[r] + .02 * used[d, exits[r][:1]], (j - slot - d) % len(reps))
            hub = reps[min(range(len(reps)), key=score)]
            if not exits[hub]:
                return plan, checked, None
            used[hub] += 1
            used[d, exits[hub][:1]] += 1
            hubs.append(hub)
            for group, rep in zip(subgroups, reps):
                for r in group:
                    if r != rep:
                        stages[0][r, rep, False, direct].append(slot)
                        stages[5][rep, r, True, direct].append(slot)
                if rep != hub:
                    stages[1][rep, hub, False, False].append(slot)
                    stages[4][hub, rep, True, False].append(slot)
        root = (slot + slot // max(1, split // 4)) % 4
        if direct:
            for i in range(4):
                if i != root:
                    stages[2][hubs[i], hubs[root], False, False].append(slot)
                    stages[3][hubs[root], hubs[i], True, False].append(slot)
        else:
            order = hubs[root:] + hubs[:root]
            for i in range(4):
                stages[2][order[i], order[i ^ 1], False, False].append(slot)
                stages[3][order[i], order[i ^ 2], False, False].append(slot)

    compiled = {}
    recipe = []
    predicted = 0.
    for stage in stages:
        if not stage:
            continue
        jobs, items = [], []
        for (a, b, copy, prefer_direct), slots in stage.items():
            key = a, b, prefer_direct
            if key not in compiled:
                paths = [(a, b)] if (a, b) in graph.edges else []
                if not paths or not prefer_direct:
                    paths.extend((a, p, b) for p in sorted(ports[a] & ports[b]))
                if not paths:
                    spines = defaultdict(list)
                    for p in exits[a]:
                        for q in exits[b]:
                            if p != q:
                                for spine in sorted(cores[p] & cores[q]):
                                    path = (a, p, spine, q, b)
                                    if len(set(path)) == len(path):
                                        serial = sum(1 / graph.edges[u, v] for u, v in zip(path, path[1:]))
                                        spines[spine].append((serial, path))
                    paths = [path for opts in spines.values() for _, path in sorted(opts)[:2]]
                if not paths:
                    paths = catalog.options(a, b)
                if not paths or len(paths) > 128:
                    return plan, checked, None
                compiled[key] = tuple(costs.compile(p) for p in paths)
            options = compiled[key]
            jobs.append((options, len(slots)))
            items.append((options, copy, slots))
        choices, _, duration, _ = native.optimize_groups(len(costs.indices), jobs, 2, costs.round_cache, 3, atom, True)
        predicted += duration
        if predicted >= checked['total_time_us'] * .993:
            return plan, checked, None
        recipe.append((items, choices, duration))
    rounds = []
    for items, choices, duration in recipe:
        grouped = defaultdict(list)
        picks = iter(choices)
        for opts, copy, slots in items:
            count = min(3, len(opts), len(slots))
            for j in range(count):
                path = opts[next(picks)][3]
                grouped[path, copy].extend(slots[len(slots) * j // count:len(slots) * (j + 1) // count])
        moves = emit_batches(grouped, duration, atom, costs)
        rounds.append(dict(round=len(rounds) + 1, transfers=moves))
    answer = _initial(task, graph.ranks, split)
    answer.update(rounds=regenerate(rounds, graph.ranks, split), round_count=len(rounds))
    report = certify(task, graph, answer)
    if report['valid'] and abs(report['total_time_us'] - predicted) < max(1e-4, predicted * 1e-9):
        return answer, report, dict(domains=len(groups), split=split, fold='star' if direct else 'double')
    return plan, checked, None
