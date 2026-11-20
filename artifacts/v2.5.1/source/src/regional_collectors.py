"""Jointly select the cross-domain parent and its physical route for each shard."""
from collections import defaultdict
from math import isqrt

from .capacity_backend import native
from .collectives import _initial
from .fabric_routes import RouteCatalog
from .reduction_parents import emit_batches, regenerate
from .replay import certify


def improve(task, graph, groups, split, catalog, upper_us, *, exchange):
    if not groups or not groups[0]:
        return None
    count, width = len(groups), len(groups[0])
    if (native is None or split < width or split % width or not 4 <= count <= 24
            or any(len(group) != width for group in groups)):
        return None
    streams = min(2, split // width)
    height = 3 if exchange else 4
    if 2 + height + streams - 1 > task['max_nonempty_rounds']:
        return None
    atom = task['data_size']['per_rank_bytes'] // split
    if not catalog.branch_ports or not catalog.deep_exits:
        catalog = RouteCatalog(graph, atom, branch_ports=True, deep_exits=True)
    costs = catalog.capacity_costs
    routes = {}

    def paths(u, v):
        if (u, v) not in routes:
            routes[u, v] = tuple(costs.compile(p) for p in catalog.options(u, v))
        return routes[u, v]

    # Approximate inter-domain distance for choosing complementary collectors.
    # Later pricing uses the exact endpoint, directed edges and every shared pool.
    if exchange:
        edge_count = len(graph.edges)

        def distance(u, v):
            return min(p[1] + atom * (p[2] + 2 * sum(rate for k, rate in p[0] if k >= edge_count))
                       for p in paths(u, v))
        forward = [[distance(a[0], b[0]) if ai != bi else 0.
                    for bi, b in enumerate(groups)] for ai, a in enumerate(groups)]
        both = [[forward[i][j] + forward[j][i] for j in range(count)] for i in range(count)]
    collector_count = 2 if exchange else max(2, isqrt(count))
    stages = [defaultdict(list) for _ in range(2 + height + streams - 1)]
    for slot in range(split):
        start = (slot // count + slot) % streams
        root = (slot + (slot // count) * max(1, count // 3)) % count
        reps = [group[(slot + 3 * d) % width] for d, group in enumerate(groups)]
        others = [(root + j) % count for j in range(1, count)]
        if exchange:
            def cover_cost(c):
                return sum(min(both[i][root], both[i][c]) for i in range(count)) + .65 * both[root][c]
            centers = [root, min(others, key=cover_cost)]
        else:
            stride_count = collector_count - 1
            centers = [root] + [others[j * (count - 1) // stride_count] for j in range(stride_count)]
        for d, group in enumerate(groups):
            rep = reps[d]
            for member in group:
                if member != rep:
                    stages[start][((member, rep),), False].append(slot)
                    stages[start + height + 1][((rep, member),), True].append(slot)
            if d not in centers:
                stages[start + 1][tuple((rep, reps[c]) for c in centers), False].append(slot)
                stages[start + height][tuple((reps[c], rep) for c in centers), True].append(slot)
        for c in centers[1:]:
            stages[start + 2][((reps[c], reps[root]),), False].append(slot)
            stages[start + 2 + int(not exchange)][((reps[root], reps[c]),), not exchange].append(slot)
    recipe = []
    elapsed_us = 0.
    for stage in stages:
        if not stage:
            continue
        jobs, records = [], []
        for (pairs, copy), slots in stage.items():
            options = tuple(p for u, v in pairs for p in paths(u, v))
            if not 1 <= len(options) <= 128:
                return None
            # A different parent can carry each shard. One local path may serve
            # an entire group of shards without changing its single-shard bound.
            if len(options) == 1:
                jobs.append((options, len(slots) * atom, atom))
                records.append((options, copy, slots))
            else:
                for slot in slots:
                    jobs.append((options, atom, atom))
                    records.append((options, copy, (slot,)))
        selected, _, duration, _ = native.optimize_round(len(costs.indices), jobs, 2, costs.round_cache)
        elapsed_us += duration
        if elapsed_us >= upper_us * .995:
            return None
        batch = defaultdict(list)
        for index, (choices, copy, slots) in zip(selected, records):
            batch[choices[index][3], copy].extend(slots)
        recipe.append((duration, batch))
    rows = []
    for duration, batch in recipe:
        moves = emit_batches(batch, duration, atom, costs)
        rows.append(dict(round=len(rows) + 1, transfers=moves))
    candidate = _initial(task, graph.ranks, split)
    candidate.update(rounds=regenerate(rows, graph.ranks, split), round_count=len(rows))
    checked = certify(task, graph, candidate)
    if checked['valid'] and abs(elapsed_us - checked['total_time_us']) <= max(.0001, elapsed_us * 1e-9):
        return candidate, checked, dict(regional_collectors=collector_count, exchange=exchange, priced_us=elapsed_us)
    return None
