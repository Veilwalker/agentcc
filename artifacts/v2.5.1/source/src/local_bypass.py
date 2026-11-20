"""Spread repeated transfers between a direct link and independent access ports."""
from collections import defaultdict
from math import floor

from .capacity_backend import native
from .replay import certify


def improve(task, graph, plan, checked, catalog):
    if native is None or task['operator'] != 'all-reduce' or plan['split'] <= 1:
        return plan, checked, []
    ranks = graph.ranks
    # Only rank networks offering multiple direct and switched neighbours
    # can spread same-pair traffic across the two kinds of path.
    rank_set = set(ranks)
    if not any(sum(peer in rank_set for peer, _, _ in graph.adjacency[r]) >= 3
               and len(catalog.outgoing[r]) >= 2 for r in ranks):
        return plan, checked, []
    atom = task['data_size']['per_rank_bytes'] // plan['split']
    costs = catalog.capacity_costs
    ports_in = {r: {p for p, _ in catalog.incoming[r]} for r in ranks}
    paths = {}
    rounds = list(plan['rounds'])
    predicted = checked['total_time_us']
    changed = []
    for i, (row, budget) in enumerate(zip(rounds, checked['round_times_us'])):
        if len(row['transfers']) < len(ranks) // 2:
            continue
        movements = defaultdict(list)
        for move in row['transfers']:
            movements[move['src'], move['dst'], move['action']].extend(move['chunk_ranges'])
        if not any(len(tokens) > 1 and (a, b) in graph.edges for (a, b, _), tokens in movements.items()):
            continue
        jobs, items = [], []
        for (a, b, action), tokens in movements.items():
            pair = a, b
            if pair not in paths:
                direct = (costs.compile(pair),) if pair in graph.edges else ()
                via = tuple(costs.compile((a, p, b)) for p, _ in catalog.outgoing[a] if p in ports_in[b])
                paths[pair] = direct, via
            direct, via = paths[pair]
            here = tuple(p for p in via if p[1] + atom * p[2] < budget - 1e-6)
            if direct and here:
                # Reserve one direct atom, then let parallel ports carry
                # the repeated atoms with their own edge and pool budgets.
                take = 1 if direct[0][1] + atom * direct[0][2] < budget - 1e-6 else 0
                if take:
                    jobs.append((direct, 1))
                    items.append((direct, action, tokens[:1]))
                tokens = tokens[take:]
            elif direct:
                here = direct
            if not tokens:
                continue
            if not here:
                break
            jobs.append((here, len(tokens)))
            items.append((here, action, tokens))
        else:
            choices, _, time_us, _ = native.optimize_groups(len(costs.indices), jobs, 2, costs.round_cache, 3, atom, True)
            if time_us >= budget * .995:
                continue
            buckets = defaultdict(list)
            picks = iter(choices)
            for options, action, tokens in items:
                n = min(3, len(options), len(tokens))
                for j in range(n):
                    path = options[next(picks)][3]
                    buckets[path, action].extend(tokens[len(tokens) * j // n:len(tokens) * (j + 1) // n])
            transfers = []
            for (path, action), tokens in buckets.items():
                _, delay, rate, _ = costs.paths[path]
                width = max(1, floor((time_us - delay) / (atom * rate) + 1e-9))
                for j in range(0, len(tokens), width):
                    transfers.append(dict(src=path[0], dst=path[-1], path=list(path),
                                          action=action, chunk_ranges=tokens[j:j + width]))
            rounds[i] = dict(row, transfers=transfers)
            predicted += time_us - budget
            changed.append(i + 1)
    if changed and predicted < checked['total_time_us'] * .997:
        candidate = dict(plan, rounds=rounds)
        report = certify(task, graph, candidate)
        if report['valid'] and abs(report['total_time_us'] - predicted) < max(1e-4, predicted * 1e-9):
            return candidate, report, changed
    return plan, checked, []
