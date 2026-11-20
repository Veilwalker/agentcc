"""Reprice small reduction rounds under a ceiling on per-atom path time."""
from collections import defaultdict
from math import floor

from .capacity_backend import native
from .replay import certify


def improve(task, graph, solution, checked, catalog):
    original = checked['round_times_us']
    rounds = solution['rounds']
    if (native is None or task['operator'] != 'all-reduce' or len(graph.ranks) < 64
            or len(original) != len(rounds)):
        return solution, checked, []
    updated = list(rounds)
    prices = list(original)
    size = task['data_size']['per_rank_bytes'] // solution['split']
    notes = []
    top = max(original, default=0)
    for i, (row, ceiling) in enumerate(zip(rounds, original)):
        if ceiling < 0.85*top or not 1 <= len(row['transfers']) <= 256:
            continue
        movements = defaultdict(list)
        for move in row['transfers']:
            movements[move['src'], move['dst'], move['action']].extend(move['chunk_ranges'])
        paths = {}
        try:
            for src, dst, _ in movements:
                key = src, dst
                if key not in catalog.compiled_options:
                    catalog.compiled_options[key] = tuple(catalog.capacity_costs.compile(p)
                                                          for p in catalog.options(src, dst))
                paths[key] = catalog.compiled_options[key]
            best, best_rows = ceiling, None
            for ratio in (.90, .98):
                jobs = []
                for (src, dst, action), atoms in movements.items():
                    options = tuple(p for p in paths[src, dst] if p[1]+size*p[2] <= ratio*ceiling)
                    if not options:
                        break
                    lanes = min(len(atoms), 2*len(options))
                    for j in range(lanes):
                        jobs.append((options, action, atoms[len(atoms)*j//lanes:len(atoms)*(j+1)//lanes]))
                else:
                    jobs.sort(key=lambda job: (len(job[0]), -len(job[2])))
                    specs = [(opts, len(atoms)*size, size) for opts, _, atoms in jobs]
                    choices, _, us, _ = native.optimize_round(len(catalog.capacity_costs.indices), specs, 2)
                    if us < best - 1e-6:
                        best, best_rows = us, (jobs, choices)
            if best_rows is None:
                continue
            batch = defaultdict(list)
            jobs, choices = best_rows
            for (options, action, atoms), choice in zip(jobs, choices):
                batch[options[choice][3], action].extend(atoms)
            transfers = []
            for (path, action), atoms in batch.items():
                _, overhead, inverse, _ = catalog.capacity_costs.paths[path]
                width = max(1, floor((best-overhead)/(size*inverse)+1e-10))
                for j in range(0, len(atoms), width):
                    transfers.append(dict(src=path[0], dst=path[-1], path=list(path), action=action,
                                          chunk_ranges=atoms[j:j+width]))
            updated[i] = dict(row, transfers=transfers)
            prices[i] = best
            notes.append(i+1)
        except ValueError:
            continue
    predicted = sum(prices)
    if predicted < checked['total_time_us']*.997:
        candidate = dict(solution, rounds=updated)
        result = certify(task, graph, candidate)
        if result['valid'] and abs(result['total_time_us'] - predicted) < max(1e-4, predicted*1e-9):
            return candidate, result, notes
    return solution, checked, []
