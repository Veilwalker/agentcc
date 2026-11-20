"""Overlap AllGather rounds by delaying each original atom's entire dependency chain."""
from collections import defaultdict
import math

from .fabric_routes import CapacityAssignment


def pipeline(candidate, parent_cost, space, max_rounds):
    # These sketches address fabric contention when only a few switches per
    # endpoint are available. Bound both round growth and response construction.
    n = len(space.order)
    spare = max_rounds - candidate['round_count']
    if n < 32 or spare < 1:
        return candidate, parent_cost
    lanes = min(8, 1 << ((spare + 1).bit_length() - 1))
    catalog = space.catalog
    if max(len(catalog.outgoing[r]) for r in space.order) > 2:
        if not catalog.shared_spine_plane or catalog.separate_spine_planes:
            return candidate, parent_cost
        lanes = 2
    original = candidate['rounds']
    if sum(len(r['transfers']) for r in original) * lanes > 55000:
        return candidate, parent_cost

    bucket = {}
    for index, rank in enumerate(space.graph.ranks):
        for slot in range(space.split):
            pos = index + slot
            bucket[f'{rank}_c{slot}'] = (pos ^ (pos >> 3)) % lanes
    batches = [defaultdict(list) for _ in range(len(original) + lanes - 1)]
    partitions = {}
    for i, row in enumerate(original):
        for transfer in row['transfers']:
            path = tuple(transfer['path'])
            payload = tuple(transfer['chunk_ranges'])
            if payload not in partitions:
                groups = defaultdict(list)
                for atom in payload:
                    groups[bucket[atom]].append(atom)
                partitions[payload] = tuple(groups.items())
            for offset, atoms in partitions[payload]:
                batches[i + offset][path].extend(atoms)

    durations, options = [], space.costs.paths
    for batch in batches:
        load = CapacityAssignment(space.costs)
        for path, atoms in batch.items():
            # Price a one-atom packet on each path, then coalesce up to that
            # same round ceiling. Aggregate link and shared-pool bytes are fixed.
            option = options.get(path) or space.costs.compile(path)
            load.choose((option,), len(atoms) * space.size, space.size)
        durations.append(load.time_us)
    total = sum(durations)
    if total >= parent_cost * 0.98:
        return candidate, parent_cost

    rounds = []
    for batch, ceiling in zip(batches, durations):
        if not batch:
            continue
        transfers = []
        for path, atoms in batch.items():
            _, overhead, serial, _ = options[path]
            step = max(1, math.floor((ceiling - overhead) / (serial * space.size) + 1e-10))
            for first in range(0, len(atoms), step):
                transfers.append({'action': 'copy', 'src': path[0], 'dst': path[-1],
                                  'chunk_ranges': atoms[first:first + step], 'path': list(path)})
        rounds.append({'round': len(rounds) + 1, 'transfers': transfers})
    return {**candidate, 'round_count': len(rounds), 'rounds': rounds}, total
