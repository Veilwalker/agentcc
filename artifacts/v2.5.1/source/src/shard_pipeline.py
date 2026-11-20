"""Pipeline equal subshards through a chosen multicast tree.

Shifting subshard j by j rounds preserves every dependency of its parent tree.
Capacity profiles are reused to price all permitted splits before materializing
the best schedule. Each path contributes once per occurrence of a shared pool.
"""
import math
from collections import defaultdict

from .collectives import _initial
from .capacity_backend import native


def _plan(candidate, graph, direct):
    plans = []
    for row in candidate['rounds']:
        grouped = defaultdict(list)
        for move in row['transfers']:
            pair = move['src'], move['dst']
            path = pair if direct and pair in graph.edges else tuple(move['path'])
            grouped[path].extend(move['chunk_ranges'])
        plans.append(grouped)
    return plans


def profiles_for(plans, costs):
    profiles = []
    for grouped in plans:
        loads = defaultdict(float)
        overhead = serial = 0.
        for path, atoms in grouped.items():
            terms, latency, inverse, _ = costs.compile(path)
            overhead = max(overhead, latency)
            serial = max(serial, inverse)
            for resource, rate in terms:
                loads[resource] += len(atoms) * rate
        profiles.append((dict(loads), overhead, serial))
    return profiles


def ready_waves(plans, candidate, costs):
    # Each source tree can progress as soon as its own parent has arrived.
    # Multiple tokens packed in one transfer may have different arrival times.
    available = {r: {a: 0 for a in atoms} for r, atoms in candidate['initial_possession'].items()}
    waves = [defaultdict(list) for _ in plans]
    moved = False
    for i, row in enumerate(plans):
        for path, atoms in row.items():
            source, dest = available[path[0]], available[path[-1]]
            for atom in atoms:
                ready = source[atom]
                waves[ready][path].append(atom)
                dest[atom] = ready+1
                moved |= ready < i
    if moved:
        waves = [wave for wave in waves if wave]
        return waves, profiles_for(waves, costs)
    return None


def price(task, graph, candidate, parent_us, catalog, allowed, *, grouped=None):
    if native is None or task['operator'] != 'all-gather':
        return None
    n, split = len(graph.ranks), candidate['split']
    count = len(candidate['rounds']) if grouped is None else len(grouped)
    # Bound output expansion as well as numeric work. At 256 ranks, four
    # lanes already declare 261120 distinct deliveries in the response.
    factors = [s//split for s in allowed if s % split == 0 and s > split
               and s*n*(n-1) <= 530000 and count+s//split-1 <= task['max_nonempty_rounds']
               and s//split <= 4]
    if not count or count > 64 or n < 8:
        return None
    costs = catalog.capacity_costs
    resources = len(costs.indices)
    if count * resources > 2000000:
        return None
    base_size = task['data_size']['per_rank_bytes']//split
    threshold = parent_us * 0.97
    best = None
    if grouped is None:
        has_direct = any((m['src'],m['dst']) in graph.edges and len(m['path']) > 2
                         for r in candidate['rounds'] for m in r['transfers'])
    else:
        has_direct = any(len(path) > 2 and (path[0],path[-1]) in graph.edges for row in grouped for path in row)

    def get_plans(direct):
        if grouped is None:
            return _plan(candidate, graph, direct)
        if not direct:
            return grouped
        # A direct alternative can merge different physical paths into one
        # pair, so preserve the same insertion and atom order as the wire form.
        result = []
        for row in grouped:
            mapped = defaultdict(list)
            for path, atoms in row.items():
                pair = path[0], path[-1]
                mapped[pair if pair in graph.edges else path].extend(atoms)
            result.append(mapped)
        return result
    narrow = max(map(len, catalog.outgoing.values()), default=0) <= 2
    # Delay whole multicast trees on narrow fabrics. This preserves the
    # original atom count and avoids expanding the SDK response.
    lanes = min(10 if narrow else 6, task['max_nonempty_rounds'] - count + 1)
    tree_plans = None
    if (narrow or has_direct) and lanes >= 2 and 2 <= n*split <= 1024 and (count+lanes-1)*resources <= 2500000:
        tree_plans = plans = get_plans(has_direct)
        atoms = {a: i for i, a in enumerate(candidate['chunks'])}
        routes, path_ids, packets = [], {}, []
        for i, row in enumerate(plans):
            for path, payload in row.items():
                if path not in path_ids:
                    path_ids[path] = len(routes)
                    routes.append(costs.compile(path))
                packets.append((i, path_ids[path], tuple(atoms[a] for a in payload)))
        try:
            offsets, times = native.choose_tree_offsets(count, resources, len(atoms),
                                                       lanes, base_size, routes, packets)
        except ValueError:
            pass  # An admitted schedule can exceed the optional incidence budget.
        else:
            if sum(times) < parent_us * 0.97:
                best = plans, 1, times, dict(zip(atoms, offsets))
                threshold = sum(times) * 0.94
        if best is not None and narrow:
            return best
    for direct in ([False, True] if has_direct else [False]):
        plans = tree_plans if tree_plans is not None and direct == has_direct else get_plans(direct)
        profiles = profiles_for(plans, costs)
        prices = native.pipeline_costs(resources, base_size, profiles, factors)
        for factor, durations in zip(factors, prices):
            projected = sum(durations)
            # Large all-to-all fanouts amplify every additional lane in the SDK
            # response. Require a material gain before multiplying these packets.
            bulky = factor >= 4 and sum(map(len, plans)) * factor > 90000
            ceiling = min(threshold, parent_us * 0.90) if bulky else threshold
            if projected < ceiling:
                best = plans, factor, durations, None
                # Additional subshards enlarge both Replay and the SDK response.
                threshold = projected * 0.96
    # Reuse the same number of physical shards and allow only graphs with
    # several access ports. The local-grid sketch prices its own ready waves.
    if (best is not None and best[1] > 1 and not narrow and count <= 6
            and not catalog.isolated_switches):
        early = ready_waves(best[0], candidate, costs)
        if early is not None:
            waves, profiles = early
            factor = best[1]
            times, = native.pipeline_costs(resources, base_size, profiles, [factor])
            if sum(times) < 0.99*sum(best[2]):
                best = waves, factor, times, None
    return best


def materialize(task, graph, candidate, catalog, plan):
    plans, factor, durations, offsets = plan
    new_split = candidate['split'] * factor
    result = _initial(task, graph.ranks, new_split)
    new_size = task['data_size']['per_rank_bytes'] // new_split
    if (offsets is not None and native is not None and hasattr(native, 'emit_shifted_trees')
            and len(graph.ranks) >= 64):
        try:
            rows = native.emit_shifted_trees(plans, catalog.capacity_costs.paths, offsets, durations, new_size)
        except (TypeError, ValueError, OverflowError, MemoryError):
            pass
        else:
            result.update(rounds=rows, round_count=len(rows))
            return result
    shards = {f'{r}_c{s}': atoms[s*factor:(s+1)*factor]
              for r, atoms in result['initial_possession'].items()
              for s in range(candidate['split'])}
    if (offsets is None and native is not None and hasattr(native, "emit_gather_shards")
            and len(graph.ranks) >= 64 and isinstance(plans, list) and isinstance(durations, list)):
        try:
            emitted = native.emit_gather_shards(
                plans, catalog.capacity_costs.paths, shards, durations, factor, new_size)
        except (TypeError, ValueError, OverflowError, MemoryError):
            pass
        else:
            result['rounds'] = emitted
            result['round_count'] = len(emitted)
            if candidate['split'] == 1 and factor > 1:
                catalog.pipeline_template = emitted, plans
            return result
    shifted = None
    if offsets is not None:
        shifted = [defaultdict(list) for _ in durations]
        for i, row in enumerate(plans):
            for path, atoms in row.items():
                for atom in atoms:
                    shifted[i+offsets[atom]][path].append(atom)
    for t, ceiling in enumerate(durations):
        batch = shifted[t] if shifted is not None else defaultdict(list)
        for i in range(max(0,t-factor+1), min(t+1,len(plans))) if shifted is None else ():
            lane = t-i
            for path, atoms in plans[i].items():
                batch[path].extend(shards[a][lane] for a in atoms)
        transfers = []
        for path, payload in batch.items():
            _, latency, serial, _ = catalog.capacity_costs.paths[path]
            width = max(1, math.floor((ceiling-latency)/(new_size*serial)+1e-10))
            for i in range(0,len(payload),width):
                transfers.append(dict(src=path[0], dst=path[-1], path=list(path),
                                      action='copy', chunk_ranges=payload[i:i+width]))
        if transfers:
            result['rounds'].append(dict(round=len(result['rounds'])+1,transfers=transfers))
    result['round_count'] = len(result['rounds'])
    if offsets is None and candidate['split'] == 1 and factor > 1:
        # Keep one numeric template for this request. The identity check at
        # consumption ensures it describes exactly the certified schedule.
        catalog.pipeline_template = result['rounds'], plans
    return result
