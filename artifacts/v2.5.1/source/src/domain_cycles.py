"""Route a set of rotating multicast lanes over balanced single-port domains."""
from collections import defaultdict
from math import floor
from time import monotonic

from .collectives import _initial
from .capacity_backend import native
from .replay import certify


def _layout(graph):
    domains = defaultdict(list)
    for rank in graph.ranks:
        neighbors = graph.adjacency[rank]
        if len(neighbors) != 1:
            return None
        leaf = neighbors[0][0]
        if leaf not in graph.switches or (leaf, rank) not in graph.edges:
            return None
        domains[leaf].append(rank)
    sizes = {len(members) for members in domains.values()}
    if (len(sizes) != 1 or not 3 <= len(domains) <= 32 or not 2 <= min(sizes) <= 32
            or len(graph.switches) > 128):
        return None
    return list(domains), list(domains.values())


def _cores(graph, leaves, costs, atom):
    domain_ports = set(leaves)
    count = len(leaves)
    options = [[] for _ in range(count * count)]
    arcs = {u: [(v,1/graph.edges[u,v]) for v,_,_ in graph.adjacency[u]
                if v in graph.switches] for u in graph.switches}
    per_switch = graph.switch_overhead_us / (atom*1e6)
    for src, leaf in enumerate(leaves):
        paths = defaultdict(list)
        stack = [(leaf, (leaf,), 0.)]
        work = 0
        while stack:
            node, prefix, serial = stack.pop()
            if len(prefix) == 6:
                continue
            for peer, inverse_rate in arcs[node]:
                if peer in prefix:
                    continue
                work += 1
                if work > 18000:
                    return None
                path = prefix + (peer,)
                cost = serial + inverse_rate
                if peer in domain_ports:
                    paths[peer].append((cost+(len(path)-2)*per_switch,path))
                else:
                    stack.append((peer,path,cost))
        for dst, target in enumerate(leaves):
            if src == dst:
                continue
            if not paths[target]:
                return None
            paths[target].sort(key=lambda item:item[0])
            options[src*count+dst] = [costs.compile(p) for _,p in paths[target][:16]]
    return options


def _local_load(graph, leaves, groups, costs, atom):
    # Any nonzero cyclic offset within every group is a perfect matching.
    # All these matchings use every rank-to-port edge exactly once in each
    # direction; shared pool incidences also remain identical across offsets.
    loads = defaultdict(float)
    serial = delay = 0.
    for leaf, group in zip(leaves, groups):
        for i, src in enumerate(group):
            terms, delay, _, _ = costs.compile((src, leaf, group[(i+1) % len(group)]))
            for resource, coefficient in terms:
                loads[resource] += atom * coefficient
        serial = max(serial, atom * 1e6 * (
            max(1 / graph.edges[src,leaf] for src in group)
            + max(1 / graph.edges[leaf,dst] for dst in group)))
    return loads, delay, serial


def _packing(steps, width, cap, max_packet):
    # Assign arrivals backward from the final local fanout. Each local sender
    # has a new token at the start of each step. All offsets due this step get
    # one slot, and free capacity packs older ready arrivals onto the same path.
    remain = [steps-1] * width
    windows = [[] for _ in range(width)]
    for t in reversed(range(steps)):
        due = [d for d in range(1,width) if remain[d] == t]
        final = t == steps-1
        room = 0 if final else cap - len(due)
        for d in due:
            size = min(t+1, 1+room, max_packet)
            windows[d].append((t,t+1-size,t+1))
            remain[d] -= size
            room -= size-1
    return windows


def improve(task, graph, plan, report, catalog, allowed):
    before = report['total_time_us']
    if native is None or task['operator'] != 'all-gather' or 1 not in allowed:
        return plan, report, None
    layout = _layout(graph)
    if layout is None:
        return plan, report, None
    leaves, groups = layout
    steps, width = len(leaves), len(groups[0])
    if steps > task['max_nonempty_rounds']:
        return plan, report, None
    started = monotonic()
    atom = task['data_size']['per_rank_bytes']
    costs = catalog.capacity_costs
    local, local_delay, local_serial = _local_load(graph, leaves, groups, costs, atom)
    # Every local rank sends and receives one atom for each of these deliveries,
    # regardless of how the inter-domain tours are chosen. Their endpoint load
    # alone can disqualify the family on a changed or constrained input.
    endpoint_units = steps*(width-1) + steps-1
    lower = max(local.values()) * endpoint_units
    if lower >= before*.993:
        return plan, report, dict(endpoint_bound_us=lower)
    cores = _cores(graph, leaves, costs, atom)
    if cores is None:
        return plan, report, None
    core_peak, tours = native.domain_cycles(steps, width, len(costs.indices), atom, cores)

    cross = defaultdict(float)
    inter_delay = inter_serial = 0.
    lanes = []
    for lane, tour in enumerate(tours):
        routes = []
        for a, b, choice in tour:
            src, dst = groups[a][lane], groups[b][lane]
            core = cores[a*steps+b][choice][3]
            terms, delay, serial, path = costs.compile((src,*core,dst))
            for resource, coefficient in terms:
                cross[resource] += atom * coefficient
            inter_delay = max(inter_delay,delay)
            inter_serial = max(inter_serial, delay + atom*serial)
            routes.append((a,path))
        lanes.append(routes)
    overhead = max(local_delay,inter_delay)
    mixed_peak = max(cross.get(k,0.) + (width-1)*local.get(k,0.) for k in cross.keys() | local.keys())
    wave = max(inter_serial,local_delay+local_serial,overhead+mixed_peak)
    tail = max(local_serial, max(local.values())*(width-1)) + local_delay
    predicted = (steps-1)*wave+tail
    audit = dict(domains=steps, lanes=width, predicted_us=predicted, core_wave_busy_us=core_peak)
    if predicted >= before * 1.06:
        return plan, report, audit
    capacity = max(width - 1, min(
        floor((wave - overhead - cross.get(key, 0.0)) / work + 1e-8)
        for key, work in local.items()
    ))
    packet = max(1, floor((wave - local_delay) / local_serial + 1e-8))
    windows = _packing(steps, width, capacity, packet)
    relay_rows = None
    if (3 <= steps <= 24 and steps * width >= 64
            and core_peak > (width - 1) * max(local.values()) * 1.1):
        # Leave room on rank access links for holders that pick up extra forwarding.
        new_capacity = width - 1 + max(1, (capacity - width + 1) // 2)
        max_packet = min(packet, max(2, (new_capacity - width + 1) // 2 + 2))
        trial_windows = _packing(steps, width, new_capacity, max_packet)
        local_loads = [[0, 0] for _ in range(steps)]
        for assignments in trial_windows:
            for wave_id, first, stop in assignments:
                load = local_loads[wave_id]
                load[0] += stop - first
                load[1] = max(load[1], stop - first)
        endpoints = [
            costs.compile(path)
            for group, leaf in zip(groups, leaves)
            for rank in group
            for path in ((rank, leaf), (leaf, rank))
        ]
        trial = native.cycle_relays(
            len(costs.indices), atom, graph.switch_overhead_us, local_delay, local_serial,
            cores, endpoints, tours, list(local.items()), [tuple(row) for row in local_loads],
        )
        if trial is not None and trial[0] < min(before, predicted) * .99:
            predicted, relay_rows = trial
            capacity, packet, windows = new_capacity, max_packet, trial_windows
            audit['relay_priced_us'] = predicted
    if predicted >= before * .993:
        return plan, report, audit

    rows = [[] for _ in range(steps)]
    for lane, routes in enumerate(lanes):
        for position, (domain, path) in enumerate(routes):
            src, dst = path[0], path[-1]
            if relay_rows is None:
                # In a rotating tour, each wave delivers the preceding domain's token.
                payload = [
                    f'{groups[routes[(position - wave_id) % steps][0]][lane]}_c0'
                    for wave_id in range(steps)
                ]
            else:
                # The local fanout can use the incoming token from the next
                # boundary. Each entry identifies the selected token's owner.
                previous = lane * steps + (position - 1) % steps
                payload = [f'{src}_c0'] + [
                    f'{groups[wave[previous][2]][lane]}_c0' for wave in relay_rows
                ]

            for wave_id in range(steps - 1):
                selected_token = payload[wave_id]
                selected_path = path
                if relay_rows is not None:
                    parent, route, owner = relay_rows[wave_id][lane * steps + position]
                    destination = tours[lane][position][1]
                    core = cores[parent * steps + destination][route][3]
                    selected_token = f'{groups[owner][lane]}_c0'
                    selected_path = (groups[parent][lane], *core, dst)
                rows[wave_id].append(dict(
                    src=selected_path[0], dst=dst, action='copy',
                    path=list(selected_path), chunk_ranges=[selected_token],
                ))

            for offset in range(1, width):
                peer = groups[domain][(lane + offset) % width]
                local_path = [src, leaves[domain], peer]
                for wave_id, first, stop in windows[offset]:
                    rows[wave_id].append(dict(
                        src=src, dst=peer, action='copy',
                        path=local_path, chunk_ranges=payload[first:stop],
                    ))
    candidate = _initial(task,graph.ranks,1)
    candidate.update(rounds=[dict(round=i+1,transfers=row) for i,row in enumerate(rows)],round_count=steps)
    checked = certify(task,graph,candidate)
    audit.update(valid=checked['valid'], elapsed_s=round(monotonic()-started,6), cap=capacity,
                 max_packet=packet, actual_us=checked['total_time_us'])
    if checked['valid'] and checked['total_time_us'] < before*.993:
        return candidate, checked, audit
    return plan, report, audit
