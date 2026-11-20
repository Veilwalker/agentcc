"""Pipe disjoint reduction stripes across local stars and inter-domain stars."""
from collections import defaultdict
from math import floor

from .capacity_backend import native
from .collectives import _initial
from .cut_bounds import covered_capacity
from .fabric_routes import RouteCatalog
from .local_flatten import _coalesce_cycles
from .reduction_parents import regenerate
from .replay import certify


def layout(graph):
    """Find uniform domains with private sets of two or three actual duplex ports."""
    groups = defaultdict(list)
    domains = {}
    for rank in graph.ranks:
        ports = tuple(sorted(peer for peer, _, _ in graph.adjacency[rank]))
        if not 2 <= len(ports) <= 3 or any(
                port not in graph.switches or (port,rank) not in graph.edges for port in ports):
            return None
        for port in ports:
            if domains.setdefault(port, ports) != ports:
                return None
        groups[ports].append(rank)
    sizes = {len(group) for group in groups.values()}
    if len(sizes) != 1 or not 4 <= len(groups) <= 32 or not 4 <= sizes.pop() <= 32:
        return None
    domain = {rank: ports for ports, ranks in groups.items() for rank in ranks}
    domain.update(domains)
    # Include every cut edge in each direction, even on asymmetric fabrics.
    cuts = [[], []]
    for u,v in graph.edges:
        if domain.get(u) != domain.get(v):
            if u in domain:
                cuts[0].append((u,v))
            if v in domain:
                cuts[1].append((u,v))
    if not all(cuts):
        return None
    bandwidth = min(covered_capacity(graph,cut) for cut in cuts)
    return list(groups.values()), bandwidth


def price(task, graph, allowed):
    if (native is None or task['operator'] != 'all-reduce' or not 64 <= len(graph.ranks) <= 256
            or task['max_nonempty_rounds'] < 5):
        return None
    structure = layout(graph)
    if structure is None:
        return None
    groups, bandwidth = structure
    count, width, split = len(groups), len(groups[0]), allowed[-1]
    if not width <= split <= 64 or split % width:
        return None
    total_bytes = task['data_size']['per_rank_bytes']
    size = total_bytes // split
    cut_us = 2 * (count-1) * total_bytes * 1e6 / bandwidth
    catalog = RouteCatalog(graph, size, branch_ports=True, deep_exits=True)
    costs = catalog.capacity_costs
    stages = [defaultdict(list) for _ in range(5)]
    for slot in range(split):
        reps = [members[(slot+3*d) % width] for d,members in enumerate(groups)]
        root = (slot//width + slot) % count
        launch = (slot//count + slot%count) % 2
        for domain, members in enumerate(groups):
            rep = reps[domain]
            for rank in members:
                if rank != rep:
                    stages[launch][rank,rep,False].append(slot)
                    stages[launch+3][rep,rank,True].append(slot)
            if domain != root:
                stages[launch+1][rep,reps[root],False].append(slot)
                stages[launch+2][reps[root],rep,True].append(slot)

    options, recipe = {}, []
    for stage in stages:
        jobs, entries = [], []
        for (u,v,copy), slots in stage.items():
            if (u,v) not in options:
                options[u,v] = tuple(costs.compile(p) for p in catalog.options(u,v))
            routes = options[u,v]
            if not routes or len(routes) > 128:
                return None
            entries.append((copy,routes,slots))
            jobs.append((routes,len(slots)))
        selected, _, duration, _ = native.optimize_groups(len(costs.indices),jobs,2,costs.round_cache,3,size,True)
        recipe.append((duration,entries,selected))
    estimate = sum(row[0] for row in recipe)
    # Spend the full generic search on poorly matched fabrics. The cut is an
    # admission heuristic; the full symbolic verification still runs below.
    if estimate > cut_us * 1.72:
        return None
    rows = []
    for duration, entries, selected in recipe:
        choices = iter(selected)
        actions = {False:[],True:[]}
        for copy, paths, slots in entries:
            count_paths = min(3,len(paths),len(slots))
            chosen = [paths[next(choices)][3] for _ in range(count_paths)]
            actions[copy].append((chosen,slots))
        moves = []
        for copy, assignments in actions.items():
            _coalesce_cycles(assignments)
            batches = defaultdict(list)
            for paths, slots in assignments:
                for i,path in enumerate(paths):
                    batches[path].extend(slots[len(slots)*i//len(paths):len(slots)*(i+1)//len(paths)])
            for path,slots in batches.items():
                _,latency,inverse,_ = costs.paths[path]
                limit = max(1,floor((duration-latency)/(size*inverse)+1e-10))
                tokens = [f'c{s}_sum' if copy else f'{path[0]}_c{s}' for s in slots]
                for i in range(0,len(tokens),limit):
                    moves.append(dict(src=path[0],dst=path[-1],path=list(path),action='copy' if copy else 'reduce',
                                      chunk_ranges=tokens[i:i+limit]))
        rows.append(dict(round=len(rows)+1,transfers=moves))
    plan = _initial(task,graph.ranks,split)
    plan.update(rounds=regenerate(rows,graph.ranks,split),round_count=len(rows))
    checked = certify(task,graph,plan)
    if checked['valid'] and abs(checked['total_time_us']-estimate) < max(1e-4,estimate*1e-9):
        return plan, checked, catalog, dict(domains=count,domain_width=width,cut_us=cut_us,
                                           priced_us=estimate,launch_rounds=2)
    return None
