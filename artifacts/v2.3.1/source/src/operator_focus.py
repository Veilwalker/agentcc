"""Give the model a bounded routing trace and apply its choice to a verified plan."""
from collections import Counter

from . import congestion_routes


def hypotheses(task, graph, split):
    """Trace legal one-atom transfers from initial owners in a sampled matching."""
    ranks = graph.ranks
    count = len(ranks)
    if count < 64 or count > 256 or count & (count - 1):
        return []
    atom = task['data_size']['per_rank_bytes'] // split
    touches = Counter()
    first = {}
    # A relatively prime stride spreads probes across the declared rank order.
    # Each pair can exchange its own initial atom without a round dependency.
    for index in range(24):
        source = (index * (count // 4 + 1)) % count
        destination = source ^ (count // 2)
        try:
            path = graph.path(ranks[source], ranks[destination], atom)
        except ValueError:
            continue
        for u, v in zip(path, path[1:]):
            for key in ((u, v), *graph.edge_resources[u, v]):
                touches[key] += 1
                first.setdefault(key, (source, destination))
    def unit_time(key):
        capacity = graph.edges[key] if isinstance(key, tuple) else graph.resources[key]
        return atom * 1e6 / capacity

    # Preserve distinct hypotheses (pooled, fabric and endpoint bottlenecks).
    result = []
    categories = set()
    for key in sorted(touches, key=lambda key: touches[key]*unit_time(key), reverse=True):
        group = ('pool' if isinstance(key, str) else 'fabric'
                 if key[0] in graph.switches and key[1] in graph.switches else 'port')
        if group in categories:
            continue
        categories.add(group)
        result.append(dict(key=key, row=[group, *first[key], touches[key], round(unit_time(key), 4)]))
    return result if len(result) > 1 else []


def apply(flow, plan, report, proposals, evidence):
    choice = evidence.get('focus_id')
    if type(choice) is not int or not 0 <= choice < len(proposals):
        return plan, report, None
    graph, task = flow.graph, flow.task
    key = proposals[choice]['key']
    times = report['round_times_us']
    audit = dict(choice=choice, resource=key, probe=proposals[choice]['row'],
                 before_us=report['total_time_us'], accepted=False)
    size = task['data_size']['per_rank_bytes'] // plan['split']
    pooled = isinstance(key, str)
    unit = size * 1e6 / (graph.resources[key] if pooled else graph.edges[key])
    # Cap the post-model work. Inspect the two slowest rounds that fit the
    # scan budget. Aggregate every transfer in each inspected round.
    steps = sorted((i for i, row in enumerate(plan['rounds']) if len(row['transfers']) <= 2400),
                   key=times.__getitem__, reverse=True)[:2]
    best_load, best_step = 0., None
    audit['inspected_rounds'] = [i+1 for i in steps]
    for step in steps:
        atoms = 0
        for move in plan['rounds'][step]['transfers']:
            path = move['path']
            edges = zip(path, path[1:])
            multiplicity = (sum(key in graph.edge_resources[edge] for edge in edges)
                            if pooled else int(key in edges))
            atoms += multiplicity * len(move['chunk_ranges'])
        load = atoms * unit
        if load >= times[step]*.55 and load > best_load:
            best_load, best_step = load, step
    if best_step is None:
        audit['decision'] = 'low_observed_load'
        return plan, report, audit
    audit.update(round=best_step+1, occupied_us=best_load)
    candidate, checked, changed = congestion_routes.improve(
        task, graph, plan, report, flow.catalog, passes=3, focused_round=best_step, resource=key)
    audit.update(accepted=bool(changed), after_us=checked['total_time_us'],
                 decision='verified_improvement' if changed else 'incumbent_retained')
    return candidate, checked, audit
