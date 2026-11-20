"""Consolidate deliveries within the certified AllGather round of their receiver."""
from .capacity_backend import native
from .cut_bounds import gather_bound
from .fabric_routes import CapacityCosts
from .replay import certify


def speculate(task, graph, plan, checked, audit):
    # Only final packet consolidation is speculative. The original plan,
    # its certificate and candidate scoring stay available to the model.
    proxy = dict(answer=dict(plan, metrics={}), diagnostics=dict(local_replay=checked,
                 search=dict(candidates=[], flow_search=audit)))
    _improve(task, graph, proxy)
    graph._prepared_aggregation = (plan['rounds'], plan['initial_possession'], checked,
                                   proxy['answer']['rounds'], proxy['diagnostics']['local_replay'],
                                   proxy['diagnostics']['search'].get('gather_aggregation'))


def improve(task, graph, result):
    cache = getattr(graph, '_prepared_aggregation', None)
    if cache is not None:
        del graph._prepared_aggregation
        source_rows, initial, previous, packed_rows, checked, detail = cache
        answer = result['answer']
        diag = result['diagnostics']
        if (answer['rounds'] is source_rows and answer['initial_possession'] is initial
                and diag['local_replay'] == previous):
            if detail:
                result['answer'] = dict(answer, rounds=packed_rows, metrics=dict(answer['metrics'],
                    total_time=checked['total_time_us']/1e6,
                    round_times=[t/1e6 for t in checked['round_times_us']]))
                diag['local_replay'] = checked
                diag['search']['gather_aggregation'] = detail
                for row in diag['search'].get('candidates', ()):
                    if row.get('selected'):
                        row['total_time_us'] = checked['total_time_us']
            return result
    return _improve(task, graph, result)


def _improve(task, graph, result):
    if native is None or not hasattr(native, 'aggregate_gather_packets'):
        return result
    answer = result['answer']
    rank_count = len(graph.ranks)
    transfers = sum(len(row['transfers']) for row in answer['rounds'])
    deliveries = rank_count * (rank_count - 1) * answer['split']
    if (task['operator'] != 'all-gather' or transfers < 20000 or deliveries > 530000
            or transfers < 8 * rank_count * len(answer['rounds'])):
        return result
    # A moderate-sized answer with at most two switch ports per rank has few
    # physically distinct alternate senders. Avoid its additional full pass.
    if transfers < 50000:
        ports = sum(neighbor in graph.switches for r in graph.ranks
                    for neighbor, _, _ in graph.adjacency[r])
        if ports < 3 * rank_count:
            return result
    diagnostics = result['diagnostics']
    parent = diagnostics['local_replay']
    if deliveries >= 5 * transfers:
        # A well-packed answer is worth revisiting when its measured time
        # still has headroom above the input-derived communication cut bound.
        bound = diagnostics['search'].get('flow_search', {}).get('cut_bound_us')
        if bound is None:
            bound = gather_bound(task, graph)
        if bound and parent['total_time_us'] < 1.4 * bound:
            return result

    size = task['data_size']['per_rank_bytes'] // answer['split']
    costs = CapacityCosts(graph)
    handle = native.make_capacity_graph(
        graph.edges, graph.edge_resources, graph.resources, costs.indices,
        graph.startup_us, graph.switch_overhead_us)
    rows, moved, saved = native.aggregate_gather_packets(
        handle, answer['initial_possession'], {rank: i for i, rank in enumerate(graph.ranks)},
        answer['rounds'], parent['round_times_us'], answer['split'], size)
    if not saved:
        return result
    candidate = dict(answer, rounds=rows)
    checked = certify(task, graph, candidate)
    if not checked['valid'] or checked['total_time_us'] > parent['total_time_us'] + 1e-6:
        return result
    candidate['metrics'] = dict(answer['metrics'], total_time=checked['total_time_us'] / 1e6,
                                round_times=[v / 1e6 for v in checked['round_times_us']])
    result['answer'] = candidate
    diagnostics['local_replay'] = checked
    search = diagnostics['search']
    for record in search.get('candidates', ()):
        if record.get('selected'):
            record['total_time_us'] = checked['total_time_us']
    search['gather_aggregation'] = dict(
        parent_us=parent['total_time_us'], replay_us=checked['total_time_us'],
        moved=moved, saved=saved)
    return result
