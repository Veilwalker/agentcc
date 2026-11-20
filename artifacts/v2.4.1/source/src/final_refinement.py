"""Price bounded timing variants of the selected plan, certify the winner."""
from . import dependency_schedule, diagonal_gather, shard_pipeline, slot_packing
from .capacity_backend import native
from .replay import certify as replay


def improve(task, graph, solution, checked, catalog, allowed):
    parent = checked['total_time_us']
    audit = {'parent_us': parent, 'selected': None, 'prices_us': {}}
    if native is None or len(graph.ranks) < 16:
        return solution, checked, audit
    plan = None
    best = parent * 0.99
    if task['operator'] == 'all-gather':
        fractional = shard_pipeline.price(task, graph, solution, parent, catalog, allowed)
        if fractional is not None:
            us = sum(fractional[2])
            audit['prices_us']['split_pipeline'] = us
            if us < best:
                best, plan = us, ('split_pipeline', fractional)
        try:
            grid = diagonal_gather.price(task, graph, solution, catalog, allowed)
        except ValueError:
            grid = None
        if grid is not None:
            us = sum(grid[2])
            audit['prices_us']['diagonal_gather'] = us
            if us < best:
                best, plan = us, ('diagonal_gather', grid)
    else:
        packed = slot_packing.price(task, graph, solution, catalog)
        if packed is not None:
            us, recipe = packed
            audit['prices_us']['slot_packing'] = us
            if us < best:
                best, plan = us, ('slot_packing', recipe)
        wave = slot_packing.wave_price(task, graph, solution, catalog)
        if wave is not None:
            us, recipe = wave
            audit['prices_us']['slot_waves'] = us
            if us < best:
                best, plan = us, ('slot_waves', recipe)
    if plan is None and (task['operator'] != 'all-reduce' or any(
            (m['src'], m['dst']) in graph.edges and len(m['path']) > 2
            for row in solution['rounds'] for m in row['transfers'])):
        timing = dependency_schedule.price(task, graph, solution, catalog, direct=True)
        if timing is not None:
            us, recipe = timing
            audit['prices_us']['causal_timing'] = us
            if us < best:
                best, plan = us, ('causal_timing', recipe)
    if plan is not None:
        family, recipe = plan
        if family in ('split_pipeline', 'diagonal_gather'):
            candidate = shard_pipeline.materialize(task, graph, solution, catalog, recipe)
        elif family == 'slot_waves':
            candidate = slot_packing.wave_materialize(solution, recipe)
        elif family == 'slot_packing':
            candidate = slot_packing.materialize(solution, recipe)
        else:
            candidate = dependency_schedule.materialize(solution, recipe)
        result = replay(task, graph, candidate)
        audit.update(verified=result['valid'], observed_us=result['total_time_us'])
        if (result['valid'] and abs(result['total_time_us'] - best) < max(1e-4, best * 1e-9)
                and result['total_time_us'] < parent - 1e-6):
            audit['selected'] = ('tree_offsets' if family == 'split_pipeline' and recipe[3] is not None else family)
            return candidate, result, audit
    return solution, checked, audit
