"""Live capacity context and strict compilation of model scheduling programs."""
from dataclasses import dataclass
import time

from .collectives import locality_order
from .fabric_routes import RouteCatalog
from .retime_plan import PlanContext, prepare as prepare_retiming, execute as execute_retiming
from .stage_program import StageSpace, validate_widths
from .replay import replay
from .schedule_edit import _input_digest


@dataclass(frozen=True)
class ProgramContext:
    retiming: PlanContext
    space: StageSpace | None
    bands: tuple[tuple[tuple[int, ...], ...], ...]
    mode: str


def _prepare_stages(task, graph, seed):
    """Bounded structural work. Failure leaves the original retiming mode."""
    n = len(graph.ranks)
    if not 2 <= n <= 512 or n & (n - 1):
        return None, 'unavailable', [], []
    catalog = RouteCatalog(graph, task['data_size']['per_rank_bytes'] // seed['split'], branch_ports=True)
    space = StageSpace(task, graph, seed['split'], catalog=catalog)
    base_cost = sum(space.transition(i, 1)[2] for i in range(space.dimensions))
    order, order_name = locality_order(graph), 'input'
    if order != list(graph.ranks):
        alternate = StageSpace(task, graph, seed['split'], order=order, catalog=catalog)
        alternate_cost = sum(alternate.transition(i, 1)[2] for i in range(alternate.dimensions))
        if alternate_cost < base_cost - 1e-6:
            space, order_name = alternate, 'capacity_locality'
    # AG volume grows with depth; AR reduction traffic falls with depth.
    # Bands bound the model's choices to independent concrete rewrites.
    d = space.dimensions
    sizes = [3] * (d // 3) + ([d % 3] if d % 3 else [])
    if task['operator'] == 'all-gather':
        sizes.reverse()
    first, bands, band_costs = 0, [], []
    for width in sizes:
        patterns = {1: ((1,),), 2: ((1, 1), (2,)),
                    3: ((1, 1, 1), (2, 1), (1, 2), (3,))}[width]
        costs = []
        for pattern in patterns:
            position, cost = first, 0.0
            for step in pattern:
                cost += space.transition(position, step)[2]
                position += step
            costs.append(float(f'{cost:.4g}'))
        bands.append(patterns)
        band_costs.append(costs)
        first += width
    return space, order_name, bands, band_costs


def prepare(task, graph, seed, report):
    started = time.monotonic()
    old, bound, evidence = prepare_retiming(task, graph, seed, report)
    try:
        space, order_name, bands, band_costs = _prepare_stages(task, graph, seed)
    except ValueError as error:
        space, order_name, bands, band_costs = None, 'unavailable', [], []
        evidence['stage_unavailable_reason'] = str(error)
    mode = 'bands' if band_costs and sum(min(row) for row in band_costs) < report['total_time_us'] * .95 else 'retime'
    named_costs = [dict(zip(pattern_names(len(row)), row)) for row in band_costs]
    problem = ({'op': old['op'], 'task': old['task'], 'bands': named_costs,
                'e': old['e'], 'p': old['p']} if mode == 'bands' else old)
    table = [[first, width, round(row[2], 6)] for (first, width), row in sorted(space.stages.items())] if space else []
    evidence.update(elapsed_s=time.monotonic() - started, stage_order=order_name,
        stage_rank_order=space.order if space else [], stage_dimensions=space.dimensions if space else 0,
        stage_table=table, band_patterns=bands, stage_protocol='capacity_band_edits_v3',
        route_relaxations=space.catalog.relaxations if space else 0,
        route_pairs=len(space.catalog.paths) if space else 0, mode=mode)
    return problem, ProgramContext(bound, space, tuple(bands), mode), evidence


def pattern_names(count):
    return {1: ('keep',), 2: ('keep', 'all'), 4: ('keep', 'left', 'right', 'all')}[count]


def compile_bands(space, bands, selected):
    """Execute the model's exact local dimension rewrites, without repair."""
    # A singleton wrapper is a syntax alias only: preserve every model index.
    # Mixed, empty or multi-value wrappers still fail strict validation below.
    if isinstance(selected, list) and selected and all(isinstance(item, list) and len(item) == 1 for item in selected):
        selected = [item[0] for item in selected]
    if isinstance(selected, list) and len(selected) == len(bands) and all(isinstance(item, str) for item in selected):
        selected = [pattern_names(len(patterns)).index(name) for name, patterns in zip(selected, bands)]
    if (not isinstance(selected, list) or len(selected) != len(bands)
            or any(type(code) is not int or not 0 <= code < len(patterns)
                   for code, patterns in zip(selected, bands))):
        raise ValueError('one offered integer pattern index is required per band')
    if not bands or not any(selected):
        return []
    widths = [width for code, patterns in zip(selected, bands) for width in patterns[code]]
    validate_widths(widths, space.dimensions)
    return widths


def binary_control(task, graph, solution, report, context):
    """A measured binary capacity baseline; no model instruction is invented."""
    space = context.space
    if space is None:
        return solution, report, {'accepted': False}
    predicted = sum(space.transition(i, 1)[2] for i in range(space.dimensions))
    if predicted >= report['total_time_us'] - 1e-6:
        return solution, report, {'accepted': False, 'predicted_us': predicted}
    try:
        candidate, predicted = space.compile([1] * space.dimensions)
    except ValueError:
        return solution, report, {'accepted': False, 'reason': 'binary_round_limit'}
    checked = replay(task, graph, candidate)
    accepted = checked['valid'] and checked['total_time_us'] < report['total_time_us'] - 1e-6
    return (candidate, checked, {'accepted': True, 'predicted_us': predicted}) if accepted else (
        solution, report, {'accepted': False, 'predicted_us': predicted})


def execute(task, graph, control, report, context, proposal):
    started = time.monotonic()
    base = {'accepted': False, 'selected': False, 'requested': proposal, 'errors': [],
            'control_time_us': report['total_time_us'], 'final_time_us': report['total_time_us'],
            'semantics': 'capacity_band_edits_v3'}
    try:
        if _input_digest(task, graph) != context.retiming.input_hash:
            raise ValueError('program input binding differs')
        if context.mode == 'retime':
            current, evidence = execute_retiming(task, graph, control, report, context.retiming, proposal)
            evidence.update(structural_selected=False, stage_widths=[])
            return current, evidence
        if isinstance(proposal, list):
            instructions = proposal
        elif isinstance(proposal, dict) and set(proposal) == {'bands'}:
            instructions = proposal['bands']
        else:
            raise ValueError('stage reply requires an edit array or a bands object')
        widths = compile_bands(context.space, context.bands, instructions)
        current, evidence = control, {**base, 'structural_selected': False, 'stage_widths': widths}
        if widths:
            candidate, predicted = context.space.compile(widths)
            evidence['structural_predicted_us'] = predicted
            if predicted < evidence['final_time_us'] - 1e-6:
                checked = replay(task, graph, candidate)
                if checked['valid'] and checked['total_time_us'] < evidence['final_time_us'] - 1e-6:
                    evidence.update(selected=True, accepted=True, structural_selected=True,
                        compiled_counts=[], compiled_atomic_count=0,
                        final_time_us=checked['total_time_us'], after_us=checked['total_time_us'],
                        semantics='mixed_radix_stage_program', validation='complete_replay',
                        _validation_report=checked)
                    evidence.pop('reason', None)
                    current = candidate
        evidence['elapsed_s'] = time.monotonic() - started
        return current, evidence
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return control, {**base, 'reason': 'rejected', 'errors': [str(error)],
                         'elapsed_s': time.monotonic() - started}
