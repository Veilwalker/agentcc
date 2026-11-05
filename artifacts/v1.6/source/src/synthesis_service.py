"""Official service: model-authored stage edits with a safe capacity baseline."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
import time

from . import __version__
from .config import load_settings
from .llm import edit_stages
from .replay import replay
from .service import _control_candidate, _owned_search, _polished_control
from .solver import solve_schedule, validate_task
from .stage_plan import prepare, binary_control, execute
from .topology import Topology


def _model(task, problem, settings):
    try:
        from agenthack import llm_call
        trace = llm_call(model=settings.model, provider='official-platform')
    except ImportError:
        trace = nullcontext(None)
    with trace as call:
        proposal, evidence = edit_stages(task, problem, settings=settings)
        if call is not None:
            usage = evidence.get('usage', {})
            call.set(input_tokens=usage.get('prompt_tokens', 0), output_tokens=usage.get('completion_tokens', 0))
            call.result(proposal)
    return proposal, evidence


def solve_input(task_input, *, use_llm=True, search_budget_s=1.0,
                refinement_budget_s=0.25, capacity_optimization=True):
    started = time.monotonic()
    if not isinstance(task_input, dict):
        raise ValueError('input must be an object')
    task, graph = task_input['task'], Topology.from_data(task_input['topology'])
    allowed = validate_task(task, graph)
    if task_input.get('instance_id', task.get('instance_id')) != task.get('instance_id'):
        raise ValueError('instance_id does not match task')
    begin = time.monotonic()
    seed = _owned_search(solve_schedule(task, graph, seed_only=True, seed_weight=4.0,
        seed_split=allowed[0] if task['operator'] == 'all-gather' else allowed[-1], time_budget_s=search_budget_s))
    checked = seed['strategy'].pop('validation_report', None) or replay(task, graph, seed)
    phases = {'construct_s': time.monotonic() - begin}
    problem, context, extraction = prepare(task, graph, seed, checked)
    phases['edit_problem_s'] = extraction['elapsed_s']
    metadata = {'called': False, 'mode': 'offline-test'}
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='agentcc-stages') as pool:
        pending = pool.submit(copy_context().run, _model, task, problem, load_settings()) if use_llm else None
        raw, report, elapsed = _control_candidate(task, graph, seed, search_budget_s=search_budget_s,
            refinement_budget_s=refinement_budget_s, continue_construction=True)
        control, report, capacity = _polished_control(task, graph, raw, report, enabled=capacity_optimization)
        search = control['strategy']
        begin = time.monotonic()
        control, report, binary = binary_control(task, graph, control, report, context)
        phases['binary_capacity_s'] = time.monotonic() - begin
        search['binary_capacity'] = binary
        if binary['accepted']:
            parent = next(i for i, row in enumerate(search['candidates']) if row['selected'])
            search['candidates'][parent]['selected'] = False
            search['candidates'].append({'source': 'binary_capacity', 'parent_candidate': parent,
                'actual': {'widths': [1] * context.space.dimensions, 'order': extraction['stage_order']},
                'selected': True, 'valid': True, 'total_time_us': report['total_time_us']})
        wait = time.monotonic()
        if pending is not None:
            proposal, metadata = pending.result()
        phases['uncovered_model_wait_s'] = time.monotonic() - wait
    phases.update(control_search_s=elapsed, control_capacity_s=capacity['elapsed_s'])
    solution = control
    if use_llm:
        begin = time.monotonic()
        solution, evidence = execute(task, graph, control, report, context, proposal)
        report = evidence.pop('_validation_report', None) or report
        evidence.update(source_input_sha256=context.retiming.input_hash,
            prompt_base_sha256=context.retiming.seed_hash, extraction=extraction)
        if evidence['selected']:
            parent = next(i for i, row in enumerate(search['candidates']) if row['selected'])
            search['candidates'][parent]['selected'] = False
            search['candidates'].append({'source': 'model_stages' if evidence['structural_selected'] else 'model_retiming',
                'parent_candidate': parent, 'actual': {'stage_widths': evidence.get('stage_widths') if evidence['structural_selected'] else None,
                    'retiming_counts': evidence.get('compiled_counts', [])},
                'selected': True, 'valid': True, 'total_time_us': report['total_time_us']})
        search['model_edit'] = evidence
        metadata.update(schedule_stage='capacity_stage_program', source_input_sha256=context.retiming.input_hash)
        phases['edit_validation_s'] = time.monotonic() - begin
    solution['metrics'] = {'total_time': report['total_time_us'] / 1e6,
                           'round_times': [value / 1e6 for value in report['round_times_us']]}
    return {'answer': {k: v for k, v in solution.items() if k != 'strategy'},
        'diagnostics': {'version': __version__, 'llm': metadata, 'search': search,
            'local_replay': report, 'phases_s': {k: round(v, 6) for k, v in phases.items()},
            'elapsed_s': time.monotonic() - started}}


async def solve_async(task_input):
    return await asyncio.to_thread(solve_input, task_input)
