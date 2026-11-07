"""Generate a solver, execute it on current capacities, verify, and repair.

The model supplies executable search logic. Capacity construction and Replay
remain deterministic engineering kernels. Invalid or inferior proposals never
replace the independently validated incumbent.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import hashlib
import time

from . import __version__
from .capacity_backend import BACKEND
from .code_policy import propose
from .config import load_settings
from .llm import LLMError
from .program_vm import Program, ProgramError
from .replay import replay
from .service import _control_candidate, _owned_search, _polished_control
from .solver import solve_schedule, validate_task
from .stage_search import SearchSpace, instance_context
from .topology import Topology


PATTERNS = {
    1: [(1,)],
    2: [(1, 1), (2,)],
    3: [(1, 1, 1), (2, 1), (1, 2), (3,)],
}


def fixed_band_guard(space):
    """Keep the previous finite band family as a deterministic quality floor."""
    plain = min(
        (profile.plain for profile in space.profiles),
        key=lambda item: sum(item.transition(i, 1)[2] for i in range(item.dimensions)),
    )
    sizes = [3] * (plain.dimensions // 3)
    if plain.dimensions % 3:
        sizes.append(plain.dimensions % 3)
    if plain.task['operator'] == 'all-gather':
        sizes.reverse()
    first, widths = 0, []
    for size in sizes:
        ranked = []
        for pattern in PATTERNS[size]:
            position, cost = first, 0.0
            for width in pattern:
                cost += plain.transition(position, width)[2]
                position += width
            ranked.append((cost, pattern))
        widths.extend(min(ranked)[1])
        first += size
    rounds = len(widths) * (2 if plain.task['operator'] == 'all-reduce' else 1)
    if rounds > plain.task['max_nonempty_rounds']:
        return None
    return plain.compile(widths)


def execute_program(source, space, protocol='outgoing_v1'):
    """Audit code against actual tables without manufacturing a replacement."""
    started = time.monotonic()
    evidence = {
        'source': source,
        'sha256': hashlib.sha256(source.encode()).hexdigest(),
        'valid_program': False,
        'selected': False,
        'program_protocol': protocol,
    }
    try:
        program = Program(source)
        outputs, instructions = [], 0
        for profile, table in enumerate(space.costs):
            if protocol == 'incoming_v1':
                inputs = [[]] + [
                    [table[i-w][w-1] if w <= len(table[i-w]) else float('inf')
                     for w in range(1, min(3, i) + 1)]
                    for i in range(1, len(table) + 1)]
            elif protocol == 'outgoing_v1':
                inputs = table + [[]]
            else:
                raise ValueError('unknown generated program protocol')
            widths = program.run(inputs, space.limit)
            profile, widths, cost = space.evaluate([profile, widths])
            outputs.append((cost, profile, widths))
            instructions += program.steps
        predicted, profile, widths = min(outputs)
        bound = space.audit_bound()
        evidence.update(
            valid_program=True, output=[profile, widths], predicted_us=predicted,
            instructions=instructions, bound_us=bound,
            gap_us=max(0.0, predicted - bound),
            profile_results=[{'cost_us': c, 'profile': p, 'widths': w} for c, p, w in outputs],
        )
    except (ProgramError, ValueError, TypeError, RecursionError) as error:
        evidence['error'] = str(error)
    evidence['execution_s'] = time.monotonic() - started
    return evidence


def verify_proposal(task, graph, space, execution, incumbent, report):
    """Compile only promising proposals; record exactly what was checked."""
    if not execution['valid_program']:
        execution['decision'] = 'invalid_program'
        return incumbent, report
    if execution['predicted_us'] >= report['total_time_us'] - 1e-6:
        execution['decision'] = 'capacity_preview_not_better'
        return incumbent, report
    started = time.monotonic()
    try:
        candidate, predicted = space.compile(execution['output'])
        checked = replay(task, graph, candidate)
        execution.update(
            replay_valid=checked['valid'], replay_errors=checked.get('errors', []),
            replay_us=checked['total_time_us'],
        )
        matches = abs(checked['total_time_us'] - predicted) < 1e-4
        if checked['valid'] and matches and checked['total_time_us'] < report['total_time_us'] - 1e-6:
            execution.update(selected=True, decision='verified_improvement')
            return candidate, checked
        execution['decision'] = 'replay_rejected_or_not_better'
    except (ValueError, TypeError, KeyError) as error:
        execution.update(decision='compile_rejected', error=str(error))
    finally:
        execution['validation_s'] = time.monotonic() - started
    return incumbent, report


def solve_input(task_input, *, use_llm=True, search_budget_s=1.0,
                refinement_budget_s=0.25, capacity_optimization=True):
    started = time.monotonic()
    if not isinstance(task_input, dict):
        raise ValueError('input must be an object')
    task, graph = task_input['task'], Topology.from_data(task_input['topology'])
    allowed = validate_task(task, graph)
    if task_input.get('instance_id', task.get('instance_id')) != task.get('instance_id'):
        raise ValueError('instance_id does not match task')

    count = len(graph.ranks)
    if not 2 <= count <= 512 or count & (count - 1):
        # The generic model-edit path remains available outside the stage
        # compiler's domain. Do not claim code generation for that path.
        from .synthesis_service import solve_input as legacy_solve
        output = legacy_solve(
            task_input, use_llm=use_llm, search_budget_s=search_budget_s,
            refinement_budget_s=refinement_budget_s,
            capacity_optimization=capacity_optimization,
        )
        output['diagnostics']['generated_solver_unavailable'] = 'rank_count_outside_stage_domain'
        output['diagnostics']['capacity_backend'] = BACKEND
        return output

    problem = instance_context(task, graph)
    phases, interactions, executions = {}, [], []
    settings = load_settings() if use_llm else None
    source, stop_reason = None, 'offline_test'
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='agentcc-code') as pool:
        pending = pool.submit(copy_context().run, propose, problem, settings) if use_llm else None
        begin = time.monotonic()
        seed = _owned_search(solve_schedule(
            task, graph, seed_only=True, seed_weight=4.0,
            seed_split=allowed[0] if task['operator'] == 'all-gather' else allowed[-1],
            time_budget_s=search_budget_s,
        ))
        seed['strategy'].pop('validation_report', None)
        phases['construct_s'] = time.monotonic() - begin
        begin = time.monotonic()
        try:
            space, space_error = SearchSpace(task, graph, seed['split']), None
        except ValueError as error:
            space, space_error = None, str(error)
        phases['capacity_space_s'] = time.monotonic() - begin
        raw, report, elapsed = _control_candidate(
            task, graph, seed, search_budget_s=search_budget_s,
            refinement_budget_s=refinement_budget_s, continue_construction=True,
        )
        solution, report, capacity = _polished_control(
            task, graph, raw, report, enabled=capacity_optimization,
        )
        phases.update(control_search_s=elapsed, control_capacity_s=capacity['elapsed_s'])
        search, guard = solution['strategy'], {'selected': False}
        if space is not None:
            begin = time.monotonic()
            old = fixed_band_guard(space)
            if old is not None:
                candidate, predicted = old
                guard['predicted_us'] = predicted
                if predicted < report['total_time_us'] - 1e-6:
                    checked = replay(task, graph, candidate)
                    if checked['valid'] and checked['total_time_us'] < report['total_time_us'] - 1e-6:
                        solution, report = candidate, checked
                        guard['selected'] = True
            phases['guard_s'] = time.monotonic() - begin
        control_us = report['total_time_us']
        wait = time.monotonic()
        if pending is not None:
            try:
                source, metadata = pending.result()
                interactions.append(metadata)
            except LLMError as error:
                interactions.append({'called': True, 'error': str(error),
                    'duration_ms': getattr(error, 'duration_ms', 0), 'usage_unavailable': True})
                stop_reason = 'model_transport_failed'
        phases['uncovered_model_wait_s'] = time.monotonic() - wait

        if source is not None and space is not None:
            for attempt in range(2):
                protocol = interactions[-1].get('program_protocol', 'outgoing_v1')
                result = execute_program(source, space, protocol)
                executions.append(result)
                solution, report = verify_proposal(task, graph, space, result, solution, report)
                if result['selected']:
                    for previous in executions[:-1]:
                        previous['selected'] = False
                gap = result.get('gap_us', 0.0)
                tolerance = max(1e-6, result.get('bound_us', 0.0) * 0.001)
                needs_repair = not result['valid_program'] or gap > tolerance or 'error' in result
                if not needs_repair:
                    stop_reason = 'finite_space_gap_within_tolerance'
                    break
                if attempt or time.monotonic() - started > 30:
                    stop_reason = 'repair_budget_exhausted'
                    break
                feedback = {
                    'source': source, 'error': result.get('error'),
                    'cost_us': result.get('predicted_us'),
                    'finite_space_bound_us': result.get('bound_us'),
                    'dimensions': space.dimensions,
                    'row_lengths': [len(row) for row in space.costs[0]] + [0],
                    'instruction': ('The source includes the host wrapper. Return only a corrected loop body.'
                                    if protocol == 'incoming_v1' else 'Return only a corrected function.'),
                }
                try:
                    source, metadata = propose(problem, settings, feedback)
                    interactions.append(metadata)
                except LLMError as error:
                    interactions.append({'called': True, 'error': str(error),
                        'duration_ms': getattr(error, 'duration_ms', 0), 'usage_unavailable': True})
                    stop_reason = 'repair_transport_failed'
                    break
        elif space is None:
            stop_reason = 'capacity_space_unavailable'

    model_selected = any(execution['selected'] for execution in executions)
    search['model_edit'] = {
        'selected': model_selected, 'accepted': model_selected,
        'semantics': 'model_generated_stage_solver_v1',
        'control_time_us': control_us, 'final_time_us': report['total_time_us'],
        'executions': executions, 'guard': guard, 'stop_reason': stop_reason,
        'space': space.evidence() if space else {'unavailable': space_error},
    }
    if guard['selected'] or model_selected:
        for candidate in search['candidates']:
            candidate['selected'] = False
        search['candidates'].append({
            'source': 'model_generated_code' if model_selected else 'fixed_band_guard',
            'actual': next((e['output'] for e in executions if e['selected']), None),
            'selected': True, 'valid': True, 'total_time_us': report['total_time_us'],
        })
    metadata = {
        'called': bool(interactions), 'mode': 'generated_solver', 'calls': len(interactions),
        'duration_ms': sum(item.get('duration_ms', 0) for item in interactions),
        'usage': {key: sum(item.get('usage', {}).get(key, 0) for item in interactions)
                  for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'interactions': interactions,
    }
    solution['metrics'] = {
        'total_time': report['total_time_us'] / 1e6,
        'round_times': [value / 1e6 for value in report['round_times_us']],
    }
    return {
        'answer': {key: value for key, value in solution.items() if key != 'strategy'},
        'diagnostics': {
            'version': __version__, 'llm': metadata, 'search': search,
            'capacity_backend': BACKEND,
            'local_replay': report, 'elapsed_s': time.monotonic() - started,
            'phases_s': {key: round(value, 6) for key, value in phases.items()},
        },
    }


async def solve_async(task_input):
    return await asyncio.to_thread(solve_input, task_input)
