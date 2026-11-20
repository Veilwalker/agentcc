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
from .cohort_pipeline import pipeline
from .code_policy import propose
from .config import load_settings
from .llm import LLMError
from .program_vm import Program, ProgramError
from .replay import replay
from .service import _control_candidate, _owned_search, _polished_control
from .solver import solve_schedule, validate_task
from .seed_preview import SeedPreview
from .refined_search import RefinedSearchSpace as SearchSpace
from .stage_search import instance_context
from .topology import Topology


PATTERNS = {
    1: [(1,)],
    2: [(1, 1), (2,)],
    3: [(1, 1, 1), (2, 1), (1, 2), (3,)],
}


def fixed_band_projection(space):
    """Price the old band family without building transfers or atom strings."""
    plain = min(
        (profile.plain for profile in space.profiles),
        key=lambda item: sum(item.transition(i, 1)[2] for i in range(item.dimensions)),
    )
    sizes = [3] * (plain.dimensions // 3)
    if plain.dimensions % 3:
        sizes.append(plain.dimensions % 3)
    if plain.task['operator'] == 'all-gather':
        sizes.reverse()
    first, widths, total = 0, [], 0.0
    for size in sizes:
        ranked = []
        for pattern in PATTERNS[size]:
            position, cost = first, 0.0
            for width in pattern:
                cost += plain.transition(position, width)[2]
                position += width
            ranked.append((cost, pattern))
        cost, pattern = min(ranked)
        total += cost
        widths.extend(pattern)
        first += size
    rounds = len(widths) * (2 if plain.task['operator'] == 'all-reduce' else 1)
    if rounds > plain.task['max_nonempty_rounds']:
        return None
    return plain, widths, total


def fixed_band_guard(space):
    """Materialize the priced guard for tests and fallback validation."""
    projection = fixed_band_projection(space)
    return projection[0].compile(projection[1]) if projection else None


def validate_guard(task, graph, projection, evidence):
    """Materialize a priced fallback once, and require independent agreement."""
    candidate, predicted = projection[0].compile(projection[1])
    checked = replay(task, graph, candidate)
    matches = abs(checked['total_time_us'] - predicted) < 1e-4
    evidence.update(materialized=True, replay_valid=checked['valid'],
                    replay_us=checked['total_time_us'], preview_matches=matches)
    return (candidate, checked) if checked['valid'] and matches else (None, None)


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
                     for w in range(1, min(max(map(len, table)), i) + 1)]
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
        split = allowed[0] if task['operator'] == 'all-gather' else allowed[-1]
        preview = SeedPreview(task, graph, split)
        phases['construct_s'] = time.monotonic() - begin
        begin = time.monotonic()
        try:
            space, space_error = SearchSpace(task, graph, split), None
        except ValueError as error:
            space, space_error = None, str(error)
        phases['capacity_space_s'] = time.monotonic() - begin
        # Compare the finite old band family before spending time on another
        # constructor family. This is a measured work-allocation heuristic, not
        # a lower-bound proof. Retain the exact seed recipe, compile it only if
        # needed, and never report the preview as a successfully replayed answer.
        guard = {'selected': False, 'materialized': False}
        projection = fixed_band_projection(space) if space is not None else None
        begin = time.monotonic()
        seed_cost, exact = preview.screen(projection[2] / 0.65 if projection else float('inf'))
        phases['construct_s'] += time.monotonic() - begin
        solution, report = None, {'total_time_us': seed_cost}
        guard['seed_us' if exact else 'seed_lower_bound_us'] = seed_cost
        if projection is not None:
            guard['predicted_us'] = projection[2]
        skip_control = projection is not None and projection[2] <= seed_cost * 0.65
        guard.update(constructor_skipped=skip_control,
                     screen_ratio=0.65)
        search = {'candidates': [], 'initial_candidate_materialized': False}
        if not skip_control:
            seed = _owned_search(solve_schedule(
                task, graph, seed_only=True, seed_weight=4.0, seed_split=split,
                time_budget_s=search_budget_s,
            ))
            seed_report = seed['strategy'].pop('validation_report', None)
            if seed_report is None:
                seed_report = replay(task, graph, seed)
            solution, report = seed, seed_report
            raw, checked, elapsed = _control_candidate(
                task, graph, seed, search_budget_s=search_budget_s,
                refinement_budget_s=refinement_budget_s, continue_construction=True,
            )
            control, checked, capacity = _polished_control(
                task, graph, raw, checked, enabled=capacity_optimization,
            )
            phases.update(control_search_s=elapsed, control_capacity_s=capacity['elapsed_s'])
            search = control['strategy']
            search['initial_candidate_materialized'] = True
            if checked['total_time_us'] < report['total_time_us'] - 1e-6:
                solution, report = control, checked
                guard['selected'] = False
            if (space is not None and task['operator'] == 'all-gather'
                    and solution['split'] == space.profiles[0].plain.split):
                parent = report['total_time_us']
                shifted, predicted = pipeline(solution, parent, space.profiles[0].plain,
                                              task['max_nonempty_rounds'])
                if shifted is not solution:
                    verified = replay(task, graph, shifted)
                    accepted = (verified['valid'] and abs(verified['total_time_us'] - predicted) < 1e-4
                                and verified['total_time_us'] < parent)
                    search['incumbent_pipeline'] = {'parent_us': parent, 'predicted_us': predicted,
                                                    'accepted': accepted}
                    if accepted:
                        solution, report = shifted, verified
        else:
            phases.update(control_search_s=0.0, control_capacity_s=0.0)
        # If the existing guard already reaches the finite-space cost bound,
        # generated code cannot improve its communication cost in this space.
        # Validate that same guard while the request is still waiting on LLM;
        # do not compute a replacement path using the bound or skip model work.
        begin = time.monotonic()
        if (pending is not None and not pending.done() and projection is not None
                and projection[2] < report['total_time_us'] - 1e-6
                and space.audit_bound() >= projection[2] - 1e-6):
            candidate, checked = validate_guard(task, graph, projection, guard)
            if candidate is not None:
                solution, report = candidate, checked
                guard.update(selected=True, prepared_during_model_wait=True)
        phases['guard_prepare_s'] = time.monotonic() - begin
        control_us = min(report['total_time_us'], projection[2]) if projection else report['total_time_us']
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
                if not result['valid_program']:
                    result['decision'] = 'invalid_program'
                elif result['predicted_us'] < control_us - 1e-6:
                    solution, report = verify_proposal(task, graph, space, result, solution, report)
                else:
                    result['decision'] = 'guard_preview_not_better'
                if result['selected']:
                    guard['selected'] = False
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
                    'raw_response': interactions[-1].get('response_evidence'),
                    'cost_us': result.get('predicted_us'),
                    'finite_space_bound_us': result.get('bound_us'),
                    'dimensions': space.dimensions,
                    'row_lengths': [len(row) for row in space.costs[0]] + [0],
                    'instruction': ('The source includes the host wrapper. Return only a corrected frontier expression using runtime a.'
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

    begin = time.monotonic()
    if (projection is not None and not guard['materialized']
            and projection[2] < report['total_time_us'] - 1e-6):
        candidate, checked = validate_guard(task, graph, projection, guard)
        if candidate is not None and checked['total_time_us'] < report['total_time_us'] - 1e-6:
            solution, report = candidate, checked
            guard['selected'] = True
            for execution in executions:
                execution['selected'] = False
    phases['guard_s'] = time.monotonic() - begin
    if solution is None:
        begin = time.monotonic()
        solution, predicted = preview.compile([1] * preview.dimensions)
        report = replay(task, graph, solution)
        if not report['valid'] or abs(report['total_time_us']-predicted) > 1e-4:
            raise ValueError('protected seed failed independent Replay')
        phases['seed_fallback_s'] = time.monotonic() - begin
        search['initial_candidate_materialized'] = True
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
