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

from . import __version__, access_forest
from .fabric_routes import RouteCatalog
from .flow_search import FlowSearch, record as record_flow
from .capacity_backend import BACKEND
from .cohort_pipeline import pipeline
from .code_policy import propose
from .config import ConfigurationError, load_settings
from .final_refinement import improve
from .llm import LLMError
from .program_vm import Program, ProgramError
from .replay import certify as replay
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
    matches = checked['valid'] and abs(checked['total_time_us'] - predicted) < 1e-4
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


def verify_proposal(task, graph, space, execution, incumbent, report, *, refine=False, allowed=()):
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
        if refine:
            # The priced parent is provisional. Only a fully replayed child
            # may bypass the parent's independent Replay.
            candidate, checked, audit = improve(task, graph, candidate,
                {'total_time_us': predicted}, space.profiles[0].plain.catalog, allowed)
            execution['refinement'] = audit
            if audit['selected'] and checked['total_time_us'] < report['total_time_us'] - 1e-6:
                execution.update(selected=True, decision='verified_refined_proposal',
                                 replay_valid=True, replay_errors=[], replay_us=checked['total_time_us'])
                return candidate, checked
        checked = replay(task, graph, candidate)
        execution.update(
            replay_valid=checked['valid'], replay_errors=checked.get('errors', []),
            replay_us=checked['total_time_us'],
        )
        matches = checked['valid'] and abs(checked['total_time_us'] - predicted) < 1e-4
        if checked['valid'] and matches and checked['total_time_us'] < report['total_time_us'] - 1e-6:
            execution.update(selected=True, decision='verified_improvement')
            return candidate, checked
        execution['decision'] = 'replay_rejected_or_not_better'
    except (ValueError, TypeError, KeyError) as error:
        execution.update(decision='compile_rejected', error=str(error))
    finally:
        execution['validation_s'] = time.monotonic() - started
    return incumbent, report


def prepare_gather_option(task, graph, space, allowed, ceiling, incumbent, report):
    """Compile one structural option during I/O. Its use still requires a matching decision."""
    if len(graph.ranks) < 64 or space.audit_bound() >= ceiling - 1e-6:
        return None
    output = space.best_legal_path()
    if output is None:
        return None
    cost = space.evaluate(output)[2]
    audit = dict(valid_program=True, selected=False, predicted_us=cost, output=output)
    candidate, checked = verify_proposal(task, graph, space, audit, incumbent, report,
                                         refine=True, allowed=allowed)
    if audit['selected']:
        return output, candidate, checked, audit
    return None


def verify_choice(task, graph, space, execution, incumbent, report, prepared, refine, allowed):
    if prepared is not None and execution['output'] == prepared[0]:
        _, candidate, checked, audit = prepared
        if checked['total_time_us'] < report['total_time_us'] - 1e-6:
            execution.update({k: v for k, v in audit.items()
                              if k not in ('valid_program', 'output', 'predicted_us')})
            execution['prepared_during_model_wait'] = True
            return candidate, checked
    return verify_proposal(task, graph, space, execution, incumbent, report,
                           refine=refine, allowed=allowed)


def solve_input(task_input, *, use_llm=True, search_budget_s=1.0,
                refinement_budget_s=0.25, capacity_optimization=True, _stage_guard=False):
    started = time.monotonic()
    if not isinstance(task_input, dict):
        raise ValueError('input must be an object')
    task, graph = task_input['task'], Topology.from_data(task_input['topology'])
    allowed = validate_task(task, graph)
    if task_input.get('instance_id', task.get('instance_id')) != task.get('instance_id'):
        raise ValueError('instance_id does not match task')

    observed = []
    try:
        return _solve_validated(task_input, task, graph, allowed, started, observed, use_llm,
                                search_budget_s, refinement_budget_s, capacity_optimization, _stage_guard)
    except ConfigurationError:
        raise
    except (ValueError, IndexError, KeyError, ArithmeticError, TimeoutError) as error:
        # Recover only after input validation. A failed optional constructor may
        # already have issued a model call; reuse that call's actual evidence.
        from .replay import replay as reference_replay
        split = allowed[0] if task['operator'] == 'all-gather' else allowed[-1]
        seed = solve_schedule(task, graph, seed_only=True, seed_split=split, seed_weight=4.0)
        answer = {k: v for k, v in seed.items() if k != 'strategy'}
        checked = reference_replay(task, graph, answer)
        if not checked['valid']:
            raise ValueError('recovery seed failed verification') from error
        answer['metrics'] = dict(total_time=checked['total_time_us'] / 1e6,
                                 round_times=[v / 1e6 for v in checked['round_times_us']])
        diagnostics = dict(version=__version__, capacity_backend=BACKEND, local_replay=checked,
            elapsed_s=time.monotonic()-started, phases_s={}, llm=model_evidence(observed),
            search=dict(candidates=[dict(source='recovery_seed', selected=True, valid=True,
                                         total_time_us=checked['total_time_us'])],
                        recovery_error=type(error).__name__,
                        model_edit=dict(selected=False, accepted=False, executions=[],
                                        final_time_us=checked['total_time_us'], stop_reason='recovery_seed')))
        return dict(answer=answer, diagnostics=diagnostics)


def model_evidence(interactions):
    return {
        'called': bool(interactions), 'mode': 'generated_solver', 'calls': len(interactions),
        'duration_ms': sum(item.get('duration_ms', 0) for item in interactions),
        'usage': {key: sum(item.get('usage', {}).get(key, 0) for item in interactions)
                  for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'interactions': interactions,
    }


def _solve_validated(task_input, task, graph, allowed, started, observed, use_llm,
                     search_budget_s, refinement_budget_s, capacity_optimization, _stage_guard):
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

    groups = None if _stage_guard else access_forest.heterogeneous_groups(task, graph, allowed)
    problem = instance_context(task, graph)
    if groups and capacity_optimization:
        problem['access_domains'] = [len(g) for g in groups]
    phases, interactions, executions = {}, [], []
    settings = load_settings() if use_llm else None
    source, stop_reason = None, 'offline_test'
    def consult():
        try:
            result = propose(problem, settings)
        except LLMError as error:
            observed.append(dict(called=True, error=str(error), usage_unavailable=True,
                                 duration_ms=getattr(error, 'duration_ms', 0)))
            raise
        observed.append(result[1])
        return result

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='agentcc-code') as pool:
        pending = pool.submit(copy_context().run, consult) if use_llm else None
        if groups and capacity_optimization:
            # Shared endpoint pools can couple otherwise local fanouts. Compare
            # a complete ordinary stage schedule on the same input in this case.
            rank_set = set(graph.ranks)
            shared_access = any(bindings and (src in rank_set or dst in rank_set)
                                for (src, dst), bindings in graph.edge_resources.items())
            competitor = (solve_input(task_input, use_llm=False, search_budget_s=search_budget_s,
                                      refinement_budget_s=refinement_budget_s, _stage_guard=True)
                          if shared_access else None)
            domain_flow = FlowSearch(task, graph, RouteCatalog(graph, task['data_size']['per_rank_bytes']
                                    // (allowed[0] if task['operator']=='all-gather' else allowed[-1])))
            stage = (competitor['answer'], competitor['diagnostics']['local_replay']) if competitor else None
            result = access_forest.solve_domains(task, graph, groups, allowed, pending, started,
                                                 flow=domain_flow, competitor=stage)
            if result is not None:
                if competitor is not None:
                    local = result['diagnostics']
                    alternative = competitor['diagnostics']['local_replay']
                    local['search']['stage_guard_us'] = alternative['total_time_us']
                    if alternative['total_time_us'] < local['local_replay']['total_time_us']:
                        local['local_replay'] = alternative
                        local['search']['access_forest']['selected'] = False
                        local['search']['model_edit'].update(selected=False, accepted=False,
                            final_time_us=alternative['total_time_us'], stop_reason='stage_guard_won')
                        for trial in local['search']['candidates']:
                            trial['selected'] = False
                        for execution in local['search']['model_edit'].get('executions', []):
                            execution['selected'] = False
                        local['search']['candidates'].append(dict(source='stage_guard', selected=True,
                            valid=True, total_time_us=alternative['total_time_us']))
                        result['answer'] = competitor['answer']
                local = result['diagnostics']
                plan = result['answer']
                plan, checked, audit = domain_flow.finish(plan, local['local_replay'])
                record_flow(local['search'], checked, audit)
                plan['metrics'] = dict(total_time=checked['total_time_us'] / 1e6,
                                       round_times=[v / 1e6 for v in checked['round_times_us']])
                result['answer'], local['local_replay'] = plan, checked
                local['elapsed_s'] = time.monotonic() - started
                return result
        begin = time.monotonic()
        split = allowed[0] if task['operator'] == 'all-gather' else allowed[-1]
        phases['construct_s'] = time.monotonic() - begin
        begin = time.monotonic()
        try:
            space, space_error = SearchSpace(task, graph, split), None
        except ValueError as error:
            space, space_error = None, str(error)
        phases['capacity_space_s'] = time.monotonic() - begin
        preview = SeedPreview(task, graph, split, catalog=space.profiles[0].plain.catalog if space else None)
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
        if space is not None and exact and task['operator'] == 'all-reduce':
            skip_control |= space.audit_bound() < seed_cost * 0.67
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
        early_plan, early_audit = None, None
        if (capacity_optimization and solution is not None and pending is not None
                and not pending.done() and space is not None
                and space.audit_bound() >= report['total_time_us'] - 1e-6
                and control_us >= report['total_time_us'] - 1e-6):
            before = time.monotonic()
            solution, report, early_audit = improve(
                task, graph, solution, report, space.profiles[0].plain.catalog, allowed)
            early_plan = solution
            phases['overlapped_refinement_s'] = time.monotonic() - before
        prepared = None
        if (capacity_optimization and space is not None and pending is not None
                and not pending.done() and early_plan is None):
            before = time.monotonic()
            prepared = prepare_gather_option(task, graph, space, allowed, control_us, solution, report)
            phases['overlapped_compile_s'] = time.monotonic() - before
        flow = (FlowSearch(task, graph, space.profiles[0].plain.catalog)
                if capacity_optimization and space is not None and not _stage_guard else None)
        if flow is not None and pending is not None and not pending.done():
            if prepared is not None:
                flow.finish(prepared[1], prepared[2])
            elif early_plan is not None:
                flow.finish(solution, report)
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
            protocol = interactions[-1].get('program_protocol', 'outgoing_v1')
            result = execute_program(source, space, protocol)
            executions.append(result)
            if not result['valid_program']:
                result['decision'] = 'invalid_program'
            elif result['predicted_us'] < control_us - 1e-6:
                solution, report = verify_choice(task, graph, space, result, solution, report,
                    prepared, capacity_optimization, allowed)
            else:
                result['decision'] = 'guard_preview_not_better'
            if result['selected']:
                guard['selected'] = False
            tolerance = max(1e-6, result.get('bound_us', 0.0) * 0.001)
            if not result['valid_program'] or result.get('gap_us', 0.0) > tolerance:
                stop_reason = 'single_call_local_fallback'
                output = space.best_legal_path()
                cost = space.evaluate(output)[2] if output else float('inf')
                fallback = {'valid_program': True, 'selected': False,
                            'origin': 'local_fallback', 'output': output, 'predicted_us': cost if output else None}
                guard['local_fallback'] = fallback
                if cost < min(report['total_time_us'], control_us) - 1e-6:
                    solution, report = verify_choice(task, graph, space, fallback, solution, report,
                        prepared, capacity_optimization, allowed)
                    if fallback['selected']:
                        result['selected'] = False
                        guard['selected'] = False
            else:
                stop_reason = 'finite_space_gap_within_tolerance'
        elif _stage_guard and space is not None:
            output = space.best_legal_path()
            cost = space.evaluate(output)[2] if output else float('inf')
            fallback = dict(valid_program=bool(output), selected=False, origin='local_fallback',
                            output=output, predicted_us=cost if output else None)
            guard['local_fallback'] = fallback
            if cost < min(report['total_time_us'], control_us) - 1e-6:
                solution, report = verify_choice(task, graph, space, fallback, solution, report,
                                                 prepared, capacity_optimization, allowed)
        elif space is None:
            stop_reason = 'capacity_space_unavailable'

    begin = time.monotonic()
    if (projection is not None and not guard['materialized']
            and projection[2] < report['total_time_us'] - 1e-6):
        candidate, checked = validate_guard(task, graph, projection, guard)
        if candidate is not None and checked['total_time_us'] < report['total_time_us'] - 1e-6:
            solution, report = candidate, checked
            guard['selected'] = True
            if 'local_fallback' in guard:
                guard['local_fallback']['selected'] = False
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
    checked_edit = next((e['refinement'] for e in [*executions, guard.get('local_fallback', {})]
                         if e.get('selected') and 'refinement' in e), None)
    if checked_edit is not None:
        search['final_refinement'] = checked_edit
    elif early_plan is solution and early_audit is not None:
        search['final_refinement'] = early_audit
    elif space is not None and capacity_optimization:
        begin = time.monotonic()
        solution, report, audit = improve(task, graph, solution, report, space.profiles[0].plain.catalog, allowed)
        search['final_refinement'] = audit
        phases['final_refinement_s'] = time.monotonic() - begin
    model_selected = any(execution['selected'] for execution in executions)
    search['model_edit'] = {
        'selected': model_selected, 'accepted': model_selected,
        'semantics': 'model_generated_stage_solver_v1',
        'control_time_us': control_us, 'final_time_us': report['total_time_us'],
        'executions': executions, 'guard': guard, 'stop_reason': stop_reason,
        'space': space.evidence() if space else {'unavailable': space_error},
    }
    recovered = guard.get('local_fallback', {}).get('selected', False)
    if guard['selected'] or model_selected or recovered:
        for candidate in search['candidates']:
            candidate['selected'] = False
        search['candidates'].append({
            'source': ('model_generated_code' if model_selected else
                       'local_fallback' if recovered else 'fixed_band_guard'),
            'actual': (guard['local_fallback']['output'] if recovered else
                       next((e['output'] for e in executions if e['selected']), None)),
            'selected': True, 'valid': True, 'total_time_us': report['total_time_us'],
        })
    if flow is None and capacity_optimization and not _stage_guard and 64 <= count <= 256:
        flow = FlowSearch(task, graph, RouteCatalog(graph, task['data_size']['per_rank_bytes'] // solution['split']))
    if flow is not None:
        solution, report, audit = flow.finish(solution, report)
        record_flow(search, report, audit)
    elif _stage_guard and space is not None and task['operator'] == 'all-reduce':
        from .threshold_routes import improve as adjust_paths
        solution, report, _ = adjust_paths(task, graph, solution, report, space.profiles[0].plain.catalog)
        search['model_edit']['final_time_us'] = report['total_time_us']
    metadata = model_evidence(interactions)
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
