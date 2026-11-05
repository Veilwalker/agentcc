"""Instance-bound retiming programs over the latest valid collective.

A live seed supplies measured capacity context. While the model chooses exact
retiming proportions, deterministic search constructs and refines its control.
Those proportions compile on current ready atoms, never stale packet indices.
The whole program survives only if independent Replay improves the control.
No executable model code runs.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
import time

from . import __version__
from .config import load_settings
from .retime_plan import prepare, execute
from .llm import edit_retiming
from .replay import replay
from .service import _control_candidate, _owned_search, _polished_control
from .solver import solve_schedule, validate_task
from .topology import Topology


def _model(task, problem, settings):
    try:
        from agenthack import llm_call
    except ImportError:
        trace = nullcontext(None)
    else:
        trace = llm_call(model=settings.model, provider="official-platform")
    with trace as call:
        proposal, evidence = edit_retiming(task, problem, settings=settings)
        if call is not None:
            usage = evidence.get("usage", {})
            call.set(input_tokens=usage.get("prompt_tokens", 0),
                     output_tokens=usage.get("completion_tokens", 0))
            call.result(proposal)
    return proposal, {**evidence, "called": True}


def solve_input(task_input, *, use_llm=True, search_budget_s=1.0,
                refinement_budget_s=0.25, capacity_optimization=True):
    started = time.monotonic()
    task, graph = task_input["task"], Topology.from_data(task_input["topology"])
    allowed = validate_task(task, graph)
    if task_input.get("instance_id", task.get("instance_id")) != task.get("instance_id"):
        raise ValueError("instance_id does not match task")
    begin = time.monotonic()
    baseline = _owned_search(solve_schedule(
        task, graph, time_budget_s=search_budget_s,
        seed_split=allowed[0] if task["operator"] == "all-gather" else allowed[-1],
        seed_weight=4.0, seed_only=True))
    baseline_report = baseline["strategy"].pop("validation_report", None)
    if baseline_report is None:
        baseline_report = replay(task, graph, baseline)
    phases = {"construct_s": time.monotonic() - begin}
    metadata = {"called": False, "mode": "offline-test"}
    if use_llm:
        problem, context, extraction = prepare(task, graph, baseline, baseline_report)
        phases["edit_problem_s"] = extraction["elapsed_s"]
        settings = load_settings()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentcc-forward") as pool:
        pending = pool.submit(copy_context().run, _model, task, problem, settings) if use_llm else None
        raw, report, elapsed = _control_candidate(task, graph, baseline,
            search_budget_s=search_budget_s, refinement_budget_s=refinement_budget_s,
            continue_construction=True)
        control, report, capacity = _polished_control(task, graph, raw, report,
                                                     enabled=capacity_optimization)
        wait_started = time.monotonic()
        if pending is not None:
            proposal, metadata = pending.result()
        phases["uncovered_model_wait_s"] = time.monotonic() - wait_started
    phases.update(control_search_s=elapsed, control_capacity_s=capacity["elapsed_s"],
                  control_s=elapsed + capacity["elapsed_s"])
    solution = control
    if use_llm:
        begin = time.monotonic()
        solution, evidence = execute(task, graph, control, report, context, proposal)
        report = evidence.pop("_validation_report", None) or report
        evidence.update(source_input_sha256=context.input_hash,
                        prompt_base_sha256=context.seed_hash, extraction=extraction)
        search = control["strategy"]
        if evidence["selected"]:
            parent = next(i for i, record in enumerate(search["candidates"]) if record["selected"])
            search["candidates"][parent]["selected"] = False
            search["candidates"].append({"source": "model_forwarding", "parent_candidate": parent,
                "actual": {**search["candidates"][parent]["actual"], "retiming_counts": evidence["compiled_counts"]},
                "valid": True, "selected": True, "total_time_us": report["total_time_us"]})
        search["model_edit"] = evidence
        metadata.update(schedule_stage="seed_bound_retiming_program", source_input_sha256=context.input_hash)
        phases["edit_validation_s"] = time.monotonic() - begin
    return {"answer": {k: v for k, v in solution.items() if k != "strategy"},
            "diagnostics": {"version": __version__, "llm": metadata,
                "search": control["strategy"], "local_replay": report,
                "phases_s": {k: round(v, 6) for k, v in phases.items()},
                "elapsed_s": round(time.monotonic() - started, 6)}}


async def solve_async(task_input):
    return await asyncio.to_thread(solve_input, task_input)
