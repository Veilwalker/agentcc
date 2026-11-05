"""Application boundary shared by the official SDK and local HTTP testing."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextvars import copy_context
import hashlib
import json
import time

from .config import load_settings
from . import __version__
from .llm import edit_schedule
from .replay import replay
from .route_balance import refine
from .route_broad import refine_broad
from .route_work import RouteWorkBudget
from .schedule_edit import apply_proposal, extract_problem
from .solver import solve_schedule, validate_task
from .selection import choose_model_candidate
from .topology import Topology


def _edit_with_trace(task, problem, settings):
    """Keep SDK attribution and span duration attached to the real model call."""
    try:
        from agenthack import llm_call
    except ImportError:
        model_trace = nullcontext(None)
    else:
        model_trace = llm_call(model=settings.model, provider="official-platform")
    with model_trace as call:
        proposal, metadata = edit_schedule(task, problem, settings=settings)
        if call is not None:
            usage = metadata.get("usage", {})
            call.set(input_tokens=usage.get("prompt_tokens", 0),
                     output_tokens=usage.get("completion_tokens", 0))
            call.result(proposal)
    return proposal, {**metadata, "called": True}


def _refined_candidate(task, graph, baseline):
    """Run the deterministic control without changing the model's frozen base."""
    solution = {**baseline, "strategy": deepcopy(baseline["strategy"])}
    search = solution["strategy"]
    if not search.get("route_variation", False):
        return solution
    weights = sorted(set(search.get("constructed_route_weights", [])) | {
        record["actual"]["resource_weight"] for record in search["candidates"] if record.get("valid")})
    try:
        improved, details = refine(task, graph, solution, weights=weights,
                                   routing_bytes=search["routing_bytes"], time_budget_s=0.25)
        search["route_refinement"] = details
        if details["improved_rounds"]:
            selected_index = next(i for i, record in enumerate(search["candidates"]) if record["selected"])
            previous = search["candidates"][selected_index]
            previous["selected"] = False
            search["candidates"].append({
                "source": "route_refinement", "parent_candidate": selected_index,
                "actual": {**previous["actual"], "route_weights": weights},
                "valid": True, "selected": True, "total_time_us": details["after_us"],
            })
            solution = improved
    except ValueError as exc:
        search["route_refinement"] = {"accepted": False, "error": str(exc)}
    return solution


def _owned_search(solution):
    """Detach mutable search history while keeping validated schedule data shared."""
    return {**solution, "strategy": deepcopy(solution.get("strategy", {}))}


def _control_candidate(task, graph, baseline, *, search_budget_s, refinement_budget_s):
    """Continue construction and unrestricted routing while the model is waiting."""
    started = time.monotonic()
    try:
        solution = _owned_search(solve_schedule(
            task, graph, time_budget_s=search_budget_s, incumbent=baseline))
    except (ValueError, TimeoutError) as exc:
        solution = _owned_search(baseline)
        solution["strategy"]["resume_error"] = str(exc)
    solution["strategy"].pop("validation_report", None)
    solution = _refined_candidate(task, graph, solution)
    search = solution["strategy"]
    try:
        finished, details = refine_broad(
            task, graph, solution, frozen=[], time_budget_s=refinement_budget_s,
            work_budget=RouteWorkBudget(20_000, 65_000, safety_s=3.0))
        report = details.pop("final_replay")
        search["route_finish"] = details
        if details["accepted"]:
            parent = next(i for i, record in enumerate(search["candidates"]) if record["selected"])
            previous = search["candidates"][parent]
            previous["selected"] = False
            search["candidates"].append({
                "source": "route_finish", "parent_candidate": parent,
                "actual": {**previous["actual"], "fixed_model_transfers": 0},
                "valid": True, "selected": True, "total_time_us": report["total_time_us"],
            })
            solution = {**finished, "strategy": search}
    except ValueError as exc:
        search["route_finish"] = {"accepted": False, "error": str(exc)}
        report = replay(task, graph, solution)
    if not report["valid"]:
        raise ValueError("deterministic control failed replay")
    return solution, report, time.monotonic() - started


def solve_input(task_input: dict, *, use_llm: bool = True,
                search_budget_s: float = 1.0, refinement_budget_s: float = 0.25) -> dict:
    """Build a seed, overlap model/CPU work, then compare complete valid answers."""
    started = time.monotonic()
    if not isinstance(task_input, dict):
        raise ValueError("input must be an object")
    task, topology = task_input["task"], task_input["topology"]
    graph = Topology.from_data(topology)
    allowed = validate_task(task, graph)
    if task_input.get("instance_id", task.get("instance_id")) != task.get("instance_id"):
        raise ValueError("instance_id does not match task")
    phases = {}
    phase_started = time.monotonic()
    # A full-size reduction seed exposes useful slots to the model immediately.
    # The constructor retains its protected fallback for unusual rank graphs.
    seed_split = allowed[0] if task["operator"] == "all-gather" else allowed[-1]
    baseline = _owned_search(solve_schedule(task, graph, time_budget_s=search_budget_s,
                                           seed_only=True, seed_split=seed_split, seed_weight=4.0))
    baseline_report = baseline["strategy"].pop("validation_report", None)
    if baseline_report is None:
        baseline_report = replay(task, graph, baseline)
    if not baseline_report["valid"]:
        raise ValueError("initial schedule failed replay")
    phases["construct_s"] = time.monotonic() - phase_started
    model_metadata = {"called": False, "mode": "offline-test"}
    if use_llm:
        settings = load_settings()
        source_digest = hashlib.sha256(json.dumps(
            {"task": task, "topology": topology}, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        phase_started = time.monotonic()
        problem, context = extract_problem(task, graph, baseline, time_budget_s=0.2,
                                           max_groups=2, max_rounds=1,
                                           validated_report=baseline_report,
                                           work_budget=RouteWorkBudget(10_000, 65_000, safety_s=3.0))
        phases["edit_problem_s"] = time.monotonic() - phase_started
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentcc-edit") as pool:
            pending = pool.submit(copy_context().run, _edit_with_trace, task, problem, settings)
            control, control_report, control_elapsed = _control_candidate(
                task, graph, baseline, search_budget_s=search_budget_s,
                refinement_budget_s=refinement_budget_s)
            proposal, model_metadata = pending.result()
        phases["control_s"] = control_elapsed
        model_metadata["source_input_sha256"] = source_digest
        phase_started = time.monotonic()
        edited, evidence = apply_proposal(task, graph, baseline, context, proposal)
        edited_report = evidence.pop("_validation_report", None)
        if evidence.get("accepted") and edited_report is None:
            edited_report = replay(task, graph, edited)
        if edited_report is not None and not edited_report["valid"]:
            raise ValueError("accepted model candidate failed replay")
        solution, report, selection = choose_model_candidate(
            task, graph, control, control_report, edited, edited_report,
            evidence.get("accepted_rounds", []), baseline=baseline,
            overlay_min_relative_gain=0.001)
        search = control["strategy"]
        search["edit_extraction_work"] = context.extraction_work
        selected_rounds = set(selection["selected_rounds"])
        group_rounds = {group["id"]: group["round"] for group in problem["groups"]}
        selected_edits = [row for row in evidence["applied_edits"]
                          if group_rounds[row[0]] in selected_rounds]
        # Atomic overlays repartition control packets. Their observed final
        # positions must come from the selector, never from the seed's packets.
        selected_positions = selection.get("selected_transfers", [
            pair for pair in evidence.get("frozen_transfers", []) if pair[0] + 1 in selected_rounds])
        selected_manifest = selection.get("model_atomic_manifest", [
            row for row in evidence.get("edit_manifest", []) if row[0] in selected_rounds])
        evidence["errors"].extend(selection.pop("errors"))
        evidence.update(selection, selected_edits=selected_edits,
                        selected_frozen_transfers=[], preservation_scope="observed_not_enforced",
                        selected_transfers=selected_positions,
                        selected_edit_manifest=selected_manifest,
                        selected_manifest_format=("atomic_paths_v1" if selection["combination"] == "atomic_overlay"
                                                  else "compiled_prefixes_v1"),
                        preserved_in_final=selection["selected"],
                        selected_time_us=report["total_time_us"], final_time_us=report["total_time_us"])
        if selection["selected"]:
            control_parent = next(i for i, record in enumerate(search["candidates"]) if record["selected"])
            control_params = search["candidates"][control_parent]["actual"]
            seed_record = next(record for record in baseline["strategy"]["candidates"] if record["selected"])
            seed_parent = next((i for i, record in enumerate(search["candidates"])
                                if record["actual"] == seed_record["actual"]
                                and record.get("total_time_us") == baseline_report["total_time_us"]), None)
            if seed_parent is None:
                seed_parent = len(search["candidates"])
                search["candidates"].append({**deepcopy(seed_record), "source": "edit_seed", "selected": False})
            search["candidates"][control_parent]["selected"] = False
            program = seed_record["actual"] if selection["combination"] == "whole_model" else control_params
            search["algorithm"] = program.get("algorithm", search.get("algorithm"))
            search["candidates"].append({
                "source": "model_edit", "parent_candidate": seed_parent,
                "control_candidate": control_parent,
                "actual": {**program, "edit_base": problem["base"], "edits": selected_edits},
                "valid": True, "selected": True, "total_time_us": report["total_time_us"],
            })
        search["model_edit"] = evidence
        solution = {**solution, "strategy": search}
        phases["edit_validation_s"] = time.monotonic() - phase_started
    else:
        solution, report, phases["control_s"] = _control_candidate(
            task, graph, baseline, search_budget_s=search_budget_s,
            refinement_budget_s=refinement_budget_s)
    answer = {key: value for key, value in solution.items() if key != "strategy"}
    return {
        "answer": answer,
        "diagnostics": {
            "version": __version__, "llm": model_metadata,
            "search": solution["strategy"], "local_replay": report,
            "phases_s": {key: round(value, 6) for key, value in phases.items()},
            "elapsed_s": round(time.monotonic() - started, 6),
        },
    }


async def solve_async(task_input: dict) -> dict:
    # Yield to the SDK event loop: its timeout can then interrupt the request.
    # CPU search also has an internal budget, so a timed-out worker terminates.
    return await asyncio.to_thread(solve_input, task_input)
