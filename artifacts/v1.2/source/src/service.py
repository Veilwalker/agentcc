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
from .schedule_edit import apply_proposal, extract_problem
from .solver import solve_schedule, validate_task
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


def solve_input(task_input: dict, *, use_llm: bool = True,
                search_budget_s: float = 3.0, refinement_budget_s: float = 0.25) -> dict:
    """Return the SDK's output object; offline mode is an explicit test facility."""
    started = time.monotonic()
    if not isinstance(task_input, dict):
        raise ValueError("input must be an object")
    task, topology = task_input["task"], task_input["topology"]
    graph = Topology.from_data(topology)
    validate_task(task, graph)
    if task_input.get("instance_id", task.get("instance_id")) != task.get("instance_id"):
        raise ValueError("instance_id does not match task")
    phases = {}
    phase_started = time.monotonic()
    baseline = solve_schedule(task, graph, time_budget_s=search_budget_s)
    phases["construct_s"] = time.monotonic() - phase_started
    model_metadata = {"called": False, "mode": "offline-test"}
    if use_llm:
        settings = load_settings()
        source_digest = hashlib.sha256(json.dumps(
            {"task": task, "topology": topology}, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        phase_started = time.monotonic()
        problem, context = extract_problem(task, graph, baseline, time_budget_s=0.5)
        phases["edit_problem_s"] = time.monotonic() - phase_started
        # One real call edits a frozen schedule. Its network wait overlaps a
        # deterministic control; the control must never mutate the model base.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentcc-edit") as pool:
            pending = pool.submit(copy_context().run, _edit_with_trace, task, problem, settings)
            solution = _refined_candidate(task, graph, baseline)
            proposal, model_metadata = pending.result()
        model_metadata["source_input_sha256"] = source_digest
        phase_started = time.monotonic()
        edited, evidence = apply_proposal(task, graph, baseline, context, proposal)
        control_time = solution["metrics"]["total_time"] * 1e6
        selected_rounds = set()
        if evidence.get("accepted"):
            # Replay metrics omit empty rounds; edit/frozen indices refer to
            # the original round array and must retain those positions.
            nonempty = [ri for ri, row in enumerate(solution["rounds"]) if row["transfers"]]
            for ri, old, new in zip(nonempty, solution["metrics"]["round_times"],
                                    edited["metrics"]["round_times"]):
                old_us, new_us = old * 1e6, new * 1e6
                if new_us < old_us - max(1e-6, old_us * 1e-10):
                    selected_rounds.add(ri)
        # Both alternatives preserve all round-start data states. Compare
        # complete rounds so a useful model edit is not lost merely because
        # the independent control improved some other round more strongly.
        merged, merged_report = None, None
        if selected_rounds:
            merged = {**solution, "rounds": [edited["rounds"][ri] if ri in selected_rounds else row
                                               for ri, row in enumerate(solution["rounds"])]}
            merged_report = replay(task, graph, merged)
            if not merged_report["valid"]:
                evidence["errors"].append("combined model/control rounds failed replay")
                selected_rounds.clear()
        selected = bool(selected_rounds)
        group_rounds = {group["id"]: group["round"] - 1 for group in problem["groups"]}
        selected_edits = [row for row in evidence["applied_edits"] if group_rounds[row[0]] in selected_rounds]
        evidence.update(selected=selected, selected_rounds=[ri + 1 for ri in sorted(selected_rounds)],
                        selected_edits=selected_edits,
                        selected_frozen_transfers=[pair for pair in evidence.get("frozen_transfers", [])
                                                   if pair[0] in selected_rounds],
                        selected_edit_manifest=[row for row in evidence.get("edit_manifest", [])
                                                if row[0] - 1 in selected_rounds], control_time_us=control_time,
                        preservation_scope="moved_prefixes")
        search = solution["strategy"]
        search["model_edit"] = evidence
        if selected:
            parent = next(i for i, record in enumerate(baseline["strategy"]["candidates"]) if record["selected"])
            control_parent = next(i for i, record in enumerate(search["candidates"]) if record["selected"])
            for record in search["candidates"]:
                record["selected"] = False
            selected_time = merged_report["total_time_us"]
            evidence["selected_time_us"] = selected_time
            merged["metrics"] = {"total_time": selected_time / 1e6,
                                 "round_times": [value / 1e6 for value in merged_report["round_times_us"]]}
            search["candidates"].append({
                "source": "model_edit", "parent_candidate": parent,
                "control_candidate": control_parent,
                "actual": {"edit_base": problem["base"], "edits": selected_edits},
                "valid": True, "selected": True, "total_time_us": selected_time,
            })
            solution = {**merged, "strategy": search}
        phases["edit_validation_s"] = time.monotonic() - phase_started
    else:
        solution = _refined_candidate(task, graph, baseline)
    # Preserve the exact moved prefixes proposed by the model. Remaining
    # transfers may use additional legal paths without undoing those actions.
    phase_started = time.monotonic()
    search = solution["strategy"]
    edit = search.get("model_edit", {})
    frozen = edit.get("selected_frozen_transfers", []) if edit.get("selected") else []
    try:
        finished, details = refine_broad(task, graph, solution, frozen=frozen,
                                         time_budget_s=refinement_budget_s)
        # This is the complete semantic Replay of the returned schedule, not
        # just a path-cost estimate. Private search metadata changes no data.
        report = details.pop("final_replay")
        search["route_finish"] = details
        if details["accepted"]:
            parent = next(i for i, record in enumerate(search["candidates"]) if record["selected"])
            previous = search["candidates"][parent]
            previous["selected"] = False
            search["candidates"].append({
                "source": "route_finish", "parent_candidate": parent,
                "actual": {**previous["actual"], "fixed_model_transfers": len(frozen)},
                "valid": True, "selected": True, "total_time_us": report["total_time_us"],
            })
            solution = {**finished, "strategy": search}
        if edit:
            edit["preserved_in_final"] = bool(edit.get("selected") and details["frozen_preserved"])
            edit["final_time_us"] = report["total_time_us"]
    except ValueError as exc:
        search["route_finish"] = {"accepted": False, "error": str(exc)}
        report = replay(task, graph, solution)
        if edit:
            edit["preserved_in_final"] = bool(edit.get("selected"))
            edit["final_time_us"] = report["total_time_us"]
    phases["route_finish_s"] = time.monotonic() - phase_started
    if not report["valid"]:
        raise ValueError("final replay validation failed")
    search_evidence = solution.pop("strategy", {})
    return {
        "answer": solution,
        "diagnostics": {
            "version": __version__,
            "llm": model_metadata,
            "search": search_evidence,
            "local_replay": report,
            "phases_s": {key: round(value, 6) for key, value in phases.items()},
            "elapsed_s": round(time.monotonic() - started, 6),
        },
    }


async def solve_async(task_input: dict) -> dict:
    # Yield to the SDK event loop: its timeout can then interrupt the request.
    # CPU search also has an internal budget, so a timed-out worker terminates.
    return await asyncio.to_thread(solve_input, task_input)
