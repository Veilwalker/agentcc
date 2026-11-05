"""Application boundary shared by the official SDK and local HTTP testing."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import hashlib
import json
import time

from .config import load_settings
from .llm import plan
from .planning import compact_summary
from .replay import replay
from .route_balance import refine
from .solver import solve_schedule, validate_task
from .topology import Topology


def _plan_with_trace(task, summary, settings):
    """Keep SDK attribution and span duration attached to the real model call."""
    try:
        from agenthack import llm_call
    except ImportError:
        model_trace = nullcontext(None)
    else:
        model_trace = llm_call(model=settings.model, provider="official-platform")
    with model_trace as call:
        strategy, metadata = plan(task, summary, settings=settings)
        if call is not None:
            usage = metadata.get("usage", {})
            call.set(input_tokens=usage.get("prompt_tokens", 0),
                     output_tokens=usage.get("completion_tokens", 0))
            call.result(strategy)
    return strategy, {**metadata, "called": True}


def solve_input(task_input: dict, *, use_llm: bool = True,
                search_budget_s: float = 3.0) -> dict:
    """Return the SDK's output object; offline mode is an explicit test facility."""
    started = time.monotonic()
    if not isinstance(task_input, dict):
        raise ValueError("input must be an object")
    task, topology = task_input["task"], task_input["topology"]
    graph = Topology.from_data(topology)
    validate_task(task, graph)
    if task_input.get("instance_id", task.get("instance_id")) != task.get("instance_id"):
        raise ValueError("instance_id does not match task")
    strategy, model_metadata = None, {"called": False, "mode": "offline-test"}
    preliminary = None
    if use_llm:
        settings = load_settings()
        summary = compact_summary(graph)
        summary["input_sha256"] = hashlib.sha256(json.dumps(
            {"task": task, "topology": topology}, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        # Network waiting overlaps CPU construction. Context propagation keeps
        # this real call attached to the SDK request even in the worker thread.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentcc-plan") as pool:
            pending = pool.submit(copy_context().run, _plan_with_trace, task, summary, settings)
            try:
                preliminary = solve_schedule(task, graph, time_budget_s=min(1.5, search_budget_s))
            except ValueError:
                preliminary = None
            strategy, model_metadata = pending.result()
    solution = solve_schedule(task, graph, strategy=strategy, time_budget_s=search_budget_s,
                              incumbent=preliminary)
    search = solution["strategy"]
    varied = search.get("route_variation", False)
    if varied:
        weights = sorted({record["actual"]["resource_weight"] for record in search["candidates"]
                          if record.get("valid")})
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
    report = replay(task, graph, solution)
    if not report["valid"]:
        raise ValueError("final replay validation failed")
    search_evidence = solution.pop("strategy", {})
    return {
        "answer": solution,
        "diagnostics": {
            "llm": model_metadata,
            "search": search_evidence,
            "local_replay": report,
            "elapsed_s": round(time.monotonic() - started, 6),
        },
    }


async def solve_async(task_input: dict) -> dict:
    # Yield to the SDK event loop: its timeout can then interrupt the request.
    # CPU search also has an internal budget, so a timed-out worker terminates.
    return await asyncio.to_thread(solve_input, task_input)
