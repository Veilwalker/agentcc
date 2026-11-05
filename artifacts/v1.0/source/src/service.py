"""Application boundary shared by the official SDK and local HTTP testing."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
import hashlib
import json
import time

from .config import load_settings
from .llm import plan
from .replay import replay
from .solver import solve_schedule, validate_task
from .topology import Topology


def solve_input(task_input: dict, *, use_llm: bool = True,
                search_budget_s: float = 8.0) -> dict:
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
    if use_llm:
        settings = load_settings()
        summary = graph.summary()
        summary["input_sha256"] = hashlib.sha256(json.dumps(
            {"task": task, "topology": topology}, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        try:
            from agenthack import llm_call
        except ImportError:
            model_trace = nullcontext(None)
        else:
            model_trace = llm_call(model=settings.model, provider="official-platform")
        with model_trace as call:
            strategy, model_metadata = plan(task, summary, settings=settings)
            if call is not None:
                usage = model_metadata.get("usage", {})
                call.set(input_tokens=usage.get("prompt_tokens", 0),
                         output_tokens=usage.get("completion_tokens", 0))
                call.result(strategy)
        model_metadata = {**model_metadata, "called": True}
    solution = solve_schedule(task, graph, strategy=strategy, time_budget_s=search_budget_s)
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
