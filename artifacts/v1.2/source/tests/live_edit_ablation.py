"""Explicit real-SDK/model test with a deterministic SAME-catalogue control.

Not part of unittest discovery. The only request-time hook retains references
to extract_problem's unchanged result; it does not serialize, copy, optimize,
replace a model response, or alter production source. Capture files and the
deterministic control are produced after the measured HTTP request finishes.
"""

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

import httpx
from agenthack import AgentRuntime

from src.my_agent import MyAgent
from src.replay import replay
from src.schedule_edit import extract_problem as real_extract_problem
from tests.edit_ablation import optimize_counts


PROJECT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((PROJECT / "src").glob("*.py"))}


def local_path(relative):
    path = (PROJECT / relative).resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError("all ablation paths must remain inside the project")
    return path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


async def run(samples, output_dir, budget):
    output_dir = local_path(output_dir)
    initial_sources = source_hashes()
    report = {"scope": "Real official-SDK/model requests and exact same-catalogue deterministic controls",
              "limits": {"cpu_affinity": sorted(os.sched_getaffinity(0)),
                         "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0]},
              "source_sha256_at_start": initial_sources,
              "deterministic_budget_s": budget, "cases": []}
    runtime = AgentRuntime(MyAgent())
    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app),
                                     base_url="http://official-sdk-test") as client:
            if (await client.get("/health")).status_code != 200:
                raise RuntimeError("official SDK health check failed")
            for sample in samples:
                folder = local_path(f"materials/spec/examples/256-rank/{sample}")
                task = json.loads((folder / "task.json").read_text())
                topology = json.loads((folder / "topology.json").read_text())
                request = {"task_id": task["task_id"], "task_type": "agentcc",
                           "input": {"instance_id": task["instance_id"], "task": task, "topology": topology},
                           "constraints": {"timeout_ms": 180000}}
                capture = []

                def observe(actual_task, graph, baseline, **kwargs):
                    problem, context = real_extract_problem(actual_task, graph, baseline, **kwargs)
                    # Only references are retained inside the measured request.
                    capture.append((actual_task, graph, baseline, problem, context))
                    return problem, context

                with patch("src.service.extract_problem", side_effect=observe):
                    started = time.monotonic()
                    response = await client.post("/solve", json=request)
                    data = response.json()
                    request_s = time.monotonic() - started
                # All file writes, copying, replay comparisons and deterministic
                # search below are excluded from measured request latency.
                stem = sample.replace("/", "_")
                write(output_dir / f"{stem}-sdk-response.json", data)
                row = {"sample": sample, "http_status": response.status_code,
                       "status": data.get("status"), "request_s": request_s,
                       "sdk_llm_calls": data.get("metadata", {}).get("llm_calls"),
                       "sdk_total_tokens": data.get("metadata", {}).get("total_tokens"),
                       "response_bytes": len(response.content), "capture_count": len(capture)}
                if len(capture) == 1:
                    actual_task, graph, baseline, problem, context = capture[0]
                    write(output_dir / f"{stem}-capture.json", {
                        "sample": sample, "task": actual_task, "topology": topology,
                        "baseline": baseline, "problem": problem, "context": asdict(context),
                        "source_sha256": initial_sources,
                    })
                try:
                    if response.status_code != 200 or data.get("status") != "success":
                        raise AssertionError("official SDK request was not successful; complete response retained")
                    if row["sdk_llm_calls"] != 1 or len(capture) != 1:
                        raise AssertionError("expected one real model call and one immutable edit window")
                    actual_task, graph, baseline, problem, context = capture[0]
                    if digest(baseline) != context.baseline_sha256:
                        raise AssertionError("measured service mutated its model baseline")
                    if digest(problem) != context.problem_sha256:
                        raise AssertionError("measured service mutated its model edit catalogue")
                    final = replay(actual_task, graph, data)
                    if not final["valid"]:
                        raise AssertionError("official-SDK response failed independent local Replay")
                    diagnostics = data["output"]["diagnostics"]
                    edit, llm = diagnostics["search"]["model_edit"], diagnostics["llm"]
                    if edit["base"] != problem["base"] or edit["baseline_sha256"] != context.baseline_sha256:
                        raise AssertionError("model evidence does not refer to the captured actual baseline")
                    if llm.get("synthetic"):
                        raise AssertionError("a live ablation must not contain synthetic model responses")
                    deterministic, proposal, control = optimize_counts(
                        actual_task, graph, baseline, problem, context, time_budget_s=budget)
                    control_check = replay(actual_task, graph, deterministic)
                    if not control_check["valid"]:
                        raise AssertionError("same-catalogue deterministic control failed Replay")
                    if digest(baseline) != context.baseline_sha256 or digest(problem) != context.problem_sha256:
                        raise AssertionError("deterministic control mutated shared ablation inputs")
                    write(output_dir / f"{stem}-deterministic-control.json", {
                        "proposal": proposal, "audit": control, "solution": deterministic,
                    })
                    row.update(valid=True, input_sha256=context.input_sha256,
                               baseline_sha256=context.baseline_sha256, problem_sha256=context.problem_sha256,
                               groups=len(problem["groups"]), baseline_us=context.before_us,
                               model_raw_proposal_us=edit.get("proposal_time_us"),
                               model_filtered_us=edit["after_us"], model_accepted=edit["accepted"],
                               model_selected=edit["selected"], model_preserved=edit.get("preserved_in_final"),
                               deterministic_raw_proposal_us=control["predicted_proposal_us"],
                               deterministic_filtered_us=control_check["total_time_us"],
                               deterministic_elapsed_s=control["with_final_validation_s"],
                               deterministic_evaluations=control["evaluations"],
                               model_vs_deterministic_us=control_check["total_time_us"] - edit["after_us"],
                               model_beats_same_catalog_control=edit["after_us"] < control_check["total_time_us"] - 1e-6,
                               full_service_final_us=final["total_time_us"],
                               llm_s=llm.get("duration_ms", 0) / 1000,
                               model_canonical_proposal=llm.get("response_proposal"),
                               model_wire_proposal=llm.get("response_wire_proposal"),
                               deterministic_proposal=proposal,
                               comparison_note="Same immutable baseline and editable groups. Deterministic control evaluates full path costs; the model sees the production compact projection. Final service additionally merges control rounds and performs broad routing.")
                except Exception as exc:
                    row.update(valid=False, error=f"{type(exc).__name__}: {exc}")
                row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                report["cases"].append(row)
                write(output_dir / "summary.json", report)
                print(json.dumps({key: row.get(key) for key in (
                    "sample", "valid", "sdk_llm_calls", "request_s", "baseline_us", "model_filtered_us",
                    "deterministic_filtered_us", "full_service_final_us", "model_selected", "error"
                )}), flush=True)
    report["source_sha256_at_end"] = source_hashes()
    report["source_files_changed_during_run"] = initial_sources != report["source_sha256_at_end"]
    report["valid"] = all(row["valid"] and row["sdk_llm_calls"] == 1 for row in report["cases"]) and not report["source_files_changed_during_run"]
    write(output_dir / "summary.json", report)
    return 0 if report["valid"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["mesh2d/ag", "mesh2d/ar"])
    parser.add_argument("--output-dir", default="artifacts/v1.2/live-ablation")
    parser.add_argument("--budget", type=float, default=1.0)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    memory_limit = 1024**3 if hard == resource.RLIM_INFINITY else min(1024**3, hard)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    raise SystemExit(asyncio.run(run(args.samples, args.output_dir, args.budget)))


if __name__ == "__main__":
    main()
