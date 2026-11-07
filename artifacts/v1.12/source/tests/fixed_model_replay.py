"""Offline replay of saved raw model content; no rebinding and no network calls.

Full-service runs vary only the simulated response wait. A second layer repeats
apply/selection on one captured baseline, catalogue and control. Binding changes
are reported rather than repaired. This is not a model-quality or SDK benchmark.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

from src.config import Settings
from src import service
from src.replay import replay
from src.schedule_edit import apply_proposal
from src.selection import choose_model_candidate


PROJECT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((PROJECT / "src").glob("*.py"))}


def program_digest(solution):
    return digest({key: value for key, value in solution.items() if key not in {"strategy", "metrics"}})


def constrain_this_process():
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    bound = 1024 * 1024**2
    if hard != resource.RLIM_INFINITY:
        bound = min(bound, hard)
    resource.setrlimit(resource.RLIMIT_AS, (bound, bound))
    return {"cpu_affinity": [cpu], "address_space_bytes": bound, "scope": "this process only"}


def recorded_completion(raw, original_metadata, wait_s, seen):
    def complete(system_prompt, user_content, settings, **kwargs):
        started = time.monotonic()
        current = json.loads(user_content)
        payload = {"model": settings.model, "messages": [
            {"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            "temperature": kwargs.get("temperature", 0.2), "max_tokens": settings.max_tokens, "stream": False}
        seen.update(request=current, request_payload=payload)
        if wait_s:
            time.sleep(wait_s)
        return raw, {"synthetic": True, "mode": "saved_raw_response_replay", "real_model_calls": 0,
                     "network_calls": 0, "model": settings.model, "usage": {},
                     "duration_ms": (time.monotonic() - started) * 1000,
                     "finish_reason": original_metadata["finish_reason"], "request_evidence": payload,
                     "input_sha256": hashlib.sha256(user_content.encode()).hexdigest(),
                     "request_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest(),
                     "response_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    return complete


def frozen_repeats(task_input, capture, proposal, count, threshold):
    rows = []
    for number in range(count):
        started = time.monotonic()
        edited, evidence = apply_proposal(task_input["task"], capture["graph"], capture["baseline"],
                                           capture["context"], proposal)
        report = evidence.pop("_validation_report", None)
        result, checked, selection = choose_model_candidate(
            task_input["task"], capture["graph"], capture["control"], capture["control_report"],
            edited, report, evidence["accepted_rounds"], baseline=capture["baseline"],
            overlay_min_relative_gain=threshold)
        rows.append({"repeat": number + 1, "elapsed_s": time.monotonic() - started,
                     "accepted": evidence["accepted"], "combination": selection["combination"],
                     "program_sha256": program_digest(result), "time_us": checked["total_time_us"],
                     "selected_rounds": selection["selected_rounds"],
                     "errors": evidence["errors"] + selection["errors"]})
    return rows


def select_recordings(rows, samples=None):
    """Keep the first recorded 1.4 response per requested distinct sample."""
    requested = set(samples) if samples is not None else None
    selected = {}
    for row in rows:
        if row.get("version") != "1.4":
            continue
        name = row["sample"]
        if requested is None or name in requested:
            selected.setdefault(name, row)
    if requested is not None and requested - selected.keys():
        raise ValueError("no 1.4 recording for samples: " + ", ".join(sorted(requested - selected.keys())))
    if not selected:
        raise ValueError("no 1.4 recordings selected")
    return list(selected.values())


def run(pilot, destination, delays, samples=None):
    original = json.loads((pilot / "summary.json").read_text())
    sources = select_recordings(original["rows"], samples)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("output directory must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    limits = constrain_this_process()
    report = {"scope": "Offline saved raw model responses, never rebound or retried; not actual model calls",
              "real_model_calls": 0, "network_calls": 0, "credentials_loaded": False,
              "synthetic": True, "limits": limits, "simulated_waits_s": delays,
              "selected_samples": [row["sample"] for row in sources],
              "expected_requests": len(sources) * len(delays),
              "recording_policy": "first 1.4 row per distinct sample, in original report order",
              "recording_sources": [],
              "source_sha256_at_start": source_hashes(), "rows": [], "frozen_layers": []}
    real_extract, real_control = service.extract_problem, service._control_candidate
    for source in sources:
        name = source["sample"].replace("/", "-")
        source_path = pilot / source["response_file"]
        source_bytes = source_path.read_bytes()
        saved = json.loads(source_bytes)
        metadata = saved["output"]["diagnostics"]["llm"]
        raw = metadata["response_evidence"]["content"]
        source_reference = {"sample": source["sample"],
                            "source_response_file": str(source_path.relative_to(PROJECT)),
                            "source_response_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
                            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest()}
        report["recording_sources"].append(source_reference)
        original_request = json.loads(metadata["request_evidence"]["messages"][1]["content"])
        original_base = original_request["problem"]["base"]
        original_semantic = deepcopy(original_request)
        original_semantic["problem"].pop("base")
        task_input = json.loads((pilot / "inputs" / f"{name}.json").read_text())
        if "input" in task_input:
            task_input = task_input["input"]
        original_input_sha256 = digest(task_input)
        settings = Settings("http://unused.invalid/v1/chat/completions", "synthetic-only", metadata["model"])
        frozen_capture = frozen_proposal = None
        threshold = 0.0
        for number, wait_s in enumerate(delays, 1):
            capture, seen = {}, {}

            def extract(task, graph, baseline, **kwargs):
                problem, context = real_extract(task, graph, baseline, **kwargs)
                capture.update(graph=graph, baseline=baseline, problem=problem, context=context)
                return problem, context

            def control(*args, **kwargs):
                result, checked, elapsed = real_control(*args, **kwargs)
                capture.update(control=result, control_report=checked)
                return result, checked, elapsed

            started, cpu_started = time.monotonic(), time.process_time()
            current_input = deepcopy(task_input)
            with patch("src.service.load_settings", return_value=settings), \
                    patch("src.service.extract_problem", side_effect=extract), \
                    patch("src.service._control_candidate", side_effect=control), \
                    patch("src.llm._chat_completion", side_effect=recorded_completion(raw, metadata, wait_s, seen)) as call, \
                    patch("src.llm.build_opener", side_effect=AssertionError("network is forbidden")):
                output = service.solve_input(current_input)
            elapsed, cpu_elapsed = time.monotonic() - started, time.process_time() - cpu_started
            edit = output["diagnostics"]["search"]["model_edit"]
            current_request = deepcopy(seen["request"])
            current_base = current_request["problem"].pop("base")
            base_matches = current_base == original_base
            checked = replay(task_input["task"], task_input["topology"], output["answer"])
            assert call.call_count == 1 and checked["valid"]
            assert output["diagnostics"]["llm"]["response_evidence"]["content"] == raw
            assert output["diagnostics"]["llm"]["synthetic"]
            assert checked["total_time_us"] <= edit["control_time_us"] + 1e-6
            assert digest(current_input) == original_input_sha256
            if not base_matches:
                assert not edit["accepted"], "a mismatched saved response must never be rebound"
            item = {"sample": source["sample"], "repeat": number, "simulated_wait_s": wait_s,
                    "source_response_file": source_reference["source_response_file"],
                    "source_response_file_sha256": source_reference["source_response_file_sha256"],
                    "request_s": elapsed, "cpu_s": cpu_elapsed, "valid": checked["valid"],
                    "base_matches_recording": base_matches, "recorded_base": original_base, "current_base": current_base,
                    "request_without_base_matches": digest(current_request) == digest(original_semantic),
                    "baseline_sha256": capture["context"].baseline_sha256,
                    "baseline_matches_recording": capture["context"].baseline_sha256 ==
                        saved["output"]["diagnostics"]["search"]["model_edit"]["baseline_sha256"],
                    "baseline_program_sha256": program_digest(capture["baseline"]),
                    "catalog_sha256": digest({k: v for k, v in capture["problem"].items() if k != "base"}),
                    "control_program_sha256": program_digest(capture["control"]),
                    "final_program_sha256": program_digest(output["answer"]),
                    "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "control_time_us": edit["control_time_us"], "final_time_us": checked["total_time_us"],
                    "accepted": edit["accepted"], "combination": edit["combination"],
                    "selected_rounds": edit["selected_rounds"], "errors": edit["errors"],
                    "phases_s": output["diagnostics"]["phases_s"]}
            report["rows"].append(item)
            (destination / f"{name}-r{number:02d}.json").write_text(json.dumps(output, ensure_ascii=False) + "\n")
            if base_matches and frozen_capture is None:
                frozen_capture, frozen_proposal = capture, output["diagnostics"]["llm"].get("response_proposal")
                threshold = edit.get("overlay_min_relative_gain", 0.0)
            print(json.dumps({k: item[k] for k in ("sample", "repeat", "valid", "base_matches_recording",
                                                 "control_time_us", "final_time_us", "combination")}), flush=True)
        if frozen_capture is not None:
            layer = frozen_repeats(task_input, frozen_capture, frozen_proposal, len(delays), threshold)
            report["frozen_layers"].append({"sample": source["sample"], "available": True, "rows": layer,
                "identical_decisions": len({digest({k: v for k, v in row.items() if k not in {"repeat", "elapsed_s"}})
                                            for row in layer}) == 1})
            bundle = {"task_input": task_input, "baseline": frozen_capture["baseline"],
                      "recording_source": source_reference,
                      "problem": frozen_capture["problem"], "context": asdict(frozen_capture["context"]),
                      "control": frozen_capture["control"], "control_report": frozen_capture["control_report"],
                      "proposal": frozen_proposal, "raw_model_content": raw}
            (destination / f"{name}-capture.json").write_text(json.dumps(bundle, ensure_ascii=False) + "\n")
        else:
            report["frozen_layers"].append({"sample": source["sample"], "available": False,
                "reason": "No generated catalogue matched the original raw response binding; raw base was never changed."})
        (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    report.update(source_sha256_at_end=source_hashes(), complete=True,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    report["source_changed"] = report["source_sha256_at_start"] != report["source_sha256_at_end"]
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=PROJECT / "artifacts/v1.4/runs/pilot-01")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delays", default="0,1,3", help="simulated response waits, one repeat per value")
    parser.add_argument("--samples", help="comma-separated sample IDs; default: all distinct 1.4 samples")
    args = parser.parse_args()
    delays = [float(value) for value in args.delays.split(",")]
    if not delays or any(not math.isfinite(value) or value < 0 or value > 10 for value in delays):
        parser.error("delays must be within 0..10 seconds")
    if any(PROJECT != path.resolve() and PROJECT not in path.resolve().parents
           for path in (args.pilot, args.output_dir)):
        parser.error("input and output must remain inside this project")
    samples = [value.strip() for value in args.samples.split(",") if value.strip()] if args.samples else None
    run(args.pilot.resolve(), args.output_dir.resolve(), delays, samples)
