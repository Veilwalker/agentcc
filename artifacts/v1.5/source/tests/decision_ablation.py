"""Offline decision ablation with identical control budgets and saved live replies.

This is not an LLM benchmark: no credentials, network calls or new Tokens. The
exact prompt must match the recorded request before replaying a reply. Zero and
invalid replies are explicitly synthetic interventions, never model evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from src import service
from src.config import Settings
from src.replay import replay
from tests.compare_versions import PROJECT, apply_limits, within_project


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(recordings, output, *, samples=None, recording_repeat=1):
    summary = json.loads((recordings / "summary.json").read_text())
    if summary.get("source_files_changed_during_run"):
        raise ValueError("recorded source changed")
    if output.exists():
        raise ValueError("do not overwrite an existing experiment")
    first = {}
    for row in summary["rows"]:
        if (row["version"] == "1.5" and row["valid"] and row["repeat"] == recording_repeat
                and (samples is None or row["sample"] in samples)):
            first.setdefault(row["sample"], row)
    if not first:
        raise ValueError("no matching real recordings")
    limits = apply_limits()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT / "src").glob("*.py")}
    result = {"synthetic": True, "actual_model_calls": 0, "actual_tokens": 0,
              "scope": __doc__, "recording_repeat": recording_repeat, "limits": limits, "source_hashes": hashes, "rows": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    for sample, row in first.items():
        saved = json.loads((recordings / row["response_file"]).read_text())
        model = saved["output"]["diagnostics"]["llm"]
        expected = model["request_evidence"]["messages"]
        original_raw = model["response_evidence"]["content"]
        task_input = json.loads((recordings / "inputs" / (sample.replace("/", "-") + ".json")).read_text())["input"]
        initial = digest(task_input)
        settings = Settings("http://unused.invalid/v1/chat/completions", "synthetic-only", model["model"])
        for mode in ("recorded", "zero", "recorded_repeat", "invalid"):
            calls = []

            def completion(system, content, _settings, **_kwargs):
                messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
                if messages != expected:
                    raise ValueError("current prompt differs from recorded prompt; do not rebind")
                groups = json.loads(content)["problem"]["groups"]
                raw = (original_raw if mode.startswith("recorded") else
                       json.dumps({"counts": [0] * len(groups)}) if mode == "zero" else "invalid JSON")
                calls.append(raw)
                return raw, {"synthetic": True, "usage": {}, "finish_reason": "stop", "duration_ms": 0}

            with patch.object(service, "load_settings", return_value=settings), \
                    patch("src.llm._chat_completion", side_effect=completion), \
                    patch("src.llm.build_opener", side_effect=AssertionError("network forbidden")):
                start, cpu = time.monotonic(), time.process_time()
                actual = service.solve_input(task_input)
                wall_s, cpu_s = time.monotonic() - start, time.process_time() - cpu
            checked = replay(task_input["task"], task_input["topology"], actual["answer"])
            edit = actual["diagnostics"]["search"]["model_edit"]
            answer = {k: v for k, v in actual["answer"].items() if k != "metrics"}
            observation = {"sample": sample, "mode": mode, "valid": checked["valid"],
                           "wall_s": wall_s, "cpu_s": cpu_s, "comm_us": checked["total_time_us"],
                           "schedule_sha256": digest(answer), "model_selected": edit["selected"],
                           "model_branch_won": edit["model_branch_won"],
                           "input_unchanged": digest(task_input) == initial, "adapter_calls": len(calls),
                           "control_us": edit["control_time_us"],
                           "recording": row["response_file"]}
            assert observation["valid"] and observation["input_unchanged"] and len(calls) == 1
            assert checked["total_time_us"] <= edit["control_time_us"] + 1e-6
            result["rows"].append(observation)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(observation), flush=True)
    result["source_unchanged"] = hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT / "src").glob("*.py")}
    assert result["source_unchanged"]
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--recording-repeat", type=int, default=1)
    args = parser.parse_args()
    run(within_project(args.recordings), within_project(args.output),
        samples=args.samples, recording_repeat=args.recording_repeat)
