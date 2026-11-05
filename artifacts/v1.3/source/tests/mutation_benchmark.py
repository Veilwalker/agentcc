"""Single-CPU offline service regression with explicitly synthetic model edits.

Current public case IDs are preserved while directions, edges, capacities and
overlapping resources change. The real edit prompt/parser/compiler/service run;
only configuration loading and the HTTP completion are replaced. No credentials
are loaded, no model/API request is sent, and results are not LLM quality claims.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

from src.config import Settings
from src.llm import edit_schedule as real_edit_schedule
from src.replay import replay
from src.schedule_edit import apply_proposal as real_apply_proposal
from src.service import solve_input
from tests.test_regressions_v11 import PROJECT, PUBLIC, perturb_topology


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((PROJECT / "src").glob("*.py"))}


def constrain_this_process():
    """Change only this test process, never machine/container configuration."""
    available = os.sched_getaffinity(0)
    selected = min(available)
    os.sched_setaffinity(0, {selected})
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    memory_limit = 1024 * 1024 * 1024
    if hard != resource.RLIM_INFINITY:
        memory_limit = min(memory_limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    return {"single_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "address_space_limit_bytes": memory_limit,
            "scope": "this process and its threads only; not a container CPU quota"}


def synthetic_completion(mode, seen):
    """Exercise the actual prompt and strict parser without opening a socket."""
    def complete(system_prompt, user_content, settings, **_kwargs):
        request_data = json.loads(user_content)
        problem = request_data["problem"]
        if mode == "invalid_unknown_group":
            edits = [["not-an-advertised-group", 1]]
        elif mode == "noop":
            edits = []
        else:
            numerator = 5 if mode == "half" else 8
            edits = [[group["id"], max(1, group["max_chunks"] * numerator // 10)]
                     for group in problem["groups"]]
        if mode == "invalid_unknown_group":
            proposal = {"base": problem["base"], "edits": edits}
        else:
            amounts = dict(edits)
            proposal = {"base": problem["base"], "counts": [
                amounts.get(group["id"], 0) for group in problem["groups"]]}

        def without_opaque_bindings(value):
            if isinstance(value, dict):
                return {key: without_opaque_bindings(item) for key, item in value.items()
                        if key not in ("base", "topology_fingerprint") and not key.endswith("sha256")}
            if isinstance(value, list):
                return [without_opaque_bindings(item) for item in value]
            return value

        semantic_hash = digest(without_opaque_bindings(request_data))
        content = json.dumps(proposal, separators=(",", ":"))
        payload = {"model": settings.model,
                   "messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content}],
                   "temperature": 0.2, "max_tokens": settings.max_tokens, "stream": False}
        request_body = json.dumps(payload, ensure_ascii=False).encode()
        seen.append({"base": problem["base"], "proposal": proposal,
                     "groups": len(problem["groups"]), "request": payload})
        return content, {
            "called": True, "synthetic": True, "mode": "offline-regression",
            "model": settings.model, "duration_ms": 0, "usage": {},
            "request_evidence": payload,
            "input_sha256": hashlib.sha256(user_content.encode()).hexdigest(),
            "request_sha256": hashlib.sha256(request_body).hexdigest(),
            "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "synthetic_semantic_request_sha256": semantic_hash,
            "finish_reason": "stop",
        }
    return complete


def run(destination: Path, seeds, families, budget=1.0):
    limits = constrain_this_process()
    report = {"scope": "Offline full-service perturbation regression with synthetic model responses; not official Judge or LLM performance",
              "real_model_calls": 0, "network_calls": 0, "credentials_loaded": False,
              "search_budget_s": budget, "cold_graph_per_case": True,
              "limits": limits, "source_sha256_at_start": source_hashes(), "rows": []}
    settings = Settings("http://unused.test/v1/chat/completions", "synthetic-only-key", "synthetic-offline-model")
    previous = {}
    modes = ("half", "eighty_percent", "invalid_unknown_group", "noop")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for family in families:
            for operator in ("ag", "ar"):
                base = PUBLIC / family / operator
                task = json.loads((base / "task.json").read_text())
                original = json.loads((base / "topology.json").read_text())
                topology, changes = perturb_topology(original, seed, preserve_rank_connectivity=True)
                task_input = {"task": task, "topology": topology}
                input_hash = digest(task_input)
                input_before = json.dumps(task_input, sort_keys=True)
                mode = modes[len(report["rows"]) % len(modes)]
                seen = []
                compiled_frozen = {}

                def capture_compilation(*args, **kwargs):
                    compiled, evidence = real_apply_proposal(*args, **kwargs)
                    if evidence["accepted"]:
                        for ri, ti in evidence["frozen_transfers"]:
                            compiled_frozen[ri, ti] = deepcopy(compiled["rounds"][ri]["transfers"][ti])
                    return compiled, evidence

                row = {"family": family, "operator": operator, "seed": seed,
                       "task_id": task["task_id"], "mode": mode, "mutations": changes,
                       "input_sha256": input_hash, "original_input_sha256": digest({"task": task, "topology": original})}
                started, cpu_started = time.monotonic(), time.process_time()
                try:
                    with patch("src.service.load_settings", return_value=settings), \
                            patch("src.service.edit_schedule", wraps=real_edit_schedule) as adapter, \
                            patch("src.service.apply_proposal", side_effect=capture_compilation), \
                            patch("src.llm._chat_completion", side_effect=synthetic_completion(mode, seen)) as completion, \
                            patch("src.llm.build_opener", side_effect=AssertionError("network forbidden in this regression")):
                        result = solve_input(task_input, search_budget_s=budget)
                    wall_s, cpu_s = time.monotonic() - started, time.process_time() - cpu_started
                    checked = replay(task, topology, result["answer"])
                    diagnostic = result["diagnostics"]
                    model, search = diagnostic["llm"], diagnostic["search"]
                    edit = search["model_edit"]
                    selected = [item for item in search["candidates"] if item.get("selected")]
                    assert checked["valid"], checked["errors"]
                    assert len(selected) == 1, "selection must be unique"
                    assert abs(selected[0]["total_time_us"] - checked["total_time_us"]) < 1e-6
                    assert abs(result["answer"]["metrics"]["total_time"] * 1e6 - checked["total_time_us"]) < 1e-6
                    assert json.dumps(task_input, sort_keys=True) == input_before, "service mutated request input"
                    assert input_hash != row["original_input_sha256"], "topology perturbation did not change input"
                    assert model["source_input_sha256"] == input_hash, "wrong current-input evidence"
                    assert search["topology_sha256"] == digest(topology), "solver used stale topology"
                    assert adapter.call_count == completion.call_count == len(seen) == 1
                    assert edit["base"] == seen[0]["base"], "proposal base was not current"
                    actual_request_hash = hashlib.sha256(json.dumps(model["request_evidence"], ensure_ascii=False).encode()).hexdigest()
                    assert model["request_sha256"] == actual_request_hash, "request evidence hash mismatch"
                    if mode in ("invalid_unknown_group", "noop"):
                        assert not edit["accepted"] and not edit["selected"]
                    assert edit["preservation_scope"] == "observed_not_enforced"
                    assert not edit["selected_frozen_transfers"], "model paths must not constrain the control"
                    assert edit["preserved_in_final"] == edit["selected"]
                    for ri, ti in edit["selected_transfers"]:
                        assert result["answer"]["rounds"][ri]["transfers"][ti] == compiled_frozen[ri, ti], "reported selected model transfer is absent from the actual answer"
                    previous_cost = edit["selected_time_us"] if edit["selected"] else edit["control_time_us"]
                    assert checked["total_time_us"] <= previous_cost + 1e-6, "final routing worsened the selected schedule"
                    if edit["selected"]:
                        assert selected[0]["actual"]["edits"] == edit["selected_edits"], "history includes unselected model edits"
                    key = family, operator
                    if key in previous:
                        old = previous[key]
                        for name, value in (("input", input_hash), ("base", edit["base"]),
                                            ("request", model["request_sha256"]),
                                            ("semantic_request", model["synthetic_semantic_request_sha256"])):
                            assert value != old[name], f"same-name mutation reused old {name} evidence"
                    previous[key] = {"input": input_hash, "base": edit["base"], "request": model["request_sha256"],
                                     "semantic_request": model["synthetic_semantic_request_sha256"]}
                    row.update(valid=True, errors=[], wall_s=round(wall_s, 6), cpu_s=round(cpu_s, 6),
                               replay_us=checked["total_time_us"], rounds=checked["nonempty_rounds"],
                               selected_source=selected[0]["source"], selected=selected[0],
                               model_edit_accepted=edit["accepted"], model_edit_selected=edit["selected"],
                               model_edit_preserved=edit["preserved_in_final"],
                               selected_rounds=edit["selected_rounds"],
                               selected_transfer_count=len(edit["selected_transfers"]),
                               route_finish=search["route_finish"],
                               edit_base=edit["base"], baseline_sha256=edit["baseline_sha256"],
                               request_sha256=model["request_sha256"], groups=seen[0]["groups"],
                               semantic_request_sha256=model["synthetic_semantic_request_sha256"],
                               synthetic_proposal=seen[0]["proposal"],
                               decoded_proposal=model.get("response_proposal"),
                               proposal_encoding=model.get("proposal_encoding"),
                               compiler_errors=edit["errors"], input_unchanged=True,
                               phases_s=diagnostic["phases_s"],
                               process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               solution_bytes=len(json.dumps(result["answer"], separators=(",", ":")).encode()))
                except Exception as exc:
                    row.update(valid=False, errors=[f"{type(exc).__name__}: {exc}"],
                               wall_s=round(time.monotonic() - started, 6),
                               cpu_s=round(time.process_time() - cpu_started, 6))
                report["rows"].append(row)
                report["summary"] = {
                    "completed": len(report["rows"]), "valid": sum(row["valid"] for row in report["rows"]),
                    "invalid": sum(not row["valid"] for row in report["rows"]),
                    "synthetic_modes": dict(Counter(row["mode"] for row in report["rows"])),
                    "model_edits_accepted": sum(row.get("model_edit_accepted", False) for row in report["rows"]),
                    "model_edits_selected": sum(row.get("model_edit_selected", False) for row in report["rows"]),
                    "model_edits_preserved": sum(row.get("model_edit_preserved", False) for row in report["rows"]),
                }
                destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                print(json.dumps({name: row.get(name) for name in (
                    "family", "operator", "seed", "mode", "valid", "selected_source", "wall_s", "cpu_s", "errors"
                )}), flush=True)
    report["source_sha256_at_end"] = source_hashes()
    report["source_files_changed_during_run"] = report["source_sha256_at_start"] != report["source_sha256_at_end"]
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/v1.3/analysis/mutations.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--families", nargs="+", default=sorted(path.name for path in PUBLIC.iterdir() if path.is_dir()))
    parser.add_argument("--budget", type=float, default=1.0)
    args = parser.parse_args()
    destination = (PROJECT / args.output).resolve()
    if not destination.is_relative_to(PROJECT.resolve()):
        raise ValueError("output must stay inside the project")
    for family in args.families:
        if not (PUBLIC / family).resolve().is_relative_to(PUBLIC.resolve()):
            raise ValueError("case inputs must stay inside public examples")
    report = run(destination, args.seeds, args.families, args.budget)
    if report["summary"]["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
