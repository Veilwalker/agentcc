"""Forwarding-window experiment on archived valid answers, not an SDK score.

Offline modes use no credentials/network. Live mode calls the official model
once per case and compares its actual selection to greedy and exhaustive
same-catalog controls. Costs include the extra extraction and compilation work.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

from src.forward_edit import apply, compile_rewrites, _round_loads
from tests.forward_catalog import extract
from src.llm import edit_forwarding, LLMError
from src.replay import replay
from src.route_balance import _PathCosts
from src.topology import Topology
from tests.compare_versions import PROJECT, apply_limits, within_project


def score(task, graph, base, context, ids):
    candidate = compile_rewrites(base, context, ids)
    costs = _PathCosts(graph)
    return sum(load.time_us for load in _round_loads(graph, candidate,
                   task["data_size"]["per_rank_bytes"] // base["split"], costs))


def controls(task, graph, base, context):
    started = time.process_time()
    best, best_ids = context.before_us, []
    attempts = 0
    greedy = []
    for i in range(len(context.rewrites)):
        try:
            value = score(task, graph, base, context, greedy + [i])
            attempts += 1
        except ValueError:
            continue
        if value < best - 1e-6:
            best, greedy = value, greedy + [i]
    greedy_result = {"ids": greedy, "time_us": best, "cpu_s": time.process_time() - started, "evaluations": attempts}
    started = time.process_time()
    best, best_ids, attempts = context.before_us, [], 0
    for bits in itertools.product((False, True), repeat=len(context.rewrites)):
        ids = [i for i, value in enumerate(bits) if value]
        try:
            value = score(task, graph, base, context, ids)
            attempts += 1
        except ValueError:
            continue
        if value < best - 1e-6:
            best, best_ids = value, ids
    compiled, validation = apply(task, graph, base, context, {"apply": best_ids})
    checked = replay(task, graph, compiled)
    assert checked["valid"] and abs(checked["total_time_us"] - best) < 1e-5
    return greedy_result, {"ids": best_ids, "time_us": best, "cpu_s": time.process_time() - started,
                          "evaluations": attempts, "valid": checked["valid"]}


def run(output, samples=None, live=False, groups=4):
    if output.exists():
        raise ValueError("preserve completed experiments; use a new output filename")
    limits = apply_limits()
    archived = PROJECT / "artifacts/v1.5/runs/revision-03-final"
    summary = json.loads((archived / "summary.json").read_text())
    rows = [r for r in summary["rows"] if r["version"] == "1.5" and r["repeat"] == 1
            and (samples is None or r["sample"] in samples)]
    result = {"scope": __doc__, "limits": limits, "live": live, "max_groups": groups, "rows": []}
    for row in rows:
        label = row["sample"]
        task_input = json.loads((archived / "inputs" / (label.replace("/", "-") + ".json")).read_text())["input"]
        base = json.loads((archived / row["response_file"]).read_text())["output"]["answer"]
        task, graph = task_input["task"], Topology.from_data(task_input["topology"])
        begin = time.process_time()
        problem, context, extraction = extract(task, graph, base, max_groups=groups, max_route_queries=0)
        extraction["cpu_s"] = time.process_time() - begin
        record = {"sample": label, "base_us": context.before_us, "extraction": extraction, "problem": problem}
        if live:
            try:
                proposal, evidence = edit_forwarding(task, problem)
                candidate, acceptance = apply(task, graph, base, context, proposal)
                acceptance.pop("_validation_report", None)
                checked = replay(task, graph, candidate)
                record.update(model=evidence, acceptance=acceptance, valid=checked["valid"], time_us=checked["total_time_us"])
            except LLMError as error:
                record.update(valid=False, error=str(error))
        greedy, exhaustive = controls(task, graph, base, context)
        record.update(greedy=greedy, exhaustive=exhaustive)
        result["rows"].append(record)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k:v for k,v in record.items() if k in ("sample", "base_us", "time_us", "valid", "error", "greedy", "exhaustive")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--groups", type=int, default=4)
    args = parser.parse_args()
    run(within_project(args.output), args.samples, args.live, args.groups)
