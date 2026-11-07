"""Offline work-budget calibration for concrete model edit extraction.

Prepared quick-seed schedules are identical across compared modes. Reported
token estimates use prompt characters / 4 and are not gateway usage.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

from src.llm import build_edit_request
from src.route_work import RouteWorkBudget
from src.schedule_edit import _input_digest, extract_problem
from src.solver import solve_schedule
from src.topology import Topology


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "materials/spec/examples/256-rank"
ARTIFACTS = ROOT / "artifacts/v1.4/analysis/edit-work"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture(case):
    target = ARTIFACTS / "fixtures" / (case.replace("/", "-") + ".json")
    if target.exists():
        return json.loads(target.read_text())
    task = json.loads((PUBLIC / case / "task.json").read_text())
    topology = json.loads((PUBLIC / case / "topology.json").read_text())
    graph = Topology.from_data(topology)
    splits = task["split_policy"]["candidates"]
    split = max(splits) if task["operator"] == "all-reduce" else min(splits)
    baseline = solve_schedule(task, graph, seed_only=True, seed_split=split, seed_weight=4.0)
    report = baseline["strategy"]["validation_report"]
    baseline = {key: value for key, value in baseline.items() if key != "strategy"}
    value = {"task": task, "topology": topology, "baseline": baseline, "report": report}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    return value


def mode_budget(mode):
    if mode == "legacy":
        return None
    if mode == "measure":
        return RouteWorkBudget(10**9, 10**9, safety_s=0.2)
    scores, edges = map(int, mode.split(":"))
    return RouteWorkBudget(scores, edges, safety_s=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--modes", nargs="+", default=["legacy", "measure", "10000:35000", "10000:65000", "20000:35000"])
    parser.add_argument("--output", default="calibration.json")
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    cases = args.cases or [str(p.parent.relative_to(PUBLIC)) for p in sorted(PUBLIC.glob("*/*/task.json"))]
    output = ARTIFACTS / args.output
    if not output.resolve().is_relative_to(ARTIFACTS):
        parser.error("output must stay in this experiment's artifact directory")
    report = {"scope": __doc__, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
              "affinity": sorted(os.sched_getaffinity(0)), "rows": [],
              "source_sha256": {name: hashlib.sha256((ROOT / "src" / (name + ".py")).read_bytes()).hexdigest()
                                for name in ("schedule_edit", "route_work", "llm")}}
    for case in cases:
        value = fixture(case)
        for mode in args.modes:
            graph = Topology.from_data(value["topology"])
            baseline_before = digest(value["baseline"])
            input_before = _input_digest(value["task"], graph)
            gc.collect()
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            problem, context = extract_problem(value["task"], graph, value["baseline"],
                                                validated_report=value["report"],
                                                work_budget=mode_budget(mode), time_budget_s=0.2)
            wall, cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
            baseline_unchanged = digest(value["baseline"]) == baseline_before
            input_unchanged = _input_digest(value["task"], graph) == input_before
            assert baseline_unchanged and input_unchanged
            system, user, _ = build_edit_request(value["task"], problem)
            characters = len(system) + len(user)
            row = {"case": case, "mode": mode, "cpu_s": cpu, "wall_s": wall,
                   "catalogue_sha256": context.problem_sha256, "base": problem["base"],
                   "baseline_sha256": context.baseline_sha256,
                   "baseline_unchanged": baseline_unchanged, "input_unchanged": input_unchanged,
                   "input_sha256": input_before,
                   "public_file_sha256": {name: hashlib.sha256((PUBLIC / case / name).read_bytes()).hexdigest()
                                          for name in ("task.json", "topology.json")},
                   "prompt_sha256": digest([system, user]), "prompt_characters": characters,
                   "estimated_tokens_4chars": math.ceil(characters / 4),
                   "group_count": len(context.groups),
                   "groups": [{"round": group.round_index, "max_chunks": group.max_chunks,
                               "members": len(group.members)} for group in context.groups],
                   "work": context.extraction_work, "problem": problem}
            report["rows"].append(row)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: row[k] for k in ("case", "mode", "cpu_s", "group_count", "groups", "work")}), flush=True)
    report["source_sha256_after"] = {name: hashlib.sha256((ROOT / "src" / (name + ".py")).read_bytes()).hexdigest()
                                     for name in ("schedule_edit", "route_work", "llm")}
    assert report["source_sha256_after"] == report["source_sha256"]
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
