"""Compare wall-limited and explicitly work-limited broad routing offline.

Each saved pre-broad input is constructed once with the v1.3 staged control.
All measured alternatives then receive the exact same schedule and a fresh
topology. No SDK, model call or credentials are used. Prepared fixtures and
results live under artifacts/v1.4/analysis/route-work.
"""

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from src.replay import replay
from src.route_balance import refine
from src.route_broad import refine_broad
from src.route_work import RouteWorkBudget
from src.solver import solve_schedule
from src.topology import Topology


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "materials/spec/examples/256-rank"
ARTIFACTS = ROOT / "artifacts/v1.4/analysis/route-work"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(case):
    target = ARTIFACTS / "fixtures" / (case.replace("/", "-") + ".json")
    if target.exists():
        return json.loads(target.read_text())
    folder = PUBLIC / case
    task = json.loads((folder / "task.json").read_text())
    topology = json.loads((folder / "topology.json").read_text())
    graph = Topology.from_data(topology)
    choices = task["split_policy"]["candidates"]
    split = max(choices) if task["operator"] == "all-reduce" else min(choices)
    seed = solve_schedule(task, graph, seed_only=True, seed_split=split, seed_weight=4.0)
    resumed = solve_schedule(task, graph, incumbent=seed, time_budget_s=1.0)
    search = resumed["strategy"]
    weights = sorted(set(search.get("constructed_route_weights", [])) | {
        row["actual"]["resource_weight"] for row in search["candidates"] if row.get("valid")})
    if search["route_variation"]:
        solution, _ = refine(task, graph, resumed, weights=weights,
                             routing_bytes=search["routing_bytes"], time_budget_s=0.25)
    else:
        solution = resumed
    report = replay(task, graph, solution)
    if not report["valid"]:
        raise ValueError(report)
    solution = {key: value for key, value in solution.items() if key != "strategy"}
    value = {"task": task, "topology": topology, "solution": solution, "report": report,
             "case": case, "solution_sha256": digest(solution)}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    return value


def parse_mode(value):
    if value == "legacy":
        return value, None
    if value == "measure":
        return value, RouteWorkBudget(10**9, 10**9, safety_s=0.1875)
    scores, edges = (int(part) for part in value.split(":"))
    return value, RouteWorkBudget(scores, edges, safety_s=3.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--modes", nargs="+", default=["legacy", "measure"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", default="calibration.json")
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    cases = args.cases or [str(path.parent.relative_to(PUBLIC)) for path in sorted(PUBLIC.glob("*/*/task.json"))]
    modes = [parse_mode(mode) for mode in args.modes]
    output = ARTIFACTS / args.output
    if not output.resolve().is_relative_to(ARTIFACTS):
        parser.error("output must stay in this experiment's artifact directory")
    source_hashes = {name: hashlib.sha256((ROOT / "src" / (name + ".py")).read_bytes()).hexdigest()
                     for name in ("route_broad", "route_work")}
    report = {"scope": __doc__, "affinity": sorted(os.sched_getaffinity(0)),
              "source_sha256": source_hashes, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
              "rows": []}
    for case in cases:
        value = prepare(case)
        for repeat in range(1, args.repeats + 1):
            for mode, budget in modes:
                graph = Topology.from_data(value["topology"])
                solution = deepcopy(value["solution"])
                gc.collect()
                started_wall, started_cpu = time.perf_counter(), time.process_time()
                final, info = refine_broad(value["task"], graph, solution, time_budget_s=0.25,
                                           work_budget=budget)
                wall, cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
                checked = info.pop("final_replay")
                if not checked["valid"]:
                    raise ValueError(checked)
                row = {"case": case, "repeat": repeat, "mode": mode,
                       "wall_s": wall, "cpu_s": cpu, "valid": checked["valid"],
                       "before_us": value["report"]["total_time_us"], "after_us": checked["total_time_us"],
                       "program_sha256": digest({k: v for k, v in final.items() if k != "metrics"}),
                       "input_sha256": value["solution_sha256"], "evidence": info}
                report["rows"].append(row)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: row[k] for k in ("case", "repeat", "mode", "cpu_s", "after_us")} |
                                 {"work": info.get("work")}), flush=True)


if __name__ == "__main__":
    main()
