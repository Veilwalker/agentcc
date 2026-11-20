"""Offline, single-CPU comparison of archived and staged deterministic search.

This invokes low-level constructors/refiners directly, without an SDK or model
request. Stage totals therefore measure local work, not end-to-end solve time.
"""

import argparse
import gc
import hashlib
import importlib
import importlib.util
from itertools import product
import json
import os
from pathlib import Path
import resource
import sys
import time

from src.route_broad import refine_broad
from src.route_balance import refine
from src.solver import solve_schedule
from src.topology import Topology


PROJECT = Path(__file__).resolve().parents[1]
PUBLIC = PROJECT / "materials/spec/examples/256-rank"


def archived_modules():
    root = PROJECT / "artifacts/v1.2/source/src"
    name = "archived_search_v12"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return {part: importlib.import_module(f"{name}.{part}")
            for part in ("solver", "topology", "route_balance", "route_broad")}


def timed(operation):
    wall, cpu = time.monotonic(), time.process_time()
    result = operation()
    return result, {"wall_s": time.monotonic() - wall, "cpu_s": time.process_time() - cpu}


def source_hashes():
    return {f"src/{name}.py": hashlib.sha256((PROJECT / f"src/{name}.py").read_bytes()).hexdigest()
            for name in ("solver", "topology", "route_balance", "route_broad")}


def run(output, budgets, samples, seed_weights, weighted_budget):
    old = archived_modules()
    report = {"scope": __doc__, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "source_sha256": source_hashes(), "rows": []}
    for sample in samples:
        folder = PUBLIC / sample
        task = json.loads((folder / "task.json").read_text())
        topology = json.loads((folder / "topology.json").read_text())
        graph = old["topology"].Topology.from_data(topology)
        baseline, construction = timed(lambda: old["solver"].solve_schedule(task, graph, time_budget_s=3))
        weights = baseline["strategy"]["constructed_route_weights"]
        if baseline["strategy"]["route_variation"]:
            (control, _), refinement = timed(lambda: old["route_balance"].refine(
                task, graph, baseline, weights=weights,
                routing_bytes=baseline["strategy"]["routing_bytes"], time_budget_s=0.25))
        else:
            control, refinement = baseline, {"wall_s": 0, "cpu_s": 0}
        (final, info), broad = timed(lambda: old["route_broad"].refine_broad(task, graph, control, time_budget_s=0.25))
        row = {"sample": sample, "version": "v1.2", "construction": construction,
               "refinement": refinement, "broad": broad,
               "wall_s": sum(stage["wall_s"] for stage in (construction, refinement, broad)),
               "cpu_s": sum(stage["cpu_s"] for stage in (construction, refinement, broad)),
               "seed_us": baseline["metrics"]["total_time"] * 1e6,
               "final_us": info["final_replay"]["total_time_us"], "valid": info["final_replay"]["valid"],
               "selected": next(r["actual"] for r in baseline["strategy"]["candidates"] if r["selected"])}
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
        for budget, seed_weight in product(budgets, seed_weights):
            gc.collect()
            graph = Topology.from_data(topology)
            split = task["split_policy"]["candidates"][-1] if task["operator"] == "all-reduce" else task["split_policy"]["candidates"][0]
            seed, seed_time = timed(lambda: solve_schedule(task, graph, seed_only=True, seed_split=split,
                                                         seed_weight=seed_weight))
            resumed, resume_time = timed(lambda: solve_schedule(task, graph, incumbent=seed, time_budget_s=budget))
            if weighted_budget and resumed["strategy"]["route_variation"]:
                (control, _), weighted_time = timed(lambda: refine(
                    task, graph, resumed, time_budget_s=weighted_budget,
                    weights=resumed["strategy"]["constructed_route_weights"],
                    routing_bytes=resumed["strategy"]["routing_bytes"]))
            else:
                control, weighted_time = resumed, {"wall_s": 0, "cpu_s": 0}
            (final, info), broad_time = timed(lambda: refine_broad(task, graph, control, time_budget_s=0.25))
            row = {"sample": sample, "version": "v1.3-staged", "resume_budget_s": budget,
                   "seed_weight": seed_weight, "weighted_budget_s": weighted_budget,
                   "seed": seed_time, "resume": resume_time, "weighted": weighted_time, "broad": broad_time,
                   "wall_s": sum(stage["wall_s"] for stage in (seed_time, resume_time, weighted_time, broad_time)),
                   "cpu_s": sum(stage["cpu_s"] for stage in (seed_time, resume_time, weighted_time, broad_time)),
                   "seed_us": seed["metrics"]["total_time"] * 1e6,
                   "resumed_us": resumed["metrics"]["total_time"] * 1e6,
                   "final_us": info["final_replay"]["total_time_us"], "valid": info["final_replay"]["valid"],
                   "broad_evidence": {k: v for k, v in info.items() if k != "final_replay"},
                   "selected": next(r["actual"] for r in resumed["strategy"]["candidates"] if r["selected"]),
                   "candidates": resumed["strategy"]["candidates"]}
            report["rows"].append(row)
            print(json.dumps({k: row[k] for k in ("sample", "version", "resume_budget_s", "seed_weight", "wall_s", "seed_us", "resumed_us", "final_us", "valid")}), flush=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
    report["source_sha256_at_end"] = source_hashes()
    report["source_changed"] = report["source_sha256"] != report["source_sha256_at_end"]
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    output.write_text(json.dumps(report, indent=2) + "\n")
    return all(row["valid"] for row in report["rows"]) and not report["source_changed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--seed-weights", nargs="+", type=float, default=[1.0])
    parser.add_argument("--weighted-budget", type=float, default=0.0)
    parser.add_argument("--output", default="artifacts/v1.3/analysis/routing/search-stages.json")
    args = parser.parse_args()
    output = (PROJECT / args.output).resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("output must stay inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min(1024**3, hard) if hard != resource.RLIM_INFINITY else 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    samples = args.samples or [str(p.parent.relative_to(PUBLIC)) for p in sorted(PUBLIC.glob("*/*/task.json"))]
    raise SystemExit(0 if run(output, args.budgets, samples, args.seed_weights, args.weighted_budget) else 1)
