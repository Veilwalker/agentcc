"""Run cold-cache offline V22-like perturbations of all public input families.

This exercises real solver deadlines but makes no model/API calls. A fixed hint
stands in for a model strategy so runs are reproducible and credential-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from src.replay import replay
from src.solver import solve_schedule
from tests.test_regressions_v11 import PUBLIC, PROJECT, perturb_topology


def run(destination: Path, seeds: list[int], families: list[str], budget: float,
        preserve_rank_connectivity: bool = False) -> dict:
    source_hashes = {
        str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((PROJECT / "src").glob("*.py"))
    }
    report = {
        "scope": "Offline public-sample perturbations, not official hidden inputs or Judge results",
        "model_calls": 0, "model_hints": "deterministic offline test hints",
        "search_budget_s": budget, "cold_cache_per_case": True,
        "cpu_quota_enforced": False, "source_sha256": source_hashes,
        "rows": [],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for family in families:
            for code in ["ag", "ar"]:
                base = PUBLIC / family / code
                task = json.loads((base / "task.json").read_text())
                original = json.loads((base / "topology.json").read_text())
                changed, mutations = perturb_topology(original, seed, preserve_rank_connectivity=preserve_rank_connectivity)
                hint = {
                    "algorithm": ("recursive_doubling" if code == "ag" else "tree") if seed == 23 else
                                 ("reverse_doubling" if code == "ag" else "xor_tree"),
                    "split": max(task["split_policy"]["candidates"]),
                    "resource_weight": 4 if seed == 37 else 2,
                    "rank_order": "topology",
                }
                row = {
                    "family": family, "operator": code, "seed": seed,
                    "mutations": mutations, "requested_strategy": hint,
                    "input_sha256": hashlib.sha256(json.dumps({"task": task, "topology": changed}, sort_keys=True).encode()).hexdigest(),
                }
                started, cpu_started = time.monotonic(), time.process_time()
                try:
                    answer = solve_schedule(task, changed, hint, time_budget_s=budget)
                    row["solver_wall_s"] = round(time.monotonic() - started, 6)
                    row["solver_cpu_s"] = round(time.process_time() - cpu_started, 6)
                    verification = replay(task, changed, answer)
                    evidence = answer.get("strategy", {})
                    records = evidence.get("candidates", [])
                    winner = next((record for record in records if record.get("selected")), {})
                    row.update(
                        valid=verification["valid"], errors=verification["errors"],
                        rounds=answer["round_count"], split=answer.get("split"),
                        replay_time_us=verification["total_time_us"],
                        selected=winner.get("actual"), selected_source=winner.get("source"),
                        candidate_count=len(records), candidates=records,
                        solution_bytes=len(json.dumps(answer, separators=(",", ":")).encode()),
                    )
                except Exception as exc:
                    row.update(valid=False, errors=[f"{type(exc).__name__}: {exc}"],
                               solver_wall_s=round(time.monotonic() - started, 6),
                               solver_cpu_s=round(time.process_time() - cpu_started, 6))
                report["rows"].append(row)
                report["summary"] = {
                    "completed": len(report["rows"]),
                    "valid": sum(item["valid"] for item in report["rows"]),
                    "invalid": sum(not item["valid"] for item in report["rows"]),
                }
                destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps({key: row.get(key) for key in (
                    "family", "operator", "seed", "valid", "solver_wall_s", "solver_cpu_s", "rounds", "selected", "replay_time_us", "errors"
                )}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/v1.3/analysis/legacy-hint-mutations.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--families", nargs="+", default=sorted(path.name for path in PUBLIC.iterdir() if path.is_dir()))
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--preserve-rank-connectivity", action="store_true")
    args = parser.parse_args()
    report = run(PROJECT / args.output, args.seeds, args.families, args.budget, args.preserve_rank_connectivity)
    if report["summary"]["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
