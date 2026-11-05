"""Reproducible local solve, replay and sample-suite commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from . import __version__
from .replay import replay
from .service import solve_input


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def solve_pair(task_path, topology_path, *, offline=False, budget=1.0):
    task, topology = read(task_path), read(topology_path)
    output = solve_input({"task": task, "topology": topology,
                          "instance_id": task.get("instance_id")},
                         use_llm=not offline, search_budget_s=budget)
    return {"task_id": task["task_id"], "status": "success", "output": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--task", required=True)
    solve.add_argument("--topology", required=True)
    solve.add_argument("--output", required=True)
    solve.add_argument("--offline", action="store_true", help="Explicit algorithm-only test; no LLM")
    solve.add_argument("--budget", type=float, default=1.0)
    verify = commands.add_parser("replay")
    verify.add_argument("--task", required=True)
    verify.add_argument("--topology", required=True)
    verify.add_argument("--solution", required=True)
    suite = commands.add_parser("benchmark")
    suite.add_argument("--samples", default="materials/spec/examples/256-rank")
    suite.add_argument("--output-dir", default=f"artifacts/v{__version__}/runs/offline")
    suite.add_argument("--offline", action="store_true")
    suite.add_argument("--budget", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "replay":
        result = replay(read(args.task), read(args.topology), read(args.solution))
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["valid"] else 1
    if args.command == "solve":
        response = solve_pair(args.task, args.topology, offline=args.offline, budget=args.budget)
        write(args.output, response)
        print(json.dumps(response["output"]["diagnostics"]["local_replay"]))
        return 0
    records = []
    paths = sorted(Path(args.samples).glob("*/*/task.json"))
    if not paths:
        parser.error("No sample tasks found")
    for task_path in paths:
        started = time.monotonic()
        label = task_path.parent.parent.name + "/" + task_path.parent.name
        record = {"sample": label, "llm_enabled": not args.offline}
        try:
            response = solve_pair(task_path, task_path.with_name("topology.json"),
                                  offline=args.offline, budget=args.budget)
            name = label.replace("/", "_") + ".json"
            write(Path(args.output_dir) / name, response)
            report = response["output"]["diagnostics"]["local_replay"]
            record.update(report)
            record["response_bytes"] = (Path(args.output_dir) / name).stat().st_size
            record["selected"] = next(c["actual"] for c in
                response["output"]["diagnostics"]["search"]["candidates"] if c["selected"])
        except Exception as error:
            # Do not expose exception text from transports, headers or configuration.
            record.update(valid=False, error_type=type(error).__name__)
        record["wall_time_s"] = round(time.monotonic() - started, 6)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        write(Path(args.output_dir) / "summary.json", records)
    return 0 if all(r["valid"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
