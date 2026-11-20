"""Offline capacity-floor packetization experiment, not imported by production.

Only partition existing complete atomic items on their existing paths. Edge and
shared-resource byte totals, round data actions and overhead maxima stay fixed.
The per-transfer path serialization term can decrease. This uses the published
Replay formula; it does not treat a reduction expression as several packets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from src.replay import replay
from src.route_balance import _PathCosts, _RoundLoad
from src.topology import Topology


PROJECT = Path(__file__).resolve().parents[1]


def atoms(solution):
    return Counter((ri, t["action"], t["src"], t["dst"], tuple(t["path"]), atom)
                   for ri, row in enumerate(solution["rounds"])
                   for t in row["transfers"] for atom in t["chunk_ranges"])


def verify_atomic_preservation(before, after):
    """Optional exhaustive proof; keep this outside the normal planning cost."""
    if ({k: v for k, v in before.items() if k not in ("rounds", "metrics")}
            != {k: v for k, v in after.items() if k not in ("rounds", "metrics")}):
        raise ValueError("packetization changed solution metadata")
    old_rounds, new_rounds = before["rounds"], after["rounds"]
    if (len(old_rounds) != len(new_rounds)
            or any({k: v for k, v in old.items() if k != "transfers"}
                   != {k: v for k, v in new.items() if k != "transfers"}
                   for old, new in zip(old_rounds, new_rounds))
            or atoms(before) != atoms(after)):
        raise ValueError("packetization changed a data/path action")


def _plan_round(transfers, costs, chunk_bytes, target, before, resource_floor):
    sizes, extra, longest = [], 0, 0.0
    for transfer in transfers:
        n = len(transfer["chunk_ranges"])
        _, _, overhead, per_byte = costs.describe(transfer["path"])
        per_atom = per_byte * chunk_bytes
        maximum = max(1, min(n, math.floor((target - overhead) / per_atom + 1e-10)))
        packets = (n + maximum - 1) // maximum
        # This maximum size is optimal for the minimum feasible packet count.
        size = (n + packets - 1) // packets
        sizes.append(size)
        extra += packets - 1
        longest = max(longest, overhead + size * per_atom)
    # The target chooses a packet count; it is not a lower bound on its cost.
    after = max(resource_floor, longest)
    return {"sizes": sizes, "extra_packets": extra, "before_us": before,
            "target_us": target, "resource_floor_us": resource_floor,
            "max_path_us": longest, "after_us": after,
            "gain_us": max(0.0, before - after)}


def _baseline_round_times(solution, baseline_report):
    """Map an existing valid Replay report's nonempty rounds to round indices.

    The caller must bind this report to the supplied task/topology/solution.
    Checking the complete binding here would defeat the inexpensive no-op path.
    """
    if baseline_report is None:
        return None
    if not isinstance(baseline_report, dict):
        raise ValueError("baseline Replay report must be an object")
    values = baseline_report.get("round_times_us")
    nonempty = sum(bool(row["transfers"]) for row in solution["rounds"])
    if (baseline_report.get("valid") is not True or not isinstance(values, list)
            or len(values) != nonempty
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in values)):
        raise ValueError("expected a valid baseline Replay report with matching rounds")
    total = baseline_report.get("total_time_us")
    if (type(total) not in (int, float) or not math.isfinite(total)
            or not math.isclose(sum(values), total, rel_tol=1e-9, abs_tol=1e-6)):
        raise ValueError("baseline Replay round times disagree with its total")
    iterator = iter(values)
    return [next(iterator) if row["transfers"] else 0.0 for row in solution["rounds"]]


def repacketize(task, graph, solution, *, max_extra_packets=2048,
                baseline_report=None, verify_preservation=False):
    """Pure planning/application on an already-valid explicit-path solution.

    The greedy marginal frontier is an experiment, not an optimality claim.
    No Replay or model call occurs here. Costs and refreshed metrics are only
    predictions until the caller performs its normal final Replay. A supplied
    baseline_report must be the existing Replay of this exact input/solution.
    Atomic counters and hashing are optional/offline checks, not the default
    planning path. With no report, fast no-ops may return unknown (None) costs.
    """
    if type(max_extra_packets) is not int or max_extra_packets < 0:
        raise ValueError("max_extra_packets must be a nonnegative integer")
    started = time.process_time()
    known_times = _baseline_round_times(solution, baseline_report)
    before_packets = sum(len(row["transfers"]) for row in solution["rounds"])
    evidence = {"scope": "prediction_only; caller must perform final Replay",
                "replay_performed": False, "max_extra_packets": max_extra_packets,
                "before_transfer_count": before_packets, "after_transfer_count": before_packets,
                "extra_packets": 0, "actual_extra_packets": 0, "plans": [],
                "before_us": sum(known_times) if known_times is not None else None,
                "predicted_after_us": sum(known_times) if known_times is not None else None,
                "predicted_round_times_us": known_times,
                "rounds_load_evaluated": 0, "rounds_skipped_by_replay": 0}

    def finish(candidate, reason):
        evidence.update(reason=reason, algorithm_cpu_s=time.process_time() - started,
                        preservation_checked=bool(verify_preservation))
        proof_started = time.process_time()
        if verify_preservation:
            verify_atomic_preservation(solution, candidate)
        evidence["proof_cpu_s"] = time.process_time() - proof_started if verify_preservation else 0.0
        return candidate, evidence

    if max_extra_packets == 0:
        return finish(solution, "zero_budget")
    if not any(len(t["chunk_ranges"]) > 1 for row in solution["rounds"] for t in row["transfers"]):
        return finish(solution, "no_splittable_transfer")
    chunk_bytes = task["data_size"]["per_rank_bytes"] // solution["split"]
    costs, frontiers = _PathCosts(graph), {}
    original_times = []
    for ri, row in enumerate(solution["rounds"]):
        transfers = row["transfers"]
        if known_times is not None:
            # If all splittable paths are already below this round's duration,
            # its unchanged resource floor or an indivisible path dominates.
            longest_splittable = 0.0
            for transfer in transfers:
                count = len(transfer["chunk_ranges"])
                if count > 1:
                    _, _, overhead, per_byte = costs.describe(transfer["path"])
                    longest_splittable = max(longest_splittable, overhead + per_byte * chunk_bytes * count)
            if not transfers or longest_splittable < known_times[ri] - max(1e-6, known_times[ri] * 1e-10):
                original_times.append(known_times[ri])
                evidence["rounds_skipped_by_replay"] += 1
                continue
        load = _RoundLoad(costs)
        atomic_path = 0.0
        for t in transfers:
            load.add(t["path"], len(t["chunk_ranges"]) * chunk_bytes)
            _, _, overhead, per_byte = costs.describe(t["path"])
            atomic_path = max(atomic_path, overhead + per_byte * chunk_bytes)
        before = load.time_us
        evidence["rounds_load_evaluated"] += 1
        if known_times is not None and not math.isclose(before, known_times[ri], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("baseline Replay does not match the modeled round cost")
        original_times.append(before)
        resource_floor = load.overhead_max + load.busy_max
        floor = max(resource_floor, atomic_path)
        if before <= floor + max(1e-6, before * 1e-10):
            continue
        plans = []
        seen = set()
        for fraction in (0.25, 0.5, 0.75, 1.0):
            target = before - (before - floor) * fraction
            plan = _plan_round(transfers, costs, chunk_bytes, target, before, resource_floor)
            signature = tuple(plan["sizes"])
            if plan["gain_us"] > 1e-6 and plan["extra_packets"] and signature not in seen:
                seen.add(signature)
                plan.update(round_index=ri, atomic_floor_us=atomic_path)
                plans.append(plan)
        if plans:
            frontiers[ri] = plans
    selected, used = {}, 0
    while True:
        choices = []
        for ri, plans in frontiers.items():
            previous = selected.get(ri, {"gain_us": 0.0, "extra_packets": 0})
            for plan in plans:
                extra = plan["extra_packets"] - previous["extra_packets"]
                gain = plan["gain_us"] - previous["gain_us"]
                if gain > 1e-6 and extra >= 0 and used + extra <= max_extra_packets:
                    choices.append((-gain / max(1, extra), -gain, extra, ri, plan))
        if not choices:
            break
        _, _, extra, ri, plan = min(choices, key=lambda item: item[:4])
        selected[ri] = plan
        used += extra
    rounds = list(solution["rounds"])
    predicted_times = original_times.copy()
    for ri, plan in selected.items():
        transfers = []
        for transfer, size in zip(rounds[ri]["transfers"], plan["sizes"]):
            chunks = transfer["chunk_ranges"]
            for begin in range(0, len(chunks), size):
                transfers.append({**transfer, "chunk_ranges": chunks[begin:begin + size]})
        rounds[ri] = {**rounds[ri], "transfers": transfers}
        predicted_times[ri] = plan["after_us"]
    result = {**solution, "rounds": rounds} if selected else solution
    after_packets = sum(len(row["transfers"]) for row in result["rounds"])
    if after_packets - before_packets != used or used > max_extra_packets:
        raise ValueError("packet budget disagrees with the actual transfer count")
    evidence.update(before_us=sum(original_times), predicted_after_us=sum(predicted_times),
                    predicted_round_times_us=predicted_times, after_transfer_count=after_packets,
                    extra_packets=used, actual_extra_packets=after_packets - before_packets,
                    plans=[{k: v for k, v in plan.items() if k != "sizes"}
                           for _, plan in sorted(selected.items())])
    if selected:
        result["metrics"] = {**solution.get("metrics", {}),
                             "total_time": sum(predicted_times) / 1e6,
                             "round_times": [value / 1e6 for value, row in zip(predicted_times, rounds)
                                             if row["transfers"]]}
    return finish(result, "planned" if selected else "no_profitable_plan_within_budget")


def _sha256(body):
    return hashlib.sha256(body).hexdigest()


def _json_sha256(value):
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def run(archive, output, budgets):
    archive, output = archive.resolve(), output.resolve()
    if not archive.is_relative_to(PROJECT) or not output.is_relative_to(PROJECT):
        raise ValueError("experiment paths must stay inside project")
    if output.exists():
        raise ValueError("experiment output already exists")
    if not budgets or any(type(value) is not int or value < 0 for value in budgets):
        raise ValueError("budgets must be nonnegative integers")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = archive / "summary.json"
    source = json.loads(summary_path.read_text())
    hashes = {str(p.relative_to(PROJECT)): _sha256(p.read_bytes())
              for p in sorted((PROJECT / "src").glob("*.py"))}
    tool_path = Path(__file__).resolve()
    evidence_files = {summary_path: _sha256(summary_path.read_bytes())}
    report = {"scope": "Offline same-path packetization of saved v1.4 answers; no network or LLM calls",
              "source_sha256": hashes, "tool_sha256": {str(tool_path.relative_to(PROJECT)): _sha256(tool_path.read_bytes())},
              "archive_summary_sha256": evidence_files[summary_path],
              "budgets": budgets, "rows": [], "baseline_replay_calls": 0, "candidate_replay_calls": 0,
              "timing": "algorithm_cpu_s excludes optional atom proof and Replay; total_cpu_s includes both; baseline Replay is measured separately"}
    seen = set()
    for row in source["rows"]:
        if row["version"] != "1.4" or row["sample"] in seen:
            continue
        seen.add(row["sample"])
        input_path = archive / "inputs" / (row["sample"].replace("/", "-") + ".json")
        response_path = archive / row["response_file"]
        if not input_path.resolve().is_relative_to(PROJECT) or not response_path.resolve().is_relative_to(PROJECT):
            raise ValueError("saved experiment inputs must stay inside project")
        evidence_files[input_path] = _sha256(input_path.read_bytes())
        evidence_files[response_path] = _sha256(response_path.read_bytes())
        request = json.loads(input_path.read_text())
        actual = request.get("input", request)
        task, graph = actual["task"], Topology.from_data(actual["topology"])
        solution = json.loads(response_path.read_text())["output"]["answer"]
        input_sha, answer_sha = _json_sha256(actual), _json_sha256(solution)
        baseline_started = time.process_time()
        baseline = replay(task, graph, solution)
        baseline_cpu = time.process_time() - baseline_started
        report["baseline_replay_calls"] += 1
        if not baseline["valid"]:
            raise ValueError("archived baseline failed Replay: " + "; ".join(baseline["errors"]))
        for budget in budgets:
            started = time.process_time()
            candidate, evidence = repacketize(task, graph, solution, max_extra_packets=budget,
                                              baseline_report=baseline, verify_preservation=True)
            checked = replay(task, graph, candidate)
            report["candidate_replay_calls"] += 1
            total_cpu = time.process_time() - started
            if not checked["valid"]:
                raise ValueError("packet candidate failed Replay: " + "; ".join(checked["errors"]))
            if not math.isclose(checked["total_time_us"], evidence["predicted_after_us"], rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("packet prediction disagrees with Replay")
            predicted_nonempty = [value for value, r in zip(evidence["predicted_round_times_us"], candidate["rounds"])
                                  if r["transfers"]]
            if (len(predicted_nonempty) != len(checked["round_times_us"])
                    or any(not math.isclose(predicted, actual_time, rel_tol=1e-9, abs_tol=1e-6)
                           for predicted, actual_time in zip(predicted_nonempty, checked["round_times_us"]))):
                raise ValueError("per-round packet predictions disagree with Replay")
            if checked["total_time_us"] > baseline["total_time_us"] + max(1e-6, baseline["total_time_us"] * 1e-10):
                raise ValueError("packetization increased communication cost")
            inputs_unchanged = _json_sha256(actual) == input_sha
            answer_unchanged = _json_sha256(solution) == answer_sha
            if not inputs_unchanged or not answer_unchanged:
                raise ValueError("experiment mutated its input or archived answer")
            result = {"sample": row["sample"], **evidence, "valid": checked["valid"],
                      "scope": "offline candidate checked by complete local Replay",
                      "replay_performed": True, "after_us": checked["total_time_us"],
                      "round_times_us": checked["round_times_us"], "total_cpu_s": total_cpu,
                      "baseline_replay_cpu_s": baseline_cpu, "input_sha256": input_sha,
                      "before_answer_sha256": answer_sha, "after_answer_sha256": _json_sha256(candidate),
                      "source_input_file": str(input_path.relative_to(PROJECT)),
                      "source_input_file_sha256": evidence_files[input_path],
                      "source_response_file": str(response_path.relative_to(PROJECT)),
                      "source_response_file_sha256": evidence_files[response_path],
                      "inputs_unchanged": inputs_unchanged, "answer_unchanged": answer_unchanged,
                      "before_bytes": len(json.dumps(solution, separators=(",", ":")).encode()),
                      "after_bytes": len(json.dumps(candidate, separators=(",", ":")).encode())}
            report["rows"].append(result)
            print(json.dumps({k: result[k] for k in ("sample", "max_extra_packets", "before_us", "after_us", "extra_packets", "total_cpu_s")}), flush=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
    report["source_unchanged"] = all(_sha256((PROJECT / p).read_bytes()) == value
                                      for p, value in hashes.items())
    report["tool_unchanged"] = all(_sha256((PROJECT / p).read_bytes()) == value
                                     for p, value in report["tool_sha256"].items())
    report["archive_files_unchanged"] = all(_sha256(path.read_bytes()) == value
                                              for path, value in evidence_files.items())
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=PROJECT / "artifacts/v1.4/runs/public-narrow")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/v1.5/analysis/packetization/pilot.json")
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    arguments = parser.parse_args()
    run(arguments.archive, arguments.output, arguments.budgets)
