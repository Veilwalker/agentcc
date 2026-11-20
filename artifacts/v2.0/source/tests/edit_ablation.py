"""Offline, bounded deterministic control for the SAME model-edit catalogue.

This is validation tooling, not a production fallback or evidence of LLM gains.
The search changes only the advertised group counts. It uses an independently
implemented exact round-cost objective and verifies its final choice through
the public edit compiler and the symbolic Replay.
"""

from collections import Counter
import argparse
import json
import math
from pathlib import Path
import time

from src.replay import replay
from src.schedule_edit import apply_proposal, extract_problem
from src.topology import Topology


class ExactEditObjective:
    """Evaluate prefix counts without using extractor effects or cost helpers."""

    def __init__(self, task, graph, baseline, context):
        self.graph, self.baseline, self.context = graph, baseline, context
        checked = replay(task, graph, baseline)
        if not checked["valid"]:
            raise ValueError("invalid ablation baseline")
        times = iter(checked["round_times_us"])
        self.round_times = [next(times) if round_["transfers"] else 0.0 for round_ in baseline["rounds"]]
        self.chunk_bytes = task["data_size"]["per_rank_bytes"] // checked["split"]
        self.affected = {group.round_index for group in context.groups}
        self.fixed_time = sum(value for ri, value in enumerate(self.round_times) if ri not in self.affected)
        self.paths = {}
        self.cache = {}
        self.proposal_cache = {}
        self.evaluations = 0

    def path_data(self, path):
        path = tuple(path)
        if path not in self.paths:
            edges = tuple(zip(path, path[1:]))
            bindings = Counter(resource for edge in edges for resource in self.graph.edge_resources[edge])
            overhead = self.graph.startup_us + (len(path) - 2) * self.graph.switch_overhead_us
            per_byte = sum(1e6 / self.graph.edges[edge] for edge in edges)
            self.paths[path] = edges, bindings, overhead, per_byte
        return self.paths[path]

    def __call__(self, counts):
        key = tuple(counts)
        if key in self.cache:
            return self.cache[key]
        moves = {}
        for group, amount in zip(self.context.groups, counts):
            remaining = amount
            for member in group.members:
                if not remaining:
                    break
                moved = min(remaining, member.chunk_count)
                moves[group.round_index, member.transfer_index] = (moved, member.path)
                remaining -= moved
        total = proposal_total = self.fixed_time
        for ri in self.affected:
            edge_bytes, resource_bytes = Counter(), Counter()
            max_path = max_overhead = 0.0
            for ti, transfer in enumerate(self.baseline["rounds"][ri]["transfers"]):
                original_path = transfer.get("path", transfer.get("route"))
                count = len(transfer["chunk_ranges"])
                moved, alternate = moves.get((ri, ti), (0, None))
                for amount, path in ((count - moved, original_path), (moved, alternate)):
                    if not amount:
                        continue
                    byte_count = amount * self.chunk_bytes
                    edges, bindings, overhead, per_byte = self.path_data(path)
                    max_path = max(max_path, overhead + byte_count * per_byte)
                    max_overhead = max(max_overhead, overhead)
                    for edge in edges:
                        edge_bytes[edge] += byte_count
                    for resource, multiplicity in bindings.items():
                        resource_bytes[resource] += byte_count * multiplicity
            busy = max([1e6 * count / self.graph.edges[edge] for edge, count in edge_bytes.items()]
                       + [1e6 * count / self.graph.resources[name] for name, count in resource_bytes.items()]
                       + [0.0])
            proposed = max(max_path, max_overhead + busy)
            original = self.round_times[ri]
            proposal_total += proposed
            # The compiler now restores a whole non-improving round. Apply the
            # same documented acceptance rule in this deterministic control;
            # do not pretend that its final cost is the unfiltered raw cost.
            total += proposed if proposed < original - max(1e-6, original * 1e-10) else original
        self.evaluations += 1
        self.cache[key] = total
        self.proposal_cache[key] = proposal_total
        return total

    def proposal_time_us(self, counts):
        self(counts)
        return self.proposal_cache[tuple(counts)]


def optimize_counts(task, graph, baseline, problem, context, *, time_budget_s=1.0):
    """Return (solution, proposal, evidence) for a bounded exact-cost control.

    Coordinate descent tests integer counts on a fixed grid plus nearby count
    changes. Joint initial allocations avoid depending on one improving edit.
    This is a reproducible heuristic, not an optimality certificate. Its budget
    includes objective preparation but final compilation/Replay always finish.
    Extraction and model calls are outside this function and must be accounted
    separately in an end-to-end comparison.
    """
    if type(time_budget_s) not in (int, float) or not math.isfinite(time_budget_s) or time_budget_s < 0:
        raise ValueError("invalid optimization budget")
    started = time.monotonic()
    end = started + time_budget_s
    objective = ExactEditObjective(task, graph, baseline, context)
    maxima = [group.max_chunks for group in context.groups]
    best = [0] * len(maxima)
    best_time = objective(best)

    def consider(counts):
        nonlocal best, best_time
        value = objective(counts)
        if value < best_time - 1e-9:
            best, best_time = list(counts), value

    for numerator in (2, 1, 3, 4):
        if time.monotonic() >= end:
            break
        consider([maximum * numerator // 4 for maximum in maxima])
    passes = 0
    for pass_number in range(3):
        previous = best_time
        for index, maximum in enumerate(maxima):
            values = {0, maximum, best[index]}
            values.update(maximum * numerator // 8 for numerator in range(1, 8))
            step = max(1, maximum // (8 * (pass_number + 1)))
            values.update((max(0, best[index] - step), min(maximum, best[index] + step)))
            for amount in sorted(values, key=lambda value: (abs(value - best[index]), value)):
                if time.monotonic() >= end:
                    break
                candidate = list(best)
                candidate[index] = amount
                consider(candidate)
            if time.monotonic() >= end:
                break
        passes += 1
        if time.monotonic() >= end or best_time >= previous - 1e-9:
            break
    proposal = {"base": problem["base"], "edits": [
        [group.id, count] for group, count in zip(context.groups, best) if count]}
    search_elapsed = time.monotonic() - started
    result, evidence = apply_proposal(task, graph, baseline, context, proposal)
    checked = replay(task, graph, result)
    if not checked["valid"]:
        raise AssertionError("ablation compiler returned an invalid result")
    if evidence["accepted"] and abs(checked["total_time_us"] - best_time) > max(1e-6, best_time * 1e-10):
        raise AssertionError("independent objective disagrees with final Replay")
    audit = {"method": "deterministic_integer_coordinate_search_same_catalog",
             "time_budget_s": time_budget_s, "search_elapsed_s": search_elapsed,
             "with_final_validation_s": time.monotonic() - started,
             "evaluations": objective.evaluations, "passes": passes,
             "accepted": evidence["accepted"], "before_us": context.before_us,
             "predicted_proposal_us": objective.proposal_time_us(best),
             "predicted_candidate_us": best_time, "returned_us": checked["total_time_us"],
             "compiler": evidence, "input_sha256": context.input_sha256,
             "baseline_sha256": context.baseline_sha256,
             "problem_sha256": context.problem_sha256}
    return result, proposal, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["mesh2d/ag", "mesh2d/ar"])
    parser.add_argument("--baseline-dir", default="artifacts/v1.1/runs/runtime-final")
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--extract-budget", type=float, default=0.5)
    parser.add_argument("--output", default="artifacts/v1.3/analysis/edit-ablation.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def local_path(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("ablation paths must stay inside the project")
        return path

    rows = []
    for case in args.cases:
        family, operator = case.split("/")
        folder = local_path(f"materials/spec/examples/256-rank/{family}/{operator}")
        task = json.loads((folder / "task.json").read_text())
        graph = Topology.from_data(json.loads((folder / "topology.json").read_text()))
        archived = json.loads((local_path(args.baseline_dir) / f"{family}_{operator}.json").read_text())
        baseline = archived.get("output", archived).get("answer", archived)
        started = time.monotonic()
        problem, context = extract_problem(task, graph, baseline, time_budget_s=args.extract_budget)
        extracted_s = time.monotonic() - started
        _, proposal, audit = optimize_counts(task, graph, baseline, problem, context, time_budget_s=args.budget)
        rows.append({"case": case, "extraction_s": extracted_s, "problem": problem,
                     "proposal": proposal, "optimizer": audit})
        print(json.dumps({"case": case, "groups": len(problem["groups"]), "before_us": audit["before_us"],
                          "after_us": audit["returned_us"], "evaluations": audit["evaluations"],
                          "accepted": audit["accepted"], "elapsed_s": time.monotonic() - started}), flush=True)
    local_path(args.output).write_text(json.dumps({
        "purpose": "Offline deterministic same-catalog control, not an LLM performance result", "cases": rows
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
