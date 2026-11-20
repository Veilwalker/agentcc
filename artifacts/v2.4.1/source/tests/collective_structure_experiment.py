"""Offline, bounded collective-structure experiments; no production changes.

Bit permutations change the physical XOR dimensions used by the existing
constructor. Optional AR root reassignment applies a bijection to each entire
slot tree, including endpoints and every contributor, then rebuilds real routes.
All candidates are replayed before and after the same fixed-work broad routing.
Reported elapsed times are research observations, not online latency claims.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from src.collectives import build, locality_order
from src.replay import replay
from src.route_balance import _PathCosts, _RoundLoad
from src.route_broad import refine_broad
from src.route_work import RouteWorkBudget
from src.topology import Topology


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT / "artifacts/v1.4/runs/public-narrow"
DEFAULT_OUTPUT = PROJECT / "artifacts/v1.5/analysis/structure"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((PROJECT / "src").glob("*.py"))}


def permute_bits(index, dimensions):
    return sum(((index >> logical) & 1) << physical
               for logical, physical in enumerate(dimensions))


def dimension_order(order, dimensions):
    n = len(order)
    if not n or n & (n - 1):
        raise ValueError("rank count must be a positive power of two")
    width = (n - 1).bit_length()
    if (len(dimensions) != width or any(type(bit) is not int for bit in dimensions)
            or sorted(dimensions) != list(range(width))):
        raise ValueError("dimensions must be a permutation of the rank index bits")
    return [order[permute_bits(index, dimensions)] for index in range(n)]


def default_roots(n, split):
    stride = n // split + 1
    if stride % 2 == 0:
        stride += 1
    return [(slot * stride) % n for slot in range(split)]


def _slot(token):
    if token.startswith("c") and token.endswith("_sum"):
        return int(token[1:-4])
    return int(token.split("+", 1)[0].rsplit("_c", 1)[1])


def _map_expression(token, mapping):
    if token.startswith("c") and token.endswith("_sum"):
        return token
    result = []
    for term in token.split("+"):
        owner, slot = term.rsplit("_c", 1)
        result.append(f"{mapping[owner]}_c{slot}")
    return "+".join(result)


def build_structure(task, graph, split, *, dimensions=None, roots=None,
                    order=None, packets=2, resource_weight=4.0):
    """Reuse the constructor, optionally mapping complete AR trees per slot.

    Roots are indices in the bit-permuted rank order. Moving a root does not
    rename slots, change a contribution's slot, or divide an AR expression.
    """
    original_order = list(graph.ranks) if order is None else list(order)
    dimensions = tuple(range((len(original_order) - 1).bit_length())) if dimensions is None else tuple(dimensions)
    ordered = dimension_order(original_order, dimensions)
    kind = "xor_tree" if task["operator"] == "all-reduce" else "reverse_doubling"
    routing_bytes = task["data_size"]["per_rank_bytes"] // split
    answer = build(task, graph, split, kind=kind, order=ordered, packets=packets,
                   resource_weight=resource_weight, routing_bytes=routing_bytes)
    if roots is None:
        return answer
    if task["operator"] != "all-reduce":
        raise ValueError("per-slot roots are only defined for all-reduce")
    if (len(roots) != split or any(type(root) is not int or not 0 <= root < len(ordered)
                                   for root in roots)):
        raise ValueError("one valid integer root index is required per slot")
    old_roots = default_roots(len(ordered), split)
    mappings = [{rank: ordered[index ^ old ^ target] for index, rank in enumerate(ordered)}
                for old, target in zip(old_roots, roots)]
    rebuilt = []
    for round_ in answer["rounds"]:
        movements = defaultdict(list)
        for transfer in round_["transfers"]:
            for token in transfer["chunk_ranges"]:
                mapping = mappings[_slot(token)]
                key = transfer["action"], mapping[transfer["src"]], mapping[transfer["dst"]]
                movements[key].append(_map_expression(token, mapping))
        transfers = []
        for (action, src, dst), tokens in movements.items():
            path = list(graph.path(src, dst, size_bytes=routing_bytes, resource_weight=resource_weight))
            packet_size = max(1, (len(tokens) + packets - 1) // packets)
            for first in range(0, len(tokens), packet_size):
                transfers.append({"action": action, "src": src, "dst": dst,
                                  "chunk_ranges": tokens[first:first + packet_size], "path": path.copy()})
        rebuilt.append({**round_, "transfers": transfers})
    return {**answer, "rounds": rebuilt}


def traffic_rounds(solution):
    """Round-wise physical payload signatures, ignoring equal-size slot labels.

    Equality implies the same current replay cost and same broad-routing work
    problem, not an assertion that arbitrary schedules have equal data state.
    Each schedule must independently pass replay before this deduplication.
    """
    return [sorted((transfer["action"], transfer["src"], transfer["dst"],
                    tuple(transfer["path"]), len(transfer["chunk_ranges"]))
                   for transfer in round_["transfers"])
            for round_ in solution["rounds"]]


def equivalent_round_permutation(left, right):
    return Counter(digest(row) for row in traffic_rounds(left)) == Counter(
        digest(row) for row in traffic_rounds(right))


def dimension_costs(task, graph, order, split):
    """Measure actual simultaneous pair-exchange pressure for each rank bit."""
    width = (len(order) - 1).bit_length()
    costs = _PathCosts(graph)
    payload = task["data_size"]["per_rank_bytes"] // split
    values = []
    for bit in range(width):
        load = _RoundLoad(costs)
        for index, src in enumerate(order):
            path = graph.path(src, order[index ^ (1 << bit)], size_bytes=payload, resource_weight=4.0)
            load.add(path, payload)
        values.append(load.time_us)
    return values


def candidate_specs(task, graph, split, catalogue="broad"):
    """A fixed small catalogue driven only by ranks, edges and capacities."""
    n = len(graph.ranks)
    identity = tuple(range((n - 1).bit_length()))
    pressure = dimension_costs(task, graph, graph.ranks, split)
    if catalogue not in {"broad", "local"}:
        raise ValueError("unknown bounded structure catalogue")
    rank_orders = [("input", list(graph.ranks)), ("locality", locality_order(graph))]
    dimensions = [("identity", identity), ("reverse", identity[::-1]),
                  ("rotate_one", identity[1:] + identity[:1]),
                  ("rotate_half", identity[len(identity) // 2:] + identity[:len(identity) // 2]),
                  ("pressure_ascending", tuple(sorted(identity, key=lambda bit: (pressure[bit], bit))))]
    if catalogue == "local":
        rank_orders = rank_orders[:1]
        dimensions = [("identity", identity)]
        for first in range(len(identity) - 1):
            swapped = list(identity)
            swapped[first], swapped[first + 1] = swapped[first + 1], swapped[first]
            dimensions.append((f"swap_{first}_{first + 1}", tuple(swapped)))
    specs, seen = [], set()
    for rank_name, order in rank_orders:
        for dimension_name, dims in dimensions if rank_name == "input" else dimensions[:1]:
            key = tuple(dimension_order(order, dims))
            if key in seen:
                continue
            seen.add(key)
            specs.append({"name": f"{rank_name}/{dimension_name}", "order": order,
                          "dimensions": list(dims), "roots": None})
    if task["operator"] == "all-reduce":
        root_patterns = [("spread", [(slot * n // split) % n for slot in range(split)]),
                         ("contiguous", [slot % n for slot in range(split)]),
                         ("orientation_flip", [root ^ (1 if n > 1 else 0)
                                               for root in default_roots(n, split)])]
        if catalogue == "local":
            strides = sorted({3, 5, 7, max(1, n // split - 1) | 1, (n // split + 3) | 1})
            root_patterns = [(f"stride_{stride}", [(slot * stride) % n for slot in range(split)])
                             for stride in strides]
        seen_roots = {tuple(sorted(default_roots(n, split)))}
        for name, roots in root_patterns:
            # Equal-sized slots are interchangeable; equal root multisets add
            # no new aggregate routing problem for this XOR constructor.
            key = tuple(sorted(roots))
            if key not in seen_roots:
                seen_roots.add(key)
                specs.append({"name": f"roots/{name}", "order": list(graph.ranks),
                              "dimensions": list(identity), "roots": roots})
    return specs, pressure


def _recorded_cases(source, names):
    summary = json.loads((source / "summary.json").read_text())
    first = {}
    for row in summary["rows"]:
        if row["version"] == "1.4" and row["sample"] in names:
            first.setdefault(row["sample"], row)
    if set(first) != set(names):
        raise ValueError("requested cases must have saved v1.4 responses")
    return first


def run(source, output, names, catalogue="broad"):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": __doc__, "catalogue": catalogue,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256_at_start": source_hashes(), "rows": [],
              "routing_policy": {"resource_weight": 4.0, "packets": 2,
                 "broad_path_scores": 20_000, "broad_edge_relaxations": 65_000,
                 "broad_wall_safety_s": 30.0},
              "comparison_limits": ["No model calls; no online latency claim.",
                 "Candidates keep the recorded winner split fixed to isolate structure.",
                 "All identity/variant candidates receive the same broad-routing work cap.",
                 "Saved final baseline is separately compared; it already includes search/model/routing.",
                 "Saved final plus another broad pass is an additional routes-only control."]}
    for name, reference in _recorded_cases(source, names).items():
        case_name = name.replace("/", "-")
        case = json.loads((source / "inputs" / f"{case_name}.json").read_text())
        case = case.get("input", case)
        input_sha = digest(case)
        saved = json.loads((source / reference["response_file"]).read_text())["output"]["answer"]
        graph = Topology.from_data(case["topology"])
        checked = replay(case["task"], graph, saved)
        if not checked["valid"]:
            raise ValueError(checked)
        specs, pressure = candidate_specs(case["task"], graph, saved["split"], catalogue)
        # Use a generous exceptional wall bound so normal stopping is the
        # identical work limit even while other research agents share the CPU.
        budget = RouteWorkBudget(20_000, 65_000, safety_s=30.0)
        extra, extra_info = refine_broad(case["task"], Topology.from_data(case["topology"]),
                                        deepcopy(saved), work_budget=budget)
        extra_report = extra_info.pop("final_replay")
        case_row = {"sample": name, "input_sha256": input_sha, "split": saved["split"],
                    "recorded_final_time_us": checked["total_time_us"],
                    "recorded_final_extra_routing_us": extra_report["total_time_us"],
                    "recorded_extra_routing_evidence": extra_info,
                    "dimension_exchange_pressure_us": pressure, "candidates": []}
        identity, seen_traffic = None, {}
        best = None
        for spec in specs:
            graph = Topology.from_data(case["topology"])
            started = time.monotonic()
            raw = build_structure(case["task"], graph, saved["split"], **{
                key: spec[key] for key in ("order", "dimensions", "roots")})
            construction_s = time.monotonic() - started
            raw_report = replay(case["task"], graph, raw)
            if not raw_report["valid"]:
                raise ValueError({"case": name, "spec": spec, "report": raw_report})
            if identity is None:
                identity = raw
            signature = digest(traffic_rounds(raw))
            row = {"name": spec["name"], "spec": spec, "raw_valid": True,
                   "raw_time_us": raw_report["total_time_us"], "construction_s": construction_s,
                   "traffic_sha256": signature,
                   "only_identity_round_permutation": equivalent_round_permutation(identity, raw)}
            if signature in seen_traffic:
                row["duplicate_traffic_of"] = seen_traffic[signature]
            else:
                seen_traffic[signature] = spec["name"]
                started = time.monotonic()
                final, info = refine_broad(case["task"], graph, raw, work_budget=budget)
                final_report = info.pop("final_replay")
                row.update(valid=final_report["valid"], routed_time_us=final_report["total_time_us"],
                           routing_s=time.monotonic() - started, routing_evidence=info,
                           program_sha256=digest({k: v for k, v in final.items() if k != "metrics"}),
                           vs_recorded_gain_us=checked["total_time_us"] - final_report["total_time_us"],
                           vs_recorded_extra_routing_gain_us=extra_report["total_time_us"] - final_report["total_time_us"])
                if best is None or final_report["total_time_us"] < best[0]:
                    best = final_report["total_time_us"], spec["name"], final
            case_row["candidates"].append(row)
            print(json.dumps({"sample": name, **{k: row[k] for k in
                  ("name", "raw_time_us", "routed_time_us", "duplicate_traffic_of") if k in row}}), flush=True)
        identity_cost = case_row["candidates"][0]["routed_time_us"]
        for row in case_row["candidates"]:
            if "routed_time_us" in row:
                row["vs_identity_same_routing_gain_us"] = identity_cost - row["routed_time_us"]
        case_row.update(best_name=best[1], best_time_us=best[0], input_unchanged=digest(case) == input_sha)
        (output / f"{case_name}-best.json").write_text(json.dumps(best[2]) + "\n")
        report["rows"].append(case_row)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    report.update(source_sha256_at_end=source_hashes(), complete=True)
    report["source_unchanged"] = report["source_sha256_at_start"] == report["source_sha256_at_end"]
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", default="mesh2d/ag,mesh2d/ar,cloudmatrix/ag,cloudmatrix/ar")
    parser.add_argument("--catalogue", choices=["broad", "local"], default="broad")
    args = parser.parse_args()
    if any(not path.resolve().is_relative_to(PROJECT) for path in (args.source, args.output)):
        parser.error("all paths must remain in the project")
    run(args.source.resolve(), args.output.resolve(), args.cases.split(","), args.catalogue)
