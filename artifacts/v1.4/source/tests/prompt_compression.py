"""Offline, reversible prompt-format comparison; never calls a model service.

Historical prompt_tokens are service measurements of v1.3 only. New formats
report bytes, not estimated tokens. They need a separate real API comparison.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def integer_numbers(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [integer_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: integer_numbers(item) for key, item in value.items()}
    return value


def dense_candidate(compact):
    """Prototype A: lossless renaming plus arrays, without shared vector tables."""
    nodes, pools, group_ids = [], [], []
    node_indices, pool_indices = {}, {}

    def alias(value, names, indices):
        if value not in indices:
            indices[value] = len(names)
            names.append(value)
        return indices[value]

    def node(value):
        return alias(value, nodes, node_indices)

    def column_key(position):
        name, capacity = compact["constraints"][position]
        return (0, name, capacity) if isinstance(name, str) else (1, tuple(name), capacity)

    column_order = sorted(range(len(compact["constraints"])), key=column_key)
    constraints = []
    for position in column_order:
        name, capacity = compact["constraints"][position]
        if isinstance(name, str):
            constraints.append([alias(name, pools, pool_indices), capacity])
        else:
            constraints.append([node(name[0]), node(name[1]), capacity])
    rounds = [[row[key] for key in ("round", "time", "path", "overhead", "floor")]
              + [[row["loads"][position] for position in column_order]]
              for row in compact["rounds"]]
    groups = []
    for row in compact["groups"]:
        group_ids.append(row["id"])
        groups.append([row["round"], row["max_chunks"], [row["delta"][position] for position in column_order],
                       [[node(name) for name in path] for path in row["paths"]], row["bounds"]])
    wire = {"base": compact["base"], "chunk_bytes": compact.get("chunk_bytes"),
            "constraints": constraints, "rounds": rounds, "groups": groups}
    if "topology" in compact:
        wire["topology"] = deepcopy(compact["topology"])
    return integer_numbers(wire), {"nodes": nodes, "pools": pools, "group_ids": group_ids,
                                   "column_order": column_order}


def factored_candidate(compact):
    """Prototype B: A plus exact deduplication of delta/path/bound vectors."""
    wire, aliases = dense_candidate(compact)
    tables, indices = {"deltas": [], "paths": [], "bounds": []}, {"deltas": {}, "paths": {}, "bounds": {}}

    def intern(name, value):
        identity = dumps(value)
        if identity not in indices[name]:
            indices[name][identity] = len(tables[name])
            tables[name].append(value)
        return indices[name][identity]

    wire["groups"] = [[row[0], row[1], intern("deltas", row[2]),
                       [intern("paths", path) for path in row[3]], intern("bounds", row[4])]
                      for row in wire["groups"]]
    wire.update(tables)
    return wire, aliases


def unpack_candidate(wire, aliases, *, factored=False):
    """Invert v2 positional or v3 named groups, preserving every value/identity."""
    order = aliases.get("column_order", list(range(len(wire["constraints"]))))

    def original_order(values):
        result = [None] * len(order)
        for position, value in zip(order, values):
            result[position] = value
        return result

    compact = {"base": wire["base"], "chunk_bytes": wire["chunk_bytes"],
               "constraints": [], "rounds": [], "groups": []}
    for row in wire["constraints"]:
        name = aliases["pools"][row[0]] if len(row) == 2 else [aliases["nodes"][row[0]], aliases["nodes"][row[1]]]
        compact["constraints"].append([name, row[-1]])
    compact["constraints"] = original_order(compact["constraints"])
    for row in wire["rounds"]:
        restored = dict(zip(("round", "time", "path", "overhead", "floor", "loads"), row))
        restored["loads"] = original_order(restored["loads"])
        compact["rounds"].append(restored)
    for i, row in enumerate(wire["groups"]):
        round_, maximum, delta, paths, bounds = (
            [row[key] for key in ("round", "max_chunks", "delta", "paths", "bounds")]
            if isinstance(row, dict) else row)
        if factored:
            delta = wire["deltas"][delta]
            paths = [wire["paths"][index] for index in paths]
            bounds = wire["bounds"][bounds]
        compact["groups"].append({"id": aliases["group_ids"][i], "round": round_, "max_chunks": maximum,
                                  "delta": original_order(delta),
                                  "paths": [[aliases["nodes"][index] for index in path] for path in paths],
                                  "bounds": bounds})
    if "topology" in wire:
        compact["topology"] = deepcopy(wire["topology"])
    return compact


def compare(root: Path, output: Path):
    from src.llm import build_edit_request
    rows = []
    for response_path in sorted(root.glob("runs/public-*/v1.3/*.json")):
        response = json.loads(response_path.read_text())
        model = response["output"]["diagnostics"]["llm"]
        messages = model["request_evidence"]["messages"]
        supplied = json.loads(messages[1]["content"])
        original = supplied["problem"]
        record = {"source": str(response_path), "historical_service_prompt_tokens": model.get("usage", {}).get("prompt_tokens"),
                  "groups": len(original["groups"]), "rounds": len(original["rounds"]),
                  "v13_problem_bytes": len(dumps(original).encode()),
                  "v13_messages_bytes": sum(len(m["content"].encode()) for m in messages), "variants": {}}
        for name, packer in (("dense", dense_candidate), ("factored", factored_candidate)):
            wire, aliases = packer(original)
            restored = unpack_candidate(wire, aliases, factored=name == "factored")
            if restored != original:
                raise AssertionError(f"{name} lost semantics: {response_path}")
            record["variants"][name] = {"problem_bytes": len(dumps(wire).encode()), "exact_roundtrip": True}
            if response_path.stem.endswith("r01"):
                target = output / "examples" / f"{response_path.stem}-{name}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({"wire": wire, "aliases": aliases}, ensure_ascii=False, indent=2) + "\n")
        # The actual builder consumes extractor fields. This inverse restores
        # the recorded finite-precision view, not an unobserved exact problem.
        # It is used only to compare request bytes; no schedule or reply is made.
        input_path = response_path.parent.parent / "inputs" / (response_path.stem.rsplit("-r", 1)[0] + ".json")
        if input_path.exists():
            task_input = json.loads(input_path.read_text())
            task = task_input.get("task", task_input.get("input", {}).get("task"))
            restored_problem = restore_projection_as_problem(original)
            system, user, audit = build_edit_request(task, restored_problem)
            record["dense_messages_bytes"] = len(system.encode()) + len(user.encode())
            record["dense_system_bytes"] = len(system.encode())
            if unpack_candidate(json.loads(user)["problem"], audit["aliases"]) != original:
                raise AssertionError(f"request builder changed recorded projection: {response_path}")
            if response_path.stem.endswith("r01"):
                target = output / "examples" / f"{response_path.stem}-request.json"
                target.write_text(json.dumps({"system": system, "user": user, "encoding_audit": audit},
                                            ensure_ascii=False, indent=2) + "\n")
        rows.append(record)
    summary = {"scope": "offline projection of recorded v1.3 requests; no model calls or synthetic replies",
               "new_service_prompt_tokens": None, "token_note": "Only historical prompt_tokens are measured. New bytes are not token estimates.",
               "requests": len(rows), "rows": rows,
               "total_problem_bytes": {"v13": sum(r["v13_problem_bytes"] for r in rows),
                   **{variant: sum(r["variants"][variant]["problem_bytes"] for r in rows) for variant in ("dense", "factored")}},
               "total_message_bytes": {"v13": sum(r["v13_messages_bytes"] for r in rows),
                                       "dense": sum(r.get("dense_messages_bytes", 0) for r in rows)}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "format-comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    message_comparison = {"scope": summary["scope"], "requests": len(rows),
                          "old_bytes": summary["total_message_bytes"]["v13"],
                          "new_bytes": summary["total_message_bytes"]["dense"],
                          "new_service_tokens": None,
                          "note": "Dense current system+user bytes; long-system intermediate is separately archived.",
                          "rows": [{"source": row["source"], "old_bytes": row["v13_messages_bytes"],
                                    "new_bytes": row.get("dense_messages_bytes"),
                                    "old_service_prompt_tokens": row["historical_service_prompt_tokens"]} for row in rows]}
    (output / "full-request-bytes.json").write_text(json.dumps(message_comparison, ensure_ascii=False, indent=2) + "\n")
    return {key: value for key, value in summary.items() if key != "rows"}


def restore_projection_as_problem(compact):
    """Invert units of the already-rounded historical view for offline analysis."""
    rounds = []
    for row in compact["rounds"]:
        constraints = []
        for i, (name, capacity) in enumerate(compact["constraints"]):
            if row["loads"][i] is not None:
                constraints.append({"id": str(i), "name": name,
                                    "kind": "resource" if isinstance(name, str) else "edge",
                                    "capacity_Bps": capacity * 1e9,
                                    "load_bytes": row["loads"][i] * capacity * 1e3})
        rounds.append({"round": row["round"], "time_us": row["time"], "path_max_us": row["path"],
                       "overhead_us": row["overhead"], "fixed_floor_us": row["floor"], "constraints": constraints})
    groups = [{"id": row["id"], "round": row["round"], "max_chunks": row["max_chunks"],
               "effects": [[str(i), delta * 1e3 * compact["constraints"][i][1] / row["max_chunks"]]
                           for i, delta in enumerate(row["delta"]) if delta],
               "path_sample": {"old": row["paths"][0], "new": row["paths"][1]},
               "full_move_path_max_us": row["bounds"][0],
               "full_move_added_edge_busy_upper_us": row["bounds"][1],
               "affected_edge_min_capacity_Bps": row["bounds"][2] * 1e9}
              for row in compact["groups"]]
    result = {"base": compact["base"], "chunk_bytes": compact["chunk_bytes"], "rounds": rounds, "groups": groups}
    if "topology" in compact:
        result["topology"] = compact["topology"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=Path("artifacts/v1.3"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v1.4/analysis/prompt"))
    args = parser.parse_args()
    print(json.dumps(compare(args.historical_root, args.output_dir), indent=2))
