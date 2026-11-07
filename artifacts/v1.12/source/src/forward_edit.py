"""Bounded forwarding rewrites, verified against complete symbolic Replay.

Unlike path-only edits, these proposals may change a copy's sender or advance
ready atoms one round. Reduction actions are immutable. Every copy destination
and atom is conserved, and all changes are committed as one validated proposal.
The module does not call a model or train/load a neural network.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
import time

from .replay import replay
from .route_balance import _PathCosts, _RoundLoad
from .schedule_edit import _input_digest


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _body(solution):
    return {key: value for key, value in solution.items() if key not in ("strategy", "metrics")}


@dataclass(frozen=True)
class Move:
    round_index: int
    transfer_index: int
    atoms: tuple[str, ...]
    source: str
    target_round: int
    path: tuple[str, ...]


@dataclass(frozen=True)
class Rewrite:
    kind: str
    moves: tuple[Move, ...]


@dataclass(frozen=True)
class Context:
    input_hash: str
    schedule_hash: str
    rewrites: tuple[Rewrite, ...]
    before_us: float


def _slot(token):
    match = re.fullmatch(r"c(\d+)_sum", token)
    return int(match[1]) if match else int(token.split("+")[0].rsplit("_c", 1)[1])


def copy_states(task, graph, solution):
    """Return round-start possession of AG atoms or complete AR slots only.

    Callers validate the base schedule first. AR masks are never interpreted as
    independently movable partial contributors; only complete copy actions are
    eligible for a rewrite.
    """
    ranks = graph.ranks
    index = {rank: i for i, rank in enumerate(ranks)}
    split = solution["split"]
    gather = task["operator"] == "all-gather"
    positions = {token: i for i, token in enumerate(solution["chunks"])}
    atom_bit = (lambda token: 1 << positions[token]) if gather else (lambda token: 1 << _slot(token))
    state = [sum(atom_bit(token) for token in solution["initial_possession"][rank])
             for rank in ranks] if gather else [0] * len(ranks)
    reduction = [[1 << i] * split for i in range(len(ranks))]
    full = (1 << len(ranks)) - 1
    expressions = {}
    snapshots = []
    for row in solution["rounds"]:
        snapshots.append(tuple(state))
        for transfer in row["transfers"]:
            dst = index[transfer["dst"]]
            if gather or transfer["action"] == "copy":
                for token in transfer["chunk_ranges"]:
                    state[dst] |= atom_bit(token)
            else:
                for token in transfer["chunk_ranges"]:
                    if token not in expressions:
                        mask = full if re.fullmatch(r"c\d+_sum", token) else sum(
                            1 << index[part.rsplit("_c", 1)[0]] for part in token.split("+"))
                        expressions[token] = (_slot(token), mask)
                    slot, mask = expressions[token]
                    reduction[dst][slot] |= mask
                    if reduction[dst][slot] == full:
                        state[dst] |= 1 << slot
    return snapshots, atom_bit


def compile_rewrites(solution, context, ids):
    """Compile only exact catalog actions; overlapping selections are rejected."""
    selected = defaultdict(list)
    used = set()
    for group_id in ids:
        for move in context.rewrites[group_id].moves:
            for token in move.atoms:
                key = move.round_index, move.transfer_index, token
                if key in used:
                    raise ValueError("selected rewrites overlap")
                used.add(key)
            selected[move.round_index, move.transfer_index].append(move)
    affected = {ri for key, moves in selected.items() for ri in (key[0], *(m.target_round for m in moves))}
    rounds = [({**row, "transfers": []} if ri in affected else row)
              for ri, row in enumerate(solution["rounds"])]
    for ri in sorted(affected):
        row = solution["rounds"][ri]
        for ti, transfer in enumerate(row["transfers"]):
            moves = selected.get((ri, ti), [])
            if not moves:
                rounds[ri]["transfers"].append(transfer)
                continue
            removed = {token for move in moves for token in move.atoms}
            remaining = [token for token in transfer["chunk_ranges"] if token not in removed]
            if remaining:
                rounds[ri]["transfers"].append({**transfer, "chunk_ranges": remaining})
            for move in moves:
                rewritten = {key: value for key, value in transfer.items() if key not in ("route", "route_id")}
                rewritten.update(src=move.source, path=list(move.path), chunk_ranges=list(move.atoms))
                rounds[move.target_round]["transfers"].append(rewritten)
    return {**solution, "rounds": rounds}


def _round_loads(graph, solution, chunk_bytes, costs):
    loads = []
    for row in solution["rounds"]:
        load = _RoundLoad(costs)
        for transfer in row["transfers"]:
            load.add(transfer["path"], len(transfer["chunk_ranges"]) * chunk_bytes)
        loads.append(load)
    return loads


def apply(task, graph, solution, context, proposal, *, compiled_here=False, validated_report=None):
    """Accept a strict improvement only after one full independent Replay.

    ``compiled_here`` is an internal fast path: the caller has just built the
    context from this exact solution and checked the input binding. External
    proposals still use the default strict binding check. A trusted control
    report can screen unchanged/losing costs; it never validates a winner.
    """
    started = time.monotonic()
    evidence = {"accepted": False, "selected": False, "before_us": context.before_us,
                "after_us": context.before_us, "requested": proposal, "errors": []}
    try:
        if not compiled_here and (context.input_hash != _input_digest(task, graph)
                                  or context.schedule_hash != _digest(_body(solution))):
            raise ValueError("forwarding proposal is bound to a different input or schedule")
        if not isinstance(proposal, dict) or set(proposal) != {"apply"}:
            raise ValueError("reply must contain only apply")
        ids = proposal["apply"]
        if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(context.rewrites) for i in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("apply must contain distinct valid integer group IDs")
        if not ids:
            evidence["reason"] = "no_edit"
            return solution, evidence
        candidate = compile_rewrites(solution, context, ids)
        old_count = sum(len(row["transfers"]) for row in solution["rounds"])
        new_count = sum(len(row["transfers"]) for row in candidate["rounds"])
        if new_count > max(old_count, min(65536, old_count + 2048)):
            raise ValueError("forwarding edit exceeds output size budget")
        if validated_report is not None:
            costs = _PathCosts(graph)
            chunk_bytes = task["data_size"]["per_rank_bytes"] // solution["split"]
            predicted = context.before_us
            nonempty_times = iter(validated_report["round_times_us"])
            base_times = [next(nonempty_times) if row["transfers"] else 0.0
                          for row in solution["rounds"]]
            changed = {ri for i in ids for m in context.rewrites[i].moves
                       for ri in (m.round_index, m.target_round)}
            for ri in sorted(changed):
                load = _RoundLoad(costs)
                for transfer in candidate["rounds"][ri]["transfers"]:
                    load.add(transfer["path"], len(transfer["chunk_ranges"]) * chunk_bytes)
                predicted += load.time_us - base_times[ri]
            evidence["screened_time_us"] = predicted
            if predicted >= context.before_us - max(1e-6, context.before_us * 1e-10):
                evidence.update(reason="no_communication_gain", validation="cost_screen_only")
                return solution, evidence
        report = replay(task, graph, candidate)
        evidence["validation"] = "complete_replay"
        if not report["valid"]:
            raise ValueError("rewritten schedule failed Replay: " + "; ".join(report["errors"]))
        evidence["proposal_us"] = report["total_time_us"]
        if report["total_time_us"] >= context.before_us - max(1e-6, context.before_us * 1e-10):
            evidence["reason"] = "no_communication_gain"
            return solution, evidence
        candidate["metrics"] = {**candidate.get("metrics", {}), "total_time": report["total_time_us"] / 1e6,
                                "round_times": [t / 1e6 for t in report["round_times_us"]]}
        evidence.update(accepted=True, selected=True, after_us=report["total_time_us"], applied=ids,
                        sender_changes=sum(m.source != solution["rounds"][m.round_index]["transfers"][m.transfer_index]["src"]
                                           for i in ids for m in context.rewrites[i].moves),
                        advanced_atoms=sum(len(m.atoms) for i in ids for m in context.rewrites[i].moves if m.target_round != m.round_index),
                        _validation_report=report)
        return candidate, evidence
    except (ValueError, TypeError, KeyError) as error:
        evidence["errors"].append(str(error))
        evidence["reason"] = "rejected"
        return solution, evidence
    finally:
        evidence["elapsed_s"] = time.monotonic() - started
