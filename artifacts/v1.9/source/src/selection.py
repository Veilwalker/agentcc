"""Compare validated schedules without forcing model actions into the winner.

Whole candidates may use different collective programs. Round-wise composition
is safe only where their complete atomic actions agree, independently of paths
and packet grouping. Every composed schedule is replayed before it can win.
"""

from collections import Counter
import math
import time

from .replay import replay
from .route_balance import _PathCosts, _RoundLoad


def round_actions(round_):
    """Count original chunks or entire reduction expressions, never contributors."""
    return Counter((transfer["action"], transfer["src"], transfer["dst"], chunk)
                   for transfer in round_["transfers"]
                   for chunk in transfer["chunk_ranges"])


def round_times(solution, report):
    values = iter(report["round_times_us"])
    return [next(values) if row["transfers"] else 0.0 for row in solution["rounds"]]


def improves(new, old):
    return new < old - max(1e-6, old * 1e-10)


def _same_layout(left, right):
    return (left["split"] == right["split"]
            and left["initial_possession"] == right["initial_possession"]
            and left["final_chunks"] == right["final_chunks"]
            and len(left["rounds"]) == len(right["rounds"]))


def _atomic_paths(round_):
    """Return unique complete actions and their paths/positions, or reject ambiguity.

    Counter equality is checked separately, so repeated actions are never
    silently deduplicated. A duplicate action cannot be assigned an occurrence
    identity from schedule data alone; migration conservatively skips its round.
    """
    atoms = {}
    for ti, transfer in enumerate(round_["transfers"]):
        path = tuple(transfer.get("path", transfer.get("route")))
        for ci, token in enumerate(transfer["chunk_ranges"]):
            key = transfer["action"], transfer["src"], transfer["dst"], token
            if key in atoms:
                return None
            atoms[key] = path, ti, ci
    return atoms


def _model_changes(baseline_round, edited_round, control_round):
    """Identify exactly the model's path changes; never infer new chunk counts."""
    actions = round_actions(edited_round)
    if round_actions(baseline_round) != actions or round_actions(control_round) != actions:
        return None
    before, after, control = map(_atomic_paths, (baseline_round, edited_round, control_round))
    if before is None or after is None or control is None:
        return None
    return {key: (before[key][0], value[0]) for key, value in after.items()
            if before[key][0] != value[0]}


def _overlay_round(control_round, changes):
    """Apply every changed atom in a model round, preserving other control paths.

    Partition an existing control packet only when its atoms need different
    paths. Never merge packets or split same-path traffic just to improve its
    path-time term. A target already used by the control is a genuine no-op.
    """
    transfers, changed = [], 0
    for transfer in control_round["transfers"]:
        original_path = tuple(transfer.get("path", transfer.get("route")))
        partitions = {}
        for token in transfer["chunk_ranges"]:
            key = transfer["action"], transfer["src"], transfer["dst"], token
            target = changes[key][1] if key in changes else original_path
            changed += target != original_path
            partitions.setdefault(target, []).append(token)
        if len(partitions) == 1 and original_path in partitions:
            transfers.append(transfer)
        else:
            transfers.extend({**transfer, "path": list(path), "chunk_ranges": tokens}
                             for path, tokens in partitions.items())
    return ({**control_round, "transfers": transfers} if changed else control_round), changed


def _atomic_evidence(baseline, edited, control, chosen, selected, overlay_rounds):
    """Rebuild final positions; seed transfer indices are not final positions.

    Manifest rows group atoms sharing a final packet and identical provenance.
    selected_transfers may also contain non-model atoms; chunk coordinates and
    manifest chunks define the exact observed model subset, including no-ops
    where the deterministic control already used the requested target path.
    """
    rows, positions = {}, []
    changed_count = overlay_changed = 0
    for number in selected:
        ri = number - 1
        changes = _model_changes(baseline["rounds"][ri], edited["rounds"][ri], control["rounds"][ri])
        final = _atomic_paths(chosen["rounds"][ri])
        previous = _atomic_paths(control["rounds"][ri])
        if changes is None or final is None or previous is None:
            return None
        round_changed = sum(value[0] != previous[key][0] for key, value in final.items())
        changed_count += round_changed
        if number in overlay_rounds:
            overlay_changed += round_changed
        for key, (old_path, target) in changes.items():
            path, ti, ci = final[key]
            if path != target:
                return None
            action, src, dst, token = key
            control_path = previous[key][0]
            identity = ri, ti, old_path, target, control_path
            if identity not in rows:
                rows[identity] = {"round": number, "transfer_index": ti,
                                  "action": action, "src": src, "dst": dst,
                                  "model_before_path": list(old_path), "model_path": list(target),
                                  "control_before_path": list(control_path),
                                  "chunks": [], "chunk_indices": []}
            rows[identity]["chunks"].append(token)
            rows[identity]["chunk_indices"].append(ci)
            positions.append([ri, ti, ci])
    return {"model_atomic_manifest": list(rows.values()), "model_atomic_count": len(positions),
            "control_changed_atoms": changed_count, "overlay_control_changed_atoms": overlay_changed,
            "selected_atomic_positions": sorted(positions),
            "selected_transfers": [list(pair) for pair in sorted({tuple(row[:2]) for row in positions})],
            "atomic_overlay_rounds": sorted(overlay_rounds)}


def choose_model_candidate(task, graph, control, control_report, edited, edited_report,
                           edited_rounds, *, baseline=None, overlay_min_relative_gain=0.0):
    """Compare whole candidates, compatible rounds, then exact model atom overlays.

    ``baseline`` is the unchanged internal seed already hash-bound by the edit
    compiler; this selector is not an external trust boundary. Cost estimates
    only prune. Every new potential winner must pass full replay, and overlay
    failure retains the actual validated winner of the original selection.
    """
    if (type(overlay_min_relative_gain) not in (int, float)
            or not math.isfinite(overlay_min_relative_gain) or overlay_min_relative_gain < 0):
        raise ValueError("overlay_min_relative_gain must be finite and nonnegative")
    evidence = {"selected": False, "selected_rounds": [], "combination": "control",
                "control_time_us": control_report["total_time_us"], "errors": [],
                "without_overlay_time_us": control_report["total_time_us"],
                "overlay_elapsed_s": 0.0, "overlay_rounds_considered": 0,
                "overlay_rejections": {}, "overlay_min_relative_gain": overlay_min_relative_gain,
                "overlay_predicted_time_us": None, "overlay_predicted_gain_us": None,
                "overlay_predicted_relative_gain": None}
    overlay_started = None

    def finish():
        if overlay_started is not None:
            evidence["overlay_elapsed_s"] = round(time.monotonic() - overlay_started, 6)
        return chosen, checked, evidence

    def skip(reason):
        counts = evidence["overlay_rejections"]
        counts[reason] = counts.get(reason, 0) + 1

    chosen, checked = control, control_report
    if edited_report is None:
        skip("no_model_candidate")
        return finish()
    if improves(edited_report["total_time_us"], checked["total_time_us"]):
        chosen, checked = edited, edited_report
        evidence.update(selected=True, selected_rounds=sorted(edited_rounds), combination="whole_model")
    if not _same_layout(control, edited):
        evidence["without_overlay_time_us"] = checked["total_time_us"]
        skip("incompatible_layout")
        return finish()

    # Establish the actual pre-overlay winner, including a compatible-round
    # combination. If the new overlay fails, this validated result survives.
    control_times, model_times = round_times(control, control_report), round_times(edited, edited_report)
    rounds, predicted = list(control["rounds"]), list(control_times)
    selected, compatible = [], []
    for number in edited_rounds:
        index = number - 1
        if round_actions(edited["rounds"][index]) != round_actions(control["rounds"][index]):
            continue
        compatible.append(number)
        if improves(model_times[index], control_times[index]):
            rounds[index], predicted[index] = edited["rounds"][index], model_times[index]
            selected.append(number)
    if selected and improves(sum(predicted), checked["total_time_us"]):
        candidate = {**control, "rounds": list(rounds)}
        combined = replay(task, graph, candidate)
        if not combined["valid"]:
            evidence["errors"].append("compatible-round candidate failed final replay")
        elif improves(combined["total_time_us"], checked["total_time_us"]):
            candidate["metrics"] = {"total_time": combined["total_time_us"] / 1e6,
                                    "round_times": [value / 1e6 for value in combined["round_times_us"]]}
            chosen, checked = candidate, combined
            evidence.update(selected=True, selected_rounds=sorted(selected), combination="compatible_rounds")
    evidence["without_overlay_time_us"] = checked["total_time_us"]
    if baseline is None:
        skip("no_baseline")
        return finish()
    overlay_started = time.monotonic()
    if not _same_layout(baseline, edited):
        skip("incompatible_baseline_layout")
        return finish()

    costs = _PathCosts(graph)
    chunk_bytes = task["data_size"]["per_rank_bytes"] // control["split"]
    overlay_rounds = set()
    for number in edited_rounds:
        evidence["overlay_rounds_considered"] += 1
        if number not in compatible:
            skip("incompatible_round_actions")
            continue
        index = number - 1
        changes = _model_changes(baseline["rounds"][index], edited["rounds"][index],
                                 control["rounds"][index])
        if changes is None:
            skip("incompatible_or_ambiguous_atoms")
            continue
        if not changes:
            skip("no_model_path_changes")
            continue
        migrated, changed = _overlay_round(control["rounds"][index], changes)
        if not changed:
            skip("no_changed_atoms")
            continue
        # Existing atoms retain their expression boundaries. Since packets are
        # only partitioned, AR slots cannot be duplicated within a new packet.
        try:
            load = _RoundLoad(costs)
            for transfer in migrated["transfers"]:
                load.add(transfer.get("path", transfer.get("route")),
                         len(transfer["chunk_ranges"]) * chunk_bytes)
        except (KeyError, ValueError, TypeError):
            skip("invalid_path_costs")
            continue
        if not improves(load.time_us, predicted[index]):
            skip("not_better_round")
            continue
        rounds[index], predicted[index] = migrated, load.time_us
        overlay_rounds.add(number)
        if number not in selected:
            selected.append(number)
    predicted_total = sum(predicted)
    if overlay_rounds:
        gain = checked["total_time_us"] - predicted_total
        ratio = gain / checked["total_time_us"] if checked["total_time_us"] else 0.0
        evidence.update(overlay_predicted_time_us=predicted_total,
                        overlay_predicted_gain_us=gain, overlay_predicted_relative_gain=ratio)
    if not overlay_rounds or not improves(predicted_total, checked["total_time_us"]):
        if overlay_rounds:
            skip("not_better_than_existing")
        return finish()
    # This optional resource policy only prunes the new overlay attempt. The
    # independently validated whole/compatible winner remains untouched. It
    # does not estimate official scoring weights or replace final replay.
    if ratio < overlay_min_relative_gain:
        skip("below_min_relative_gain")
        return finish()

    candidate = {**control, "rounds": rounds}
    combined = replay(task, graph, candidate)
    if not combined["valid"]:
        evidence["errors"].append("atomic-overlay candidate failed final replay")
        skip("replay_failed")
        return finish()
    if not improves(combined["total_time_us"], checked["total_time_us"]):
        skip("replay_not_better")
        return finish()
    atomic = _atomic_evidence(baseline, edited, control, candidate, sorted(selected), overlay_rounds)
    if atomic is None:
        evidence["errors"].append("atomic overlay evidence does not match the returned candidate")
        skip("evidence_mismatch")
        return finish()
    candidate["metrics"] = {"total_time": combined["total_time_us"] / 1e6,
                            "round_times": [value / 1e6 for value in combined["round_times_us"]]}
    chosen, checked = candidate, combined
    evidence.update(selected=True, selected_rounds=sorted(selected), combination="atomic_overlay", **atomic)
    return finish()
