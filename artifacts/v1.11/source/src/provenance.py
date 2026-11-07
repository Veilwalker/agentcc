"""Observe original model path targets after unrestricted candidate selection.

The edit compiler binds the input and baseline before this internal hand-off.
This module does not validate schedules or decide which one wins. It reads the
compiler's accepted prefixes, never differences introduced by postprocessing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelAtom:
    round_index: int
    action: str
    src: str
    dst: str
    token: str
    before_path: tuple[str, ...]
    target_path: tuple[str, ...]
    group_id: str | None

    @property
    def key(self):
        return self.round_index, self.action, self.src, self.dst, self.token


@dataclass(frozen=True)
class ModelLedger:
    """Private immutable intent; do not serialize the complete ledger."""

    available: bool
    reason: str | None = None
    layout: tuple | None = None
    atoms: tuple[ModelAtom, ...] = ()
    requested_edits: tuple[tuple[str, int], ...] = ()
    applied_edits: tuple[tuple[str, int], ...] = ()
    group_counts_available: bool = False
    ignored_noop_atomic_count: int = 0
    noop_group_counts: tuple[tuple[str, int], ...] = ()


def _layout(solution):
    """Copy only ownership/layout fields, not all round atomic actions."""
    if type(solution["split"]) is not int:
        raise ValueError("invalid split")
    return (solution["operator"], solution["split"], solution["round_count"],
            tuple(sorted((rank, tuple(tokens)) for rank, tokens in solution["initial_possession"].items())),
            tuple(solution["chunks"]), tuple(solution["final_chunks"]),
            tuple(row["round"] for row in solution["rounds"]))


def _edit_counts(rows):
    if not isinstance(rows, list):
        raise ValueError("edit counts must be an array")
    result, seen = [], set()
    for row in rows:
        if (not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
                or row[0] in seen or type(row[1]) is not int or row[1] < 0):
            raise ValueError("edit counts must contain unique IDs and nonnegative integers")
        result.append((row[0], row[1]))
        seen.add(row[0])
    return tuple(result)


def build_model_ledger(baseline, edit_evidence, *, context=None):
    """Capture accepted original prefixes without rescanning entire programs.

    Call immediately after ``apply_proposal`` on the same unchanged baseline.
    The optional compiler context supplies exact per-group membership and cheap
    existing binding checks. No new whole-baseline hash is computed here.
    Bad or unavailable provenance returns ``available=False``; it must never
    discard an independently validated faster solution.
    """
    requested, applied = (), ()
    try:
        if not isinstance(edit_evidence, dict):
            return ModelLedger(False, "no_accepted_model_edits")
        if not edit_evidence.get("accepted"):
            proposal = edit_evidence.get("requested_edits")
            if isinstance(proposal, dict) and isinstance(proposal.get("edits"), list):
                requested = _edit_counts(proposal["edits"])
            return ModelLedger(False, "no_accepted_model_edits", requested_edits=requested)
        proposal = edit_evidence["requested_edits"]
        if proposal["base"] != edit_evidence["base"]:
            raise ValueError("proposal and accepted evidence bindings differ")
        requested = _edit_counts(proposal["edits"])
        applied = _edit_counts(edit_evidence["applied_edits"])
        requested_map = dict(requested)
        if not applied or any(count <= 0 or requested_map.get(name) != count for name, count in applied):
            raise ValueError("applied counts differ from original requested counts")
        members = {}
        if context is not None:
            for key in ("base", "baseline_sha256", "input_sha256"):
                if getattr(context, key) != edit_evidence[key]:
                    raise ValueError("context and accepted evidence bindings differ")
            for group in context.groups:
                if group.id not in dict(applied):
                    continue
                for member in group.members:
                    key = group.round_index, member.transfer_index
                    if key in members:
                        raise ValueError("ambiguous group member")
                    members[key] = group.id, member.chunk_count, tuple(member.path)
        accepted_rounds = edit_evidence["accepted_rounds"]
        if (not isinstance(accepted_rounds, list) or any(type(number) is not int for number in accepted_rounds)
                or len(set(accepted_rounds)) != len(accepted_rounds)):
            raise ValueError("invalid accepted round list")
        accepted_rounds = set(accepted_rounds)
        atoms, seen, prefixes, per_group = [], set(), set(), defaultdict(int)
        noop_groups = defaultdict(int)
        counted = noops = 0
        for entry in edit_evidence["edit_manifest"]:
            if not isinstance(entry, list) or len(entry) != 5:
                raise ValueError("invalid accepted prefix manifest")
            number, ti, count, before, target = entry
            if (type(number) is not int or number not in accepted_rounds or number < 1
                    or type(ti) is not int or ti < 0 or type(count) is not int or count <= 0):
                raise ValueError("invalid accepted prefix coordinates")
            ri = number - 1
            if (ri, ti) in prefixes:
                raise ValueError("duplicate accepted prefix")
            prefixes.add((ri, ti))
            transfer = baseline["rounds"][ri]["transfers"][ti]
            before, target = tuple(before), tuple(target)
            if (before != tuple(transfer.get("path", transfer.get("route")))
                    or count > len(transfer["chunk_ranges"]) or len(target) < 2
                    or any(not isinstance(node, str) for node in target)
                    or target[0] != transfer["src"] or target[-1] != transfer["dst"]):
                raise ValueError("accepted prefix does not match its bound baseline")
            group_id = None
            if context is not None:
                group_id, member_count, member_path = members[ri, ti]
                if count > member_count or target != member_path:
                    raise ValueError("accepted prefix differs from its compiler member")
                per_group[group_id] += count
            counted += count
            for token in transfer["chunk_ranges"][:count]:
                key = ri, transfer["action"], transfer["src"], transfer["dst"], token
                if key in seen:
                    raise ValueError("ambiguous duplicate accepted model atom")
                seen.add(key)
                if before == target:
                    noops += 1
                    if group_id is not None:
                        noop_groups[group_id] += 1
                    continue
                atoms.append(ModelAtom(*key, before, target, group_id))
        if counted != sum(count for _, count in applied):
            raise ValueError("manifest atomic count differs from original applied counts")
        if context is not None and dict(per_group) != dict(applied):
            raise ValueError("manifest group counts differ from original applied counts")
        return ModelLedger(True, layout=_layout(baseline), atoms=tuple(atoms),
                           requested_edits=requested, applied_edits=applied,
                           group_counts_available=context is not None,
                           ignored_noop_atomic_count=noops, noop_group_counts=tuple(noop_groups.items()))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        return ModelLedger(False, "invalid_model_provenance: " + str(exc),
                           requested_edits=requested, applied_edits=applied)


def _find_atoms(solution, atoms):
    """Visit only intent rounds/pairs; retain at most one occurrence per atom."""
    wanted = defaultdict(lambda: defaultdict(set))
    for atom in atoms:
        wanted[atom.round_index][atom.action, atom.src, atom.dst].add(atom.token)
    found = {}
    for ri, pairs in wanted.items():
        for ti, transfer in enumerate(solution["rounds"][ri]["transfers"]):
            pair = transfer["action"], transfer["src"], transfer["dst"]
            tokens = pairs.get(pair)
            if tokens is None:
                continue
            path = tuple(transfer.get("path", transfer.get("route")))
            for ci, token in enumerate(transfer["chunk_ranges"]):
                if token not in tokens:
                    continue
                key = ri, *pair, token
                found[key] = None if key in found else (path, ti, ci)
    return found


def observe_model_ledger(ledger, final, *, control=None):
    """Describe final target survival and its difference from the actual control.

    ``new_target_atoms_vs_control`` is an observed path difference, not a causal
    model-performance claim. Caller records branch lineage separately. Layout
    mismatch/ambiguous occurrence is conservative and never changes a winner.
    """
    result = {"available": False, "reason": ledger.reason,
              "scope": "original_accepted_model_paths", "preservation_scope": "observed_not_enforced",
              "original_requested_edits": [list(row) for row in ledger.requested_edits],
              "original_applied_edits": [list(row) for row in ledger.applied_edits],
              "requested_atomic_count": sum(count for _, count in ledger.requested_edits),
              "applied_atomic_count": sum(count for _, count in ledger.applied_edits),
              "accepted_atomic_count": len(ledger.atoms),
              "ignored_noop_atomic_count": ledger.ignored_noop_atomic_count,
              "preserved_atomic_count": None, "overwritten_atomic_count": None,
              "absent_atomic_count": None, "ambiguous_atomic_count": None,
              "all_targets_preserved": False, "control_available": False,
              "control_unavailable_reason": "not_observed",
              "new_target_atoms_vs_control": None, "already_control_target_atomic_count": None,
              "control_unknown_atomic_count": None, "group_counts_available": ledger.group_counts_available,
              "group_outcomes": None, "model_atomic_manifest": [], "model_atomic_count": 0,
              "selected_atomic_positions": [], "selected_transfers": [],
              "fully_preserved_edits": [], "partially_preserved_edits": []}
    if not ledger.available:
        return result
    try:
        if _layout(final) != ledger.layout:
            result["reason"] = "incompatible_final_layout"
            return result
        found = _find_atoms(final, ledger.atoms)
        control_found = {}
        try:
            control_ok = control is not None and _layout(control) == ledger.layout
            control_reason = None if control_ok else ("not_supplied" if control is None else "incompatible_layout")
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            control_ok, control_reason = False, "malformed_control"
        if control_ok:
            control_found = (found if control is final or control["rounds"] is final["rounds"]
                             else _find_atoms(control, ledger.atoms))
        groups = {name: {"group_id": name, "original_requested_count": count,
                        "original_applied_count": dict(ledger.applied_edits).get(name, 0),
                        "preserved_count": 0, "overwritten_count": 0, "absent_count": 0,
                        "ambiguous_count": 0, "new_targets_vs_control": 0,
                        "ignored_noop_count": dict(ledger.noop_group_counts).get(name, 0),
                        "already_control_target_count": 0, "control_unknown_count": 0}
                  for name, count in ledger.requested_edits} if ledger.group_counts_available else {}
        preserved = overwritten = absent = ambiguous = added = already = unknown = 0
        positions, manifests = [], {}
        for atom in ledger.atoms:
            group = groups.get(atom.group_id)
            observed = found.get(atom.key)
            if atom.key not in found:
                absent += 1
                if group is not None:
                    group["absent_count"] += 1
                continue
            if observed is None:
                ambiguous += 1
                if group is not None:
                    group["ambiguous_count"] += 1
                continue
            path, ti, ci = observed
            if path != atom.target_path:
                overwritten += 1
                if group is not None:
                    group["overwritten_count"] += 1
                continue
            preserved += 1
            if group is not None:
                group["preserved_count"] += 1
            prior = control_found.get(atom.key) if control_ok else None
            if prior is None:
                relation = "unknown"
                unknown += 1
                if group is not None:
                    group["control_unknown_count"] += 1
            elif prior[0] == atom.target_path:
                relation = "already_in_control"
                already += 1
                if group is not None:
                    group["already_control_target_count"] += 1
            else:
                relation = "different_from_control"
                added += 1
                if group is not None:
                    group["new_targets_vs_control"] += 1
            positions.append([atom.round_index, ti, ci])
            control_path = prior[0] if prior is not None else None
            identity = atom.round_index, ti, atom.group_id, atom.before_path, atom.target_path, relation, control_path
            if identity not in manifests:
                manifests[identity] = {"round": atom.round_index + 1, "transfer_index": ti,
                    "action": atom.action, "src": atom.src, "dst": atom.dst, "group_id": atom.group_id,
                    "model_before_path": list(atom.before_path), "model_path": list(atom.target_path),
                    "control_before_path": list(control_path) if control_path is not None else None,
                    "control_relation": relation, "chunks": [], "chunk_indices": []}
            manifests[identity]["chunks"].append(atom.token)
            manifests[identity]["chunk_indices"].append(ci)
        fully = [[row["group_id"], row["original_applied_count"]] for row in groups.values()
                 if row["original_applied_count"] > 0 and row["preserved_count"] == row["original_applied_count"]]
        partially = [[row["group_id"], row["preserved_count"]] for row in groups.values()
                     if 0 < row["preserved_count"] < row["original_applied_count"]]
        result.update(available=True, reason=None, preserved_atomic_count=preserved,
                      overwritten_atomic_count=overwritten, absent_atomic_count=absent,
                      ambiguous_atomic_count=ambiguous,
                      all_targets_preserved=bool(ledger.atoms) and preserved == len(ledger.atoms),
                      control_available=control_ok, control_unavailable_reason=control_reason,
                      new_target_atoms_vs_control=added if control_ok else None,
                      already_control_target_atomic_count=already if control_ok else None,
                      control_unknown_atomic_count=unknown,
                      group_outcomes=list(groups.values()) if ledger.group_counts_available else None,
                      model_atomic_manifest=list(manifests.values()), model_atomic_count=preserved,
                      selected_atomic_positions=sorted(positions),
                      selected_transfers=[list(pair) for pair in sorted({tuple(row[:2]) for row in positions})],
                      fully_preserved_edits=fully, partially_preserved_edits=partially)
        return result
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        result["reason"] = "incompatible_or_malformed_observation"
        return result
