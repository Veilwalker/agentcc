"""Bounded capacity proposals followed by one independent complete Replay.

The original validated schedule always remains a candidate. Predicted timings
screen proposals; they never masquerade as Replay results. No model action is
frozen, and model provenance must be observed on the chosen final schedule.
"""

from __future__ import annotations

import time

from .packetization import repacketize, MAX_OUTPUT_TRANSFERS
from .replay import certify as replay
from .route_joint import propose_joint
from .selection import improves


def polish(task, graph, solution, validated_report, *, enabled=True):
    """Return a valid nonworse candidate, its actual report, and phase evidence.

    ``validated_report`` is a trusted internal result for this exact solution.
    Route and packet proposals preserve complete atoms. Changed candidates are
    checked together once; an unchanged candidate reuses its genuine report.
    """
    started = time.monotonic()
    evidence = {"accepted": False, "before_us": validated_report["total_time_us"],
                "after_us": validated_report["total_time_us"], "replay_calls": 0,
                "joint_s": 0.0, "packet_s": 0.0, "replay_s": 0.0}
    if not enabled:
        return solution, validated_report, {**evidence, "reason": "disabled", "elapsed_s": 0.0}
    try:
        original_count = sum(len(row["transfers"]) for row in solution["rounds"])
        growth_budget = min(2048, max(0, MAX_OUTPUT_TRANSFERS - original_count))
        if growth_budget == 0:
            evidence["reason"] = "output_size_limit"
            return solution, validated_report, evidence
        # Try the cheap same-path capacity correction first. Once it already
        # yields a material gain, avoid paying for a second routing search.
        phase = time.monotonic()
        candidate, packets = repacketize(
            task, graph, solution, validated_report,
            max_extra_packets=growth_budget, min_relative_gain=0.002)
        evidence.update(packetization=packets, packet_s=time.monotonic() - phase)
        if packets.get("predicted_relative_gain", 0) >= 0.01:
            evidence["joint"] = {"proposed": False, "reason": "cheap_capacity_gain"}
        else:
            phase = time.monotonic()
            routed, joint = propose_joint(
                task, graph, solution, validated_report, max_sweeps=1,
                max_extra_packets=min(1024, growth_budget), min_relative_gain=0.002)
            evidence.update(joint=joint, joint_s=time.monotonic() - phase)
            if routed is not solution:
                extra = sum(len(row["transfers"]) for row in routed["rounds"]) - original_count
                phase = time.monotonic()
                routed, routed_packets = repacketize(
                    task, graph, routed, None,
                    max_extra_packets=max(0, growth_budget - max(0, extra)),
                    min_relative_gain=0.002)
                evidence["packet_s"] += time.monotonic() - phase
                evidence["routed_packetization"] = routed_packets
                if routed_packets.get("predicted_after_us", joint.get("candidate_time_us", float("inf"))) < packets.get("predicted_after_us", validated_report["total_time_us"]):
                    candidate = routed
        if candidate is solution:
            evidence["reason"] = "no_profitable_proposal"
            return solution, validated_report, evidence
        phase = time.monotonic()
        evidence["replay_calls"] = 1
        checked = replay(task, graph, candidate)
        evidence["replay_s"] = time.monotonic() - phase
        if not checked["valid"] or not improves(checked["total_time_us"], validated_report["total_time_us"]):
            evidence.update(reason="invalid_or_not_better", validation_errors=checked["errors"])
            return solution, validated_report, evidence
        candidate = {**candidate, "metrics": {**candidate.get("metrics", {}),
                     "total_time": checked["total_time_us"] / 1e6,
                     "round_times": [value / 1e6 for value in checked["round_times_us"]]}}
        evidence.update(accepted=True, reason="validated_improvement", after_us=checked["total_time_us"])
        return candidate, checked, evidence
    except (ValueError, KeyError, IndexError, TypeError, ArithmeticError) as exc:
        # Optional optimization failure cannot discard a valid solution.
        evidence.update(reason="proposal_failed", error_type=type(exc).__name__)
        return solution, validated_report, evidence
    finally:
        evidence["elapsed_s"] = time.monotonic() - started


def compare_with_control(task, graph, baseline, raw_control, raw_report, control,
                         control_report, edited, edited_report, accepted_rounds,
                         *, enabled=True):
    """Keep the old winner and compare model edits against the stronger control.

    The original selector is also evaluated against the unchanged control. Its
    validated winner protects against route interactions in atomic overlays.
    Only the competitive non-control winner receives another bounded polish;
    losing raw model branches are not all exhaustively optimized. This is a
    finite candidate search, not a fair-budget causal ablation of model value.
    """
    from .selection import choose_model_candidate

    def select(base, report):
        return choose_model_candidate(
            task, graph, base, report, edited, edited_report, accepted_rounds,
            baseline=baseline, overlay_min_relative_gain=0.001)

    old, old_report, old_selection = select(raw_control, raw_report)
    if control is raw_control:
        chosen, checked, selection = old, old_report, old_selection
    else:
        chosen, checked, selection = select(control, control_report)
        if improves(old_report["total_time_us"], checked["total_time_us"]):
            chosen, checked, selection = old, old_report, old_selection
    details = {"legacy_winner_us": old_report["total_time_us"],
               "strong_control_us": control_report["total_time_us"],
               "competitive_polish": None}
    if chosen is not control:
        chosen, checked, details["competitive_polish"] = polish(
            task, graph, chosen, checked, enabled=enabled)
    prior_scope = {key: value for key, value in selection.items() if key not in {
        "model_atomic_manifest", "selected_atomic_positions", "selected_transfers"}}
    selection = {**selection, "pre_capacity_selection": prior_scope,
                 "control_time_us": control_report["total_time_us"],
                 "capacity_comparison": details}
    return chosen, checked, selection
