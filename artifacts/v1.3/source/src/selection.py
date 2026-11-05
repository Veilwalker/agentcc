"""Compare validated schedules without forcing model actions into the winner.

Whole candidates may use different collective programs. Round-wise composition
is safe only where their complete atomic actions agree, independently of paths
and packet grouping. Every composed schedule is replayed before it can win.
"""

from collections import Counter

from .replay import replay


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


def choose_model_candidate(task, graph, control, control_report, edited, edited_report,
                           edited_rounds):
    """Return the best validated whole or compatible-round candidate.

    The deterministic control has already completed unrestricted route search.
    No path is frozen to preserve attribution. A model candidate must beat that
    control, or contribute a strictly faster compatible round, to be selected.
    """
    evidence = {"selected": False, "selected_rounds": [], "combination": "control",
                "control_time_us": control_report["total_time_us"], "errors": []}
    if edited_report is None:
        return control, control_report, evidence
    chosen, checked = control, control_report
    if improves(edited_report["total_time_us"], checked["total_time_us"]):
        chosen, checked = edited, edited_report
        evidence.update(selected=True, selected_rounds=sorted(edited_rounds), combination="whole_model")
    same_layout = (control["split"] == edited["split"]
                   and control["initial_possession"] == edited["initial_possession"]
                   and control["final_chunks"] == edited["final_chunks"]
                   and len(control["rounds"]) == len(edited["rounds"]))
    if not same_layout:
        return chosen, checked, evidence
    control_times, model_times = round_times(control, control_report), round_times(edited, edited_report)
    rounds = list(control["rounds"])
    selected = []
    for number in edited_rounds:
        index = number - 1
        if (improves(model_times[index], control_times[index])
                and round_actions(edited["rounds"][index]) == round_actions(control["rounds"][index])):
            rounds[index] = edited["rounds"][index]
            selected.append(number)
    if not selected:
        return chosen, checked, evidence
    candidate = {**control, "rounds": rounds}
    combined = replay(task, graph, candidate)
    if not combined["valid"]:
        evidence["errors"].append("compatible-round candidate failed final replay")
        return chosen, checked, evidence
    if improves(combined["total_time_us"], checked["total_time_us"]):
        candidate["metrics"] = {"total_time": combined["total_time_us"] / 1e6,
                                "round_times": [value / 1e6 for value in combined["round_times_us"]]}
        chosen, checked = candidate, combined
        evidence.update(selected=True, selected_rounds=selected, combination="compatible_rounds")
    return chosen, checked, evidence
