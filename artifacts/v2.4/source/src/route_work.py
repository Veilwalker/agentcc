"""Deterministic work limits for optional route refinement.

Work is consumed in complete transfer batches. A shortest-path search reports
its attempted edge relaxations once per expanded node; it may exceed its limit
by that node's out-degree. The wall bound is an exceptional safety stop, never
the normal criterion for deciding which transfer receives optimization.
"""

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class RouteWorkBudget:
    path_scores: int
    edge_relaxations: int
    safety_s: float = 2.0

    def __post_init__(self):
        for value in (self.path_scores, self.edge_relaxations):
            if type(value) is not int or value < 0:
                raise ValueError("route work limits must be nonnegative integers")
        if (type(self.safety_s) not in (int, float) or not math.isfinite(self.safety_s)
                or self.safety_s <= 0):
            raise ValueError("route safety time must be positive and finite")


class RouteWorkMeter:
    def __init__(self, budget, started):
        self.budget = budget
        self.deadline = started + budget.safety_s
        self.path_scores = 0
        self.edge_relaxations = 0
        self.completed_transfers = 0
        self.stop_reason = None

    def can_start_transfer(self):
        if self.stop_reason:
            return False
        if self.path_scores >= self.budget.path_scores:
            self.stop_reason = "path_score_limit"
        elif self.edge_relaxations >= self.budget.edge_relaxations:
            self.stop_reason = "edge_relaxation_limit"
        elif time.monotonic() >= self.deadline:
            self.stop_reason = "wall_safety"
        return self.stop_reason is None

    def consume_relaxations(self, count):
        self.edge_relaxations += count
        # Finish the current transfer using every already-known legal route,
        # even when its optional new path runs out of construction work.
        return self.edge_relaxations < self.budget.edge_relaxations

    def finish_transfer(self, path_count):
        self.path_scores += path_count
        self.completed_transfers += 1

    def evidence(self):
        return {"path_scores": self.path_scores, "edge_relaxations": self.edge_relaxations,
                "completed_transfers": self.completed_transfers,
                "stop_reason": self.stop_reason or "exhausted_candidates",
                "limits": {"path_scores": self.budget.path_scores,
                           "edge_relaxations": self.budget.edge_relaxations,
                           "safety_s": self.budget.safety_s}}
