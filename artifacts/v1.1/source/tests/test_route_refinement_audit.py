"""Service-level evidence when an optional route refinement wins or fails."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src import service
from src.collectives import build
from src.replay import replay
from src.route_balance import refine
from src.solver import solve_schedule
from src.topology import Topology
from tests.test_solver import make_case


def congested_fixture(operator):
    task, topology = make_case(8, operator)
    topology["switches"] = ["fast", "alternate"]
    topology["edges"] = [
        {"src": rank, "dst": switch, "bw": capacity, "bidirectional": True}
        for rank in task["ranks"]
        for switch, capacity in [("fast", "10GB/s"), ("alternate", "8GB/s")]
    ]
    topology["resources"] = [
        {"id": name, "capacity": capacity,
         "edges": [{"src": rank, "dst": "fast", "bidirectional": True}
                   for rank in task["ranks"]]}
        for name, capacity in [("shared-a", "4GB/s"), ("shared-b", "5GB/s")]
    ]
    graph = Topology.from_data(topology)
    algorithm = "reverse_doubling" if operator == "all-gather" else "xor_tree"
    hint = {"algorithm": algorithm, "split": 1, "resource_weight": 0, "rank_order": "input"}
    # Retain the valid first candidate, then attach a separately measured route
    # alternative. This controlled selection exercises the service's acceptance
    # branch without depending on which candidate a timed search happens to win.
    initial = solve_schedule(task, graph, hint, time_budget_s=1e-9)
    alternative = build(task, graph, 1, kind=algorithm, resource_weight=4,
                        order=task["ranks"], packets=2)
    report = replay(task, graph, alternative)
    initial["strategy"]["candidates"].append({
        "source": "local_search", "requested": {**hint, "resource_weight": 4},
        "actual": {**hint, "resource_weight": 4, "packets": 2},
        "valid": report["valid"], "selected": False,
        "total_time_us": report["total_time_us"],
    })
    initial["strategy"]["route_variation"] = True
    return task, topology, initial


class RouteRefinementAuditTests(unittest.TestCase):
    def test_winning_refinement_has_one_selection_and_fresh_metrics(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, topology, initial = congested_fixture(operator)
                initial_snapshot = deepcopy(initial)
                with patch("src.service.solve_schedule", return_value=deepcopy(initial)), \
                        patch("src.service.refine", wraps=refine) as refinement:
                    result = service.solve_input({"task": task, "topology": topology}, use_llm=False)
                refinement.assert_called_once()
                report = replay(task, topology, result["answer"])
                search = result["diagnostics"]["search"]
                details = search["route_refinement"]
                selected = [record for record in search["candidates"] if record["selected"]]
                self.assertTrue(report["valid"], report)
                self.assertGreater(details["improved_rounds"], 0)
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0]["source"], "route_refinement")
                parent = search["candidates"][selected[0]["parent_candidate"]]
                self.assertFalse(parent["selected"])
                self.assertEqual(parent["requested"], initial["strategy"]["model_requested"])
                self.assertAlmostEqual(parent["total_time_us"], details["before_us"])
                self.assertAlmostEqual(selected[0]["total_time_us"], report["total_time_us"])
                self.assertAlmostEqual(details["after_us"], report["total_time_us"])
                self.assertAlmostEqual(result["answer"]["metrics"]["total_time"] * 1e6, report["total_time_us"])
                for metric, measured in zip(result["answer"]["metrics"]["round_times"], report["round_times_us"]):
                    self.assertAlmostEqual(metric * 1e6, measured)
                self.assertEqual(initial, initial_snapshot)
                self.assertNotIn("strategy", result["answer"])

    def test_failed_optional_refinement_keeps_verified_answer_and_selection(self):
        task, topology, initial = congested_fixture("all-reduce")
        with patch("src.service.solve_schedule", return_value=deepcopy(initial)), \
                patch("src.service.refine", side_effect=ValueError("synthetic optional routing failure")):
            result = service.solve_input({"task": task, "topology": topology}, use_llm=False)
        expected = {key: value for key, value in initial.items() if key != "strategy"}
        self.assertEqual(result["answer"], expected)
        search = result["diagnostics"]["search"]
        self.assertEqual(search["candidates"], initial["strategy"]["candidates"])
        self.assertFalse(search["route_refinement"]["accepted"])
        self.assertTrue(result["diagnostics"]["local_replay"]["valid"])

    def test_expired_optional_refinement_does_not_invent_selected_candidate(self):
        task, topology, initial = congested_fixture("all-gather")

        def expired(task, graph, solution, **kwargs):
            return refine(task, graph, solution, **{**kwargs, "time_budget_s": 0})

        with patch("src.service.solve_schedule", return_value=deepcopy(initial)), \
                patch("src.service.refine", side_effect=expired):
            result = service.solve_input({"task": task, "topology": topology}, use_llm=False)
        search = result["diagnostics"]["search"]
        self.assertEqual(search["route_refinement"]["improved_rounds"], 0)
        self.assertEqual(search["candidates"], initial["strategy"]["candidates"])
        self.assertTrue(result["diagnostics"]["local_replay"]["valid"])


if __name__ == "__main__":
    unittest.main()
