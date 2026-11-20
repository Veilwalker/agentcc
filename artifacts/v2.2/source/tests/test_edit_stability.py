"""Stable semantic edit bindings without weakening stale-schedule rejection."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from tests.image_runtime import interpreter_command

from src.collectives import build
from src.llm import build_edit_request
from src.schedule_edit import apply_proposal, extract_problem
from tests.test_route_balance import parallel_case


class EditStabilityTests(unittest.TestCase):
    def fixture(self):
        task, graph = parallel_case("all-reduce")
        baseline = build(task, graph, 4, kind="xor_tree", resource_weight=0)
        return task, graph, baseline

    def test_diagnostic_timers_do_not_change_model_input_or_binding(self):
        task, graph, baseline = self.fixture()
        baseline["strategy"] = {"elapsed_s": 0.123, "candidates": [{"elapsed_s": 0.1}]}
        first, context = extract_problem(task, graph, baseline, time_budget_s=1)
        baseline["strategy"] = {"elapsed_s": 0.987, "candidates": [{"elapsed_s": 0.8}]}
        second, other = extract_problem(task, graph, baseline, time_budget_s=1)
        self.assertEqual(first, second)
        self.assertEqual(context, other)
        self.assertEqual(build_edit_request(task, first), build_edit_request(task, second))
        _, evidence = apply_proposal(task, graph, baseline, context,
                                     {"base": first["base"], "edits": []})
        self.assertEqual(evidence["errors"], ["proposal moves no chunks"])

    def test_different_concrete_catalogues_have_different_bindings(self):
        task, graph, baseline = self.fixture()
        full, _ = extract_problem(task, graph, baseline, time_budget_s=1, max_groups=4)
        limited, context = extract_problem(task, graph, baseline, time_budget_s=1,
                                           max_groups=1, max_rounds=1)
        self.assertNotEqual(len(full["groups"]), len(limited["groups"]))
        self.assertNotEqual(full["base"], limited["base"])
        _, evidence = apply_proposal(task, graph, baseline, context,
                                     {"base": full["base"], "edits": []})
        self.assertIn("base does not match", evidence["errors"][0])

    def test_schedule_mutations_still_invalidate_an_existing_context(self):
        task, graph, baseline = self.fixture()
        problem, context = extract_problem(task, graph, baseline, time_budget_s=1)
        changed = deepcopy(baseline)
        changed["rounds"][0]["transfers"][0]["path"].reverse()
        _, evidence = apply_proposal(task, graph, changed, context,
                                     {"base": problem["base"], "edits": []})
        self.assertIn("stale baseline", evidence["errors"][0])

    def test_process_hash_seed_does_not_change_identical_model_input(self):
        script = """
import hashlib,json
from tests.test_route_balance import parallel_case
from src.collectives import build
from src.schedule_edit import extract_problem
from src.llm import build_edit_request
task,graph=parallel_case('all-reduce')
baseline=build(task,graph,4,kind='xor_tree',resource_weight=0)
problem,_=extract_problem(task,graph,baseline,time_budget_s=1)
system,user,_=build_edit_request(task,problem)
print(hashlib.sha256(json.dumps([system,user]).encode()).hexdigest())
"""
        root = Path(__file__).resolve().parents[1]
        results = [subprocess.check_output([*interpreter_command(), "-c", script], cwd=root,
                                           env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True).strip()
                   for seed in (3, 41, 137)]
        self.assertEqual(len(set(results)), 1)


if __name__ == "__main__":
    unittest.main()
