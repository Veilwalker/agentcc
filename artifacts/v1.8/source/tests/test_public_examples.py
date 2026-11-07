"""Read the actual published two-rank JSON examples as compatibility fixtures."""

import json
from pathlib import Path
import unittest

from src.replay import replay
from src.solver import solve_schedule


EXAMPLES = Path(__file__).resolve().parents[1] / "materials/spec/examples/shared-resource"


@unittest.skipUnless(EXAMPLES.is_dir(), "public samples have not been placed in materials/spec")
class PublicExampleTests(unittest.TestCase):
    def load_case(self, operator):
        task = json.loads((EXAMPLES / f"{operator}-task.json").read_text())
        topology = json.loads((EXAMPLES / "topology.json").read_text())
        response = json.loads((EXAMPLES / f"{operator}-solution.json").read_text())
        return task, topology, response

    def test_published_allgather_solution_matches_documented_time(self):
        report = replay(*self.load_case("all-gather"))
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["nonempty_rounds"], 1)
        self.assertAlmostEqual(report["total_time_us"], 66.836, places=6)

    def test_published_allreduce_solution_matches_the_cost_formula(self):
        report = replay(*self.load_case("all-reduce"))
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["nonempty_rounds"], 2)
        self.assertAlmostEqual(report["total_time_us"], 100.904, places=6)

    def test_generated_solutions_pass_same_rules_as_public_solutions(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, topology, reference = self.load_case(operator)
                answer = solve_schedule(task, topology, time_budget_s=5)
                report = replay(task, topology, answer)
                self.assertTrue(report["valid"], report)
                reference_report = replay(task, topology, reference)
                self.assertLessEqual(report["total_time_us"], reference_report["total_time_us"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
