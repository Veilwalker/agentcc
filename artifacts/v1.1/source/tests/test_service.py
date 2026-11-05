import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.service import solve_input


class PreflightTests(unittest.TestCase):
    def test_invalid_task_is_rejected_before_calling_model(self):
        folder = Path(__file__).resolve().parents[1] / "materials/spec/examples/shared-resource"
        task = json.loads((folder / "all-gather-task.json").read_text())
        topology = json.loads((folder / "topology.json").read_text())
        cases = [{"operator": "broadcast"}, {"ranks": ["r0"]},
                 {"max_nonempty_rounds": True}, {"data_size": {"per_rank_bytes": 0}},
                 {"split_policy": {"candidates": [1], "alignment_bytes": True}}]
        for change in cases:
            invalid = copy.deepcopy(task)
            invalid.update(change)
            with self.subTest(change=change), patch("src.service.plan") as model:
                with self.assertRaises(ValueError):
                    solve_input({"task": invalid, "topology": topology})
                model.assert_not_called()
