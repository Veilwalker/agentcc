"""Saved-response tooling must not rebind raw model decisions or claim calls."""

import json
import unittest
from unittest.mock import patch

from src.collectives import build
from src.config import Settings
from src.replay import replay
from src.schedule_edit import extract_problem
from tests.fixed_model_replay import frozen_repeats, recorded_completion, select_recordings
from tests.test_route_balance import parallel_case


class FixedModelReplayTests(unittest.TestCase):
    def test_repeated_comparison_rows_keep_only_first_14_recording_per_sample(self):
        rows = [
            {"version": "1.3", "sample": "mesh2d/ag", "response_file": "old.json"},
            {"version": "1.4", "sample": "mesh2d/ag", "response_file": "first-ag.json"},
            {"version": "1.4", "sample": "mesh2d/ar", "response_file": "first-ar.json"},
            {"version": "1.4", "sample": "mesh2d/ag", "response_file": "second-ag.json"},
            {"version": "1.4", "sample": "mesh2d/ar", "response_file": "second-ar.json"},
        ]
        selected = select_recordings(rows)
        self.assertEqual([row["response_file"] for row in selected], ["first-ag.json", "first-ar.json"])
        self.assertIs(selected[0], rows[1], "recording metadata must not be rebound or reconstructed")
        self.assertEqual(select_recordings(rows, ["mesh2d/ar", "mesh2d/ar"]), [rows[2]])

    def test_missing_recording_filter_is_explicit_instead_of_silently_skipped(self):
        with self.assertRaisesRegex(ValueError, "no 1.4 recording for samples: cloudmatrix/ag"):
            select_recordings([{"version": "1.3", "sample": "cloudmatrix/ag"}], ["cloudmatrix/ag"])
        with self.assertRaisesRegex(ValueError, "no 1.4 recordings selected"):
            select_recordings([])

    def test_raw_reply_keeps_old_base_even_when_current_binding_differs(self):
        raw = '{"base":"original-base","counts":[3,1]}'
        seen = {}
        fake = recorded_completion(raw, {"finish_reason": "stop"}, 0, seen)
        request = json.dumps({"problem": {"base": "different-current-base", "groups": []}})
        with patch("src.llm.build_opener", side_effect=AssertionError("no network")):
            returned, metadata = fake("test", request,
                Settings("http://unused.invalid", "synthetic-only", "test-model"), temperature=0)
        self.assertEqual(returned, raw)
        self.assertEqual(seen["request"]["problem"]["base"], "different-current-base")
        self.assertTrue(metadata["synthetic"])
        self.assertEqual(metadata["real_model_calls"], 0)
        self.assertEqual(metadata["network_calls"], 0)
        self.assertEqual(metadata["usage"], {})
        self.assertNotIn("synthetic-only", json.dumps(metadata))

    def test_fixed_context_control_and_proposal_produce_identical_decisions(self):
        task, graph = parallel_case("all-reduce")
        baseline = build(task, graph, 4, kind="xor_tree", resource_weight=0)
        problem, context = extract_problem(task, graph, baseline, time_budget_s=1)
        proposal = {"base": problem["base"], "edits": [[g["id"], 1] for g in problem["groups"]]}
        capture = {"graph": graph, "baseline": baseline, "problem": problem, "context": context,
                   "control": baseline, "control_report": replay(task, graph, baseline)}
        rows = frozen_repeats({"task": task}, capture, proposal, 3, 0.001)
        self.assertTrue(all(row["accepted"] for row in rows))
        normalized = [{k: v for k, v in row.items() if k not in {"repeat", "elapsed_s"}} for row in rows]
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[1], normalized[2])


if __name__ == "__main__":
    unittest.main()
