"""Audit frozen model-edit inputs, trace lifetime and incumbent provenance.

All model responses are synthetic. These tests never read .env or contact an
external model; they verify context, provenance and error handling boundaries.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src import service
from src.config import Settings
from src.llm import LLMError
from src.replay import replay
from src.solver import SolverError, solve_schedule
from tests.test_solver import make_case


SETTINGS = Settings("http://unused.test/v1/chat/completions", "test-only", "test-model")
HINT = {"algorithm": "tree", "split": 2, "resource_weight": 3, "rank_order": "topology"}


def synthetic_plan(task, problem, **kwargs):
    return {"base": problem["base"], "edits": []}, {"called": True, "usage": {"prompt_tokens": 10, "completion_tokens": 4}}


class PresearchAuditTests(unittest.TestCase):
    def test_model_thread_preserves_request_context(self):
        task, topology = make_case(8)
        marker = ContextVar("test_request_context", default=None)
        observed = []

        def model(*args, **kwargs):
            observed.append(marker.get())
            return synthetic_plan(*args, **kwargs)

        token = marker.set("this-request")
        try:
            with patch("src.service.load_settings", return_value=SETTINGS), patch("src.service.edit_schedule", side_effect=model):
                result = service.solve_input({"task": task, "topology": topology}, search_budget_s=0.1)
        finally:
            marker.reset(token)
        self.assertEqual(observed, ["this-request"])
        self.assertTrue(result["diagnostics"]["local_replay"]["valid"])

    def test_model_span_ends_when_model_returns_not_when_control_finishes(self):
        task, topology = make_case(8)
        presearch_started, model_span_finished = threading.Event(), threading.Event()
        spans, token_totals = [], []
        fake_sdk = ModuleType("agenthack")

        @contextmanager
        def trace(**kwargs):
            spans.append(kwargs)
            operation = SimpleNamespace(set=lambda **values: token_totals.append(values), result=lambda result: None)
            try:
                yield operation
            finally:
                model_span_finished.set()

        fake_sdk.llm_call = trace

        def model(*args, **kwargs):
            self.assertTrue(presearch_started.wait(2), "control was not started concurrently")
            return synthetic_plan(*args, **kwargs)

        original_control = service._refined_candidate

        def control(*args, **kwargs):
            presearch_started.set()
            self.assertTrue(model_span_finished.wait(2), "LLM span incorrectly waits for control completion")
            return original_control(*args, **kwargs)

        with patch.dict("sys.modules", {"agenthack": fake_sdk}), \
                patch("src.service.load_settings", return_value=SETTINGS), \
                patch("src.service.edit_schedule", side_effect=model), \
                patch("src.service._refined_candidate", side_effect=control):
            result = service.solve_input({"task": task, "topology": topology}, search_budget_s=0.1)
        self.assertTrue(result["diagnostics"]["local_replay"]["valid"])
        self.assertEqual(len(spans), 1)
        self.assertEqual(token_totals, [{"input_tokens": 10, "output_tokens": 4}])

    def test_presearch_does_not_mask_model_failure(self):
        task, topology = make_case(8)
        with patch("src.service.load_settings", return_value=SETTINGS), \
                patch("src.service.edit_schedule", side_effect=LLMError("synthetic official-model failure")) as model, \
                patch("src.service.solve_schedule", wraps=solve_schedule) as search:
            with self.assertRaisesRegex(LLMError, "official-model failure"):
                service.solve_input({"task": task, "topology": topology}, search_budget_s=0.1)
        model.assert_called_once()
        self.assertEqual(search.call_count, 2, "seed and concurrent control run, but model failure must still propagate")

    def test_no_valid_edit_base_fails_before_model_call(self):
        task, topology = make_case(8)
        with patch("src.service.load_settings", return_value=SETTINGS), \
                patch("src.service.edit_schedule", side_effect=synthetic_plan) as model, \
                patch("src.service.solve_schedule", side_effect=SolverError("synthetic construction failure")):
            with self.assertRaisesRegex(SolverError, "construction failure"):
                service.solve_input({"task": task, "topology": topology}, search_budget_s=0.1)
        model.assert_not_called()

    def test_same_name_input_changes_have_distinct_current_input_evidence(self):
        task, topology = make_case(8)
        changed = deepcopy(topology)
        changed["edges"][0]["bw"] = "0.25GB/s"
        response = MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": json.dumps(HINT)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }).encode()
        with patch("src.service.load_settings", return_value=SETTINGS), patch("src.llm.build_opener") as factory:
            factory.return_value.open.return_value.__enter__.return_value = response
            first = service.solve_input({"task": task, "topology": topology}, search_budget_s=0.1)
            second = service.solve_input({"task": task, "topology": changed}, search_budget_s=0.1)
        self.assertEqual(factory.return_value.open.call_count, 2)
        old, new = first["diagnostics"]["llm"], second["diagnostics"]["llm"]
        self.assertNotEqual(old["source_input_sha256"], new["source_input_sha256"])
        self.assertNotEqual(old["request_sha256"], new["request_sha256"])
        expected = hashlib.sha256(json.dumps({"task": task, "topology": changed}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(new["source_input_sha256"], expected)
        self.assertTrue(replay(task, changed, second["answer"])["valid"])

    def test_revalidated_incumbent_costs_match_current_topology(self):
        task, topology = make_case(8, "all-reduce")
        previous = solve_schedule(task, topology, {"algorithm": "xor_tree", "split": 4, "resource_weight": 2, "rank_order": "topology"}, time_budget_s=0.1)
        previous_snapshot = deepcopy(previous)
        changed = deepcopy(topology)
        for edge in changed["edges"]:
            edge["bw"] = "0.5GB/s"
        answer = solve_schedule(task, changed, {"algorithm": "tree", "split": 1, "resource_weight": 2, "rank_order": "topology"}, time_budget_s=1e-9, incumbent=previous)
        report = replay(task, changed, answer)
        selected = next(record for record in answer["strategy"]["candidates"] if record["selected"])
        self.assertTrue(report["valid"], report)
        self.assertEqual(selected["source"], "incumbent")
        self.assertTrue(selected["incumbent_revalidated"])
        self.assertAlmostEqual(answer["metrics"]["total_time"] * 1e6, report["total_time_us"])
        self.assertAlmostEqual(selected["total_time_us"], report["total_time_us"])
        self.assertEqual(previous, previous_snapshot, "revalidation must not overwrite historical evidence")
        inherited = [record for record in answer["strategy"]["candidates"] if record["source"] in ("incumbent", "presearch")]
        self.assertEqual(len(inherited), 1, "different input must not inherit unchecked historical costs")
        self.assertNotEqual(answer["strategy"]["topology_sha256"], previous["strategy"]["topology_sha256"])

    def test_same_input_reuse_preserves_model_candidate_evidence(self):
        task, topology = make_case(8, "all-reduce")
        previous = solve_schedule(task, topology, time_budget_s=0.1)
        old_snapshot = deepcopy(previous)
        hint = {"algorithm": "tree", "split": 2, "resource_weight": 3, "rank_order": "input"}
        answer = solve_schedule(task, topology, hint, time_budget_s=0.1, incumbent=previous)
        records = answer["strategy"]["candidates"]
        self.assertEqual(records[0]["source"], "model_seed")
        self.assertTrue(records[0]["valid"])
        self.assertEqual(records[0]["requested"], hint)
        self.assertEqual(records[0]["actual"]["algorithm"], hint["algorithm"])
        self.assertEqual(records[0]["actual"]["resource_weight"], hint["resource_weight"])
        self.assertTrue(any(record["source"] == "model" and record["actual"]["split"] == 2 and record["valid"] for record in records))
        inherited = [record for record in records if record["source"] == "presearch"]
        self.assertEqual(len(inherited), len(previous["strategy"]["candidates"]))
        self.assertEqual(sum(bool(record["selected"]) for record in records), 1)
        self.assertEqual(previous, old_snapshot)
        self.assertTrue(replay(task, topology, answer)["valid"])


if __name__ == "__main__":
    unittest.main()
