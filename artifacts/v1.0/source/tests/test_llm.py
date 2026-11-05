import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from src.config import ConfigurationError, Settings, completion_endpoint, load_settings, read_env_file
from src.llm import LLMError, parse_strategy, plan, validate_strategy


class ConfigTests(unittest.TestCase):
    def test_endpoint_normalization(self):
        self.assertEqual(completion_endpoint("http://example.test/v1/"), "http://example.test/v1/chat/completions")
        self.assertEqual(completion_endpoint("https://example.test"), "https://example.test/v1/chat/completions")
        self.assertEqual(completion_endpoint("https://example.test/api/v1/chat/completions"), "https://example.test/api/v1/chat/completions")
        for url in ("file:///private", "https://example.test/unknown", "https://user:secret@example.test/v1"):
            with self.assertRaises(ConfigurationError):
                completion_endpoint(url)

    def test_environment_precedence_and_no_secret_in_repr(self):
        # Tests create temporary files inside the project only.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            path = Path(folder) / ".env"
            path.write_text('PLATFORM_LLM_URL=http://example.test/v1\nPLATFORM_API_KEY="fake-secret"\n'
                            'PLATFORM_LLM_MODEL=old-model\nAGENTCC_LLM_TIMEOUT_S=5\n')
            settings = load_settings(path, {"PLATFORM_LLM_MODEL": "new-model"})
            self.assertEqual(settings.model, "new-model")
            self.assertEqual(settings.api_key, "fake-secret")
            self.assertEqual(settings.timeout_s, 5)
            self.assertNotIn("fake-secret", repr(settings))
            self.assertNotIn("example.test", repr(settings))

    def test_missing_configuration_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            with self.assertRaisesRegex(ConfigurationError, "PLATFORM_LLM_URL"):
                load_settings(Path(folder) / "absent", {})

    def test_invalid_env_does_not_echo_contents(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            path = Path(folder) / ".env"
            path.write_text('PLATFORM_API_KEY="fake-secret\n')
            with self.assertRaises(ConfigurationError) as failure:
                read_env_file(path)
            self.assertNotIn("fake-secret", str(failure.exception))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings("http://example.test/v1/chat/completions", "fake-secret", "test-model")
        self.task = {"operator": "all-gather", "ranks": ["r0", "r1"],
                     "data_size": {"per_rank_bytes": 32768},
                     "split_policy": {"candidates": [1, 2], "alignment_bytes": 4096}}

    def test_strict_strategy_validation(self):
        self.assertEqual(parse_strategy('```json\n{"split":2}\n```', [1, 2]), {"split": 2})
        for strategy in ({"split": True}, {"split": 3}, {"resource_weight": float("nan")},
                         {"resource_weight": -1}, {"algorithm": "execute_shell"},
                         {"rank_order": "unknown"}, {"path": ["r0", "r1"]}, {}):
            with self.assertRaises(LLMError):
                validate_strategy(strategy, [1, 2])

    @patch("src.llm.build_opener")
    def test_success_and_bounded_prompt(self, factory):
        response = MagicMock()
        response.read.return_value = json.dumps({"choices": [{"message": {"content": '{"algorithm":"tree","split":2}'},
                                                               "finish_reason": "stop"}],
                                                "usage": {"total_tokens": 50, "secret": "discard"}}).encode()
        factory.return_value.open.return_value.__enter__.return_value = response
        strategy, metadata = plan(self.task, {"edge_count": 4}, self.settings)
        self.assertEqual(strategy["split"], 2)
        self.assertTrue(metadata["called"])
        self.assertEqual(metadata["usage"], {"total_tokens": 50})
        request = factory.return_value.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer fake-secret")
        self.assertNotIn("fake-secret", request.data.decode())
        self.assertNotIn("fake-secret", str(metadata))
        self.assertEqual(factory.return_value.open.call_args.kwargs["timeout"], 15)
        self.assertEqual(metadata["request_evidence"], json.loads(request.data))
        self.assertEqual(metadata["request_sha256"], hashlib.sha256(request.data).hexdigest())
        self.assertEqual(metadata["response_strategy"], strategy)
        self.assertEqual(json.loads(metadata["response_evidence"]["strategy_json"]), strategy)
        self.assertNotIn("example.test", json.dumps(metadata))

    @patch("src.llm.build_opener")
    def test_current_topology_and_task_are_auditable(self, factory):
        response = MagicMock()
        raw_strategy = '{"algorithm":"tree","split":2,"resource_weight":3,"rank_order":"topology"}'
        response.read.return_value = json.dumps({"choices": [{"message": {"content": raw_strategy},
                                                               "finish_reason": "stop"}]}).encode()
        factory.return_value.open.return_value.__enter__.return_value = response
        self.task.update({"task_id": "task-current", "instance_id": "case-current", "split": 1,
                          "allow_detour": True, "allow_copy": True,
                          "communication_constraints": {"send_only_possessed_chunks": True}})
        summary = {
            "input_sha256": "a" * 64,
            "direction_counts": {"asymmetric_pairs": 2, "unpaired_edges": 1},
            "rank_features": [{"rank": "r0", "out_degree": 1}],
            "edge_samples": [{"src": "r0", "dst": "s0", "bw_Bps": 1e9}],
            "resource_samples": [{"id": "shared", "capacity_Bps": 1e9,
                                   "edges": [["r0", "s0"], ["r1", "s0"]]}],
        }
        strategy, metadata = plan(self.task, summary, self.settings)
        actual_request = factory.return_value.open.call_args.args[0]
        sent_payload = json.loads(actual_request.data)
        sent_input = sent_payload["messages"][1]["content"]
        evidence_input = json.loads(metadata["request_evidence"]["messages"][1]["content"])
        self.assertEqual(evidence_input, json.loads(sent_input))
        self.assertEqual(evidence_input["topology"], summary)
        self.assertEqual(evidence_input["task"]["communication_constraints"],
                         self.task["communication_constraints"])
        self.assertEqual(evidence_input["task"]["task_id"], "task-current")
        self.assertEqual(metadata["input_sha256"], hashlib.sha256(sent_input.encode()).hexdigest())
        self.assertEqual(metadata["source_input_sha256"], "a" * 64)
        self.assertEqual(metadata["response_sha256"], hashlib.sha256(raw_strategy.encode()).hexdigest())
        strategy["split"] = 1
        self.assertEqual(metadata["response_strategy"]["split"], 2)
        # Changing a resource produces a different actual prompt and input digest.
        summary["resource_samples"][0]["capacity_Bps"] = 2e9
        _, changed = plan(self.task, summary, self.settings)
        self.assertNotEqual(changed["input_sha256"], metadata["input_sha256"])
        self.assertEqual(evidence_input["topology"]["resource_samples"][0]["capacity_Bps"], 1e9)

    @patch("src.llm.build_opener")
    def test_auth_errors_are_sanitized(self, factory):
        factory.return_value.open.side_effect = HTTPError(
            "http://example.test/?key=fake-secret", 401, "fake-secret", {}, io.BytesIO(b"fake-secret"))
        with self.assertRaisesRegex(LLMError, "authentication") as failure:
            plan(self.task, {}, self.settings)
        self.assertNotIn("fake-secret", str(failure.exception))

    @patch("src.llm.build_opener")
    def test_timeout_does_not_silently_fallback(self, factory):
        factory.return_value.open.side_effect = URLError(TimeoutError("fake-secret"))
        with self.assertRaisesRegex(LLMError, "timed out") as failure:
            plan(self.task, {}, self.settings)
        self.assertNotIn("fake-secret", str(failure.exception))

    @patch("src.llm.build_opener")
    def test_invalid_response_format(self, factory):
        response = MagicMock()
        response.read.return_value = b'{"choices":[]}'
        factory.return_value.open.return_value.__enter__.return_value = response
        with self.assertRaisesRegex(LLMError, "response format"):
            plan(self.task, {}, self.settings)


if __name__ == "__main__":
    unittest.main()
