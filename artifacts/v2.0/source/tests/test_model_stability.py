"""Frozen-prompt evidence and CLI checks; all completions are local mocks."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import Settings
from src.llm import LLMError
from tests import model_stability as stability


class ModelStabilityTests(unittest.TestCase):
    def setUp(self):
        folder = stability.PROJECT / "artifacts/v1.4/reports"
        folder.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="stability-test-", dir=folder)
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.settings = Settings("http://example.test/v1/chat/completions", "fake-secret", "test-model")

    def fixture(self, *, name="sample-r01.json", named=False):
        group = ({"round": 1, "max_chunks": 2, "delta": [], "paths": [[], []], "bounds": []}
                 if named else [1, 2])
        request = {"model": "test-model", "messages": [
            {"role": "system", "content": "Exact system text\nincluding whitespace"},
            {"role": "user", "content": json.dumps({"problem": {"base": "fixed", "groups": [group]}})},
        ], "temperature": 0.0, "max_tokens": 64, "stream": False}
        path = self.project / name
        saved = {"status": "success", "metadata": {"llm_calls": 1}, "output": {"diagnostics": {"llm": {
            "called": True, "request_evidence": request,
            "request_sha256": stability.digest(json.dumps(request, ensure_ascii=False).encode()),
        }}}}
        path.write_text(json.dumps(saved))
        return stability.load_prompts([path], project=self.project)[0]

    def test_v2_positional_and_v3_named_group_bounds_are_both_supported(self):
        positional = self.fixture(name="positional.json")
        named = self.fixture(name="named.json", named=True)
        self.assertEqual(positional.maxima, named.maxima)
        self.assertEqual(named.maxima, (2,))
        for prompt in [positional, named]:
            self.assertEqual(stability.validate_reply('{"base":"fixed","counts":[2]}', prompt)["counts"], [2])
            with self.assertRaisesRegex(ValueError, "bounds"):
                stability.validate_reply('{"base":"fixed","counts":[3]}', prompt)

    def test_invalid_named_group_bounds_are_rejected_before_networking(self):
        for maximum in [True, -1, 1.5, None]:
            prompt = self.fixture(named=True)
            saved = json.loads(prompt.path.read_text())
            llm = saved["output"]["diagnostics"]["llm"]
            supplied = json.loads(llm["request_evidence"]["messages"][1]["content"])
            supplied["problem"]["groups"][0]["max_chunks"] = maximum
            llm["request_evidence"]["messages"][1]["content"] = json.dumps(supplied)
            llm.pop("request_sha256")
            prompt.path.write_text(json.dumps(saved))
            with self.subTest(maximum=maximum), self.assertRaisesRegex(ValueError, "invalid saved response"):
                stability.load_prompts([prompt.path], project=self.project)

    @staticmethod
    def completion(content):
        def call(system, user, settings, *, allow_truncated, temperature):
            payload = {"model": settings.model, "messages": [{"role": "system", "content": system},
                       {"role": "user", "content": user}], "temperature": temperature,
                       "max_tokens": settings.max_tokens, "stream": False}
            return content, {"request_evidence": payload, "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                             "total_tokens": 10}, "duration_ms": 12.5, "finish_reason": "stop"}
        return call

    def test_default_dry_run_never_reads_credentials_or_creates_output(self):
        prompt = self.fixture()
        output = self.project / "output"
        with patch.object(stability, "load_settings") as settings, patch.object(stability, "_chat_completion") as call, \
                redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(stability.main(["--responses", str(prompt.path), "--output-dir", str(output)]), 0)
        self.assertEqual(json.loads(stream.getvalue())["expected_requests"], 6)
        settings.assert_not_called()
        call.assert_not_called()
        self.assertFalse(output.exists())

    def test_two_prompts_rotate_temperature_order_for_each_repeat(self):
        prompts = [self.fixture(name="one.json"), self.fixture(name="two.json")]
        plan = list(stability.request_order(prompts, 3))
        self.assertEqual(len(plan), 12)
        for index in range(0, len(plan), 2):
            self.assertEqual({plan[index][3], plan[index + 1][3]}, {0.0, 0.2})
        self.assertNotEqual(plan[0][3], plan[2][3])
        self.assertNotEqual(plan[0][3], plan[4][3])

    def test_exact_prompt_and_recorded_output_limit_are_retained(self):
        prompt = self.fixture()
        self.assertEqual(prompt.payload["messages"][0]["content"], "Exact system text\nincluding whitespace")
        content = '{ "counts": [1], "base": "fixed" }'
        call = self.completion(content)
        observed = []
        def capture(*args, **kwargs):
            observed.append((args, kwargs))
            return call(*args, **kwargs)
        with redirect_stdout(io.StringIO()):
            report = stability.run([prompt], 3, self.project / "run", self.settings, completion=capture)
        self.assertTrue(report["complete"])
        self.assertTrue(report["source_files_unchanged"])
        self.assertEqual(report["known_total_tokens"], 60)
        self.assertEqual(report["protocol_valid_requests"], 6)
        self.assertFalse(report["replay_performed"])
        self.assertFalse(report["sdk_benchmark"])
        for args, kwargs in observed:
            self.assertEqual(args[0], prompt.payload["messages"][0]["content"])
            self.assertEqual(args[1], prompt.payload["messages"][1]["content"])
            self.assertEqual(args[2].max_tokens, 64)
            self.assertTrue(kwargs["allow_truncated"])
        evidence = json.loads((self.project / "run" / report["rows"][0]["evidence_file"]).read_text())
        self.assertEqual(evidence["raw_response"], content)
        self.assertEqual(evidence["request_payload"], evidence["metadata"]["request_evidence"])
        self.assertEqual(report["groups"][0]["canonical_answer_count"], 1)

    def test_protocol_checks_base_exact_count_integer_bounds_and_truncation(self):
        prompt = self.fixture()
        for value in [{"base": "other", "counts": [1]}, {"base": "fixed", "counts": []},
                      {"base": "fixed", "counts": [True]}, {"base": "fixed", "counts": [1.0]},
                      {"base": "fixed", "counts": [-1]}, {"base": "fixed", "counts": [3]},
                      {"base": "fixed", "counts": [1], "extra": 1}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                stability.validate_reply(json.dumps(value), prompt)
        for content in ['{"base":"fixed","counts":[NaN]}', '{"base":"fixed","counts":[0],"counts":[1]}']:
            with self.assertRaises(ValueError):
                stability.validate_reply(content, prompt)
        with self.assertRaisesRegex(ValueError, "truncated"):
            stability.validate_reply('{"base":"fixed","counts":[1]}', prompt, finish_reason="length")

    def test_variants_preserve_semantic_differences_and_usage(self):
        rows = []
        for temperature in stability.TEMPERATURES:
            for count in [1, 2, 1]:
                rows.append({"prompt": "one", "temperature": temperature, "protocol_valid": True,
                             "canonical_reply": {"base": "fixed", "counts": [count]},
                             "response_received": True, "response_sha256": str(count),
                             "request_s": count, "usage": {"total_tokens": 10}})
        for group in stability.summarize(rows):
            self.assertEqual(group["canonical_answer_count"], 2)
            self.assertTrue(group["variation_observed"])
            self.assertEqual(group["count_values_per_group"], [[1, 2]])
            self.assertEqual(group["request_s"], {"count": 3, "min": 1, "median": 1, "max": 2})
            self.assertEqual(group["known_total_tokens"], 30)
            self.assertNotIn("p95", group)

    def test_missing_usage_and_transport_failure_are_not_reported_as_zero_cost_success(self):
        prompt = self.fixture()
        def fail(*_args, **_kwargs):
            raise LLMError("Official model request timed out")
        with redirect_stdout(io.StringIO()):
            report = stability.run([prompt], 1, self.project / "failed", self.settings, completion=fail)
        self.assertTrue(report["complete"])
        self.assertEqual(report["protocol_valid_requests"], 0)
        self.assertEqual(report["token_observations"], 0)
        self.assertTrue(all(group["tokens"]["median"] is None for group in report["groups"]))
        self.assertTrue(all(group["request_s"]["count"] == 1 for group in report["groups"]))

    def test_credential_with_json_escape_is_redacted_before_saving(self):
        prompt = self.fixture()
        settings = replace(self.settings, api_key='fake\\"secret')
        content = json.dumps({"base": "fixed", "counts": [1], "echo": settings.api_key})
        # Only the raw assistant JSON text contains the escaped credential;
        # no decoded metadata field can trigger detection on its behalf.
        with redirect_stdout(io.StringIO()):
            report = stability.run([prompt], 1, self.project / "redacted", settings,
                                   completion=self.completion(content))
        self.assertEqual(report["protocol_valid_requests"], 0)
        for row in report["rows"]:
            evidence = json.loads((self.project / "redacted" / row["evidence_file"]).read_text())
            self.assertTrue(evidence["credential_redacted"])
            self.assertEqual(json.loads(evidence["raw_response"])["echo"], "[REDACTED]")
            self.assertNotIn(json.dumps(settings.api_key)[1:-1], evidence["raw_response"])

    def test_redaction_covers_unicode_json_and_nested_metadata(self):
        credential = 'fake-密钥-\\"secret'
        value = {"raw": json.dumps({"echo": credential}, ensure_ascii=True),
                 "metadata": [{"echo": credential}]}
        sanitized, redacted = stability.redact_credential(value, credential)
        self.assertTrue(redacted)
        self.assertEqual(json.loads(sanitized["raw"])["echo"], "[REDACTED]")
        self.assertEqual(sanitized["metadata"][0]["echo"], "[REDACTED]")
        self.assertEqual(value["metadata"][0]["echo"], credential)

    def test_escaped_credential_in_frozen_input_prevents_any_request(self):
        prompt = self.fixture()
        settings = replace(self.settings, api_key='fake\\"secret')
        payload = {**prompt.payload, "messages": [
            prompt.payload["messages"][0],
            {"role": "user", "content": json.dumps({"echo": settings.api_key})}]}
        with patch.object(stability, "_chat_completion") as call, self.assertRaisesRegex(ValueError, "credential"):
            stability.run([replace(prompt, payload=payload)], 1, self.project / "refused", settings)
        call.assert_not_called()
        self.assertFalse((self.project / "refused").exists())

    def test_wrong_model_is_rejected_before_creating_output_or_calling(self):
        prompt = self.fixture()
        output = self.project / "wrong-model"
        with patch.object(stability, "_chat_completion") as call, self.assertRaisesRegex(ValueError, "configured model"):
            stability.run([prompt], 1, output, replace(self.settings, model="other"))
        call.assert_not_called()
        self.assertFalse(output.exists())

    def test_invalid_cli_and_inputs_never_load_credentials(self):
        prompt = self.fixture()
        for args in [["--repeat", "0"], ["--repeat", "1.5"], ["--run", "--dry-run"],
                     ["--responses", str(prompt.path), str(prompt.path)], ["--responses", "missing.json"]]:
            with self.subTest(args=args), patch.object(stability, "load_settings") as settings, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                stability.main(args)
            self.assertEqual(error.exception.code, 2)
            settings.assert_not_called()
        with self.assertRaisesRegex(ValueError, "inside the project"):
            stability.local_path("../outside", self.project)

    def test_output_overwrite_and_modified_request_hash_are_rejected(self):
        prompt = self.fixture()
        with self.assertRaisesRegex(ValueError, "not overwritten"):
            stability.check_output(self.project)
        saved = json.loads(prompt.path.read_text())
        saved["output"]["diagnostics"]["llm"]["request_evidence"]["messages"][0]["content"] = "changed"
        prompt.path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "invalid saved response"):
            stability.load_prompts([prompt.path], project=self.project)


if __name__ == "__main__":
    unittest.main()
