"""Small offline checks for saved-reply SDK comparison and evidence boundaries."""

import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from src.config import Settings
from tests import compare_fixed_versions as fixed
from tests import compare_versions as compare
from tests.test_solver import make_case


class FixedComparisonTests(unittest.TestCase):
    def setUp(self):
        folder = compare.PROJECT / f"artifacts/v{compare.__version__}/reports"
        folder.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="fixed-compare-test-", dir=folder)
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.sample = compare.Sample("fixture/ag", *make_case(2))
        self.recordings = self.folder / "recordings"
        self.input_path = self.recordings / "inputs/fixture-ag.json"
        self.response_path = self.recordings / "v1.4/fixture-ag-r01.json"
        self.payload = {
            "model": "recorded-model", "messages": [
                {"role": "system", "content": "Original system prompt"},
                {"role": "user", "content": json.dumps({"problem": {
                    "base": "recorded-base", "groups": []}}, ensure_ascii=False)}],
            "temperature": 0.0, "max_tokens": 317, "stream": False}
        self.raw = '{\n  "base": "recorded-base",\n  "counts": []\n}\n'
        self.source = {
            "status": "success", "metadata": {"llm_calls": 1, "total_tokens": 19},
            "output": {"diagnostics": {"llm": {
                "called": True, "model": "recorded-model", "request_evidence": self.payload,
                "request_sha256": fixed.sha256(fixed.payload_bytes(self.payload)),
                "response_sha256": fixed.sha256(self.raw.encode()), "finish_reason": "stop",
                "response_evidence": {"format": "raw_model_content", "content": self.raw,
                                      "credential_redacted": False},
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}}}}
        compare.write_json(self.response_path, self.source)
        compare.write_json(self.input_path, self.sample.request())
        self.rows = [{"sample": "fixture/ag", "version": "1.4", "repeat": 1,
                      "response_file": "v1.4/fixture-ag-r01.json"},
                     {"sample": "fixture/ag", "version": "1.4", "repeat": 2,
                      "response_file": "missing-later-response.json"}]
        compare.write_json(self.recordings / "summary.json", {"rows": self.rows})

    def recording(self):
        return fixed.load_recordings(self.recordings, [self.sample])[self.sample.label]

    def test_first_original_reply_and_historical_usage_are_retained(self):
        recording = self.recording()
        self.assertEqual(recording.raw, self.raw)
        self.assertEqual(recording.payload, self.payload)
        self.assertEqual(recording.response_path, self.response_path)
        reference = recording.reference()
        self.assertEqual(reference["historical_usage_not_measured"]["total_tokens"], 19)
        self.assertEqual(reference["source_input_file_sha256"], fixed.sha256(self.input_path.read_bytes()))

    def test_saved_input_must_match_public_input_exactly(self):
        saved = self.sample.request()
        saved["constraints"]["timeout_ms"] += 1
        compare.write_json(self.input_path, saved)
        with self.assertRaisesRegex(ValueError, "public input differs"):
            self.recording()

    def test_missing_sample_changed_source_and_escaping_response_are_rejected(self):
        for summary, message in [
            ({"rows": []}, "no first v1.4"),
            ({"rows": self.rows, "source_files_changed_during_run": True}, "source changed"),
            ({"rows": [{**self.rows[0], "response_file": "../outside.json"}]}, "within its run"),
            ({"rows": [{**self.rows[0], "response_file": "../.env"}]}, "within its run"),
        ]:
            with self.subTest(message=message):
                compare.write_json(self.recordings / "summary.json", summary)
                with self.assertRaisesRegex(ValueError, message):
                    self.recording()

    def test_modified_or_redacted_raw_response_is_rejected(self):
        for change in ("content", "redacted", "synthetic"):
            source = deepcopy(self.source)
            llm = source["output"]["diagnostics"]["llm"]
            if change == "content":
                llm["response_evidence"]["content"] += " "
            elif change == "redacted":
                llm["response_evidence"]["credential_redacted"] = True
            else:
                llm["synthetic"] = True
            compare.write_json(self.response_path, source)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.recording()

    def test_mock_transport_keeps_original_base_counts_and_zero_actual_usage(self):
        recording = self.recording()
        seen = []
        complete = fixed.recorded_completion(recording, 2.0, seen)
        settings = Settings("http://unused.invalid", "", "recorded-model", max_tokens=317)
        changed_prompt = json.dumps({"problem": {"base": "new-base", "groups": []}})
        with patch.object(fixed.time, "sleep") as sleep:
            raw, metadata = complete("new system", changed_prompt, settings, temperature=0.0)
        sleep.assert_called_once_with(2.0)
        self.assertEqual(raw, self.raw)
        self.assertEqual(json.loads(raw), {"base": "recorded-base", "counts": []})
        self.assertEqual(metadata["usage"], {})
        self.assertTrue(metadata["synthetic"])
        self.assertEqual(metadata["actual_network_calls"], 0)
        self.assertEqual(metadata["actual_model_calls"], 0)
        self.assertEqual(metadata["actual_tokens"], 0)
        self.assertEqual(seen[0]["max_tokens"], 317)
        measured = fixed.compare_prompt(recording, seen[0])
        self.assertFalse(measured["base_matches_recording"])
        self.assertFalse(measured["prompt_matches_recording"])
        self.assertTrue(measured["model_matches_recording"])

    def test_exact_payload_comparison_includes_temperature_and_system(self):
        recording = self.recording()
        self.assertTrue(fixed.compare_prompt(recording, deepcopy(self.payload))["request_payload_matches_recording"])
        changed = deepcopy(self.payload)
        changed["temperature"] = 0.2
        measurement = fixed.compare_prompt(recording, changed)
        self.assertTrue(measurement["prompt_matches_recording"])
        self.assertFalse(measurement["request_payload_matches_recording"])
        changed["messages"][0]["content"] += " "
        self.assertFalse(fixed.compare_prompt(recording, changed)["prompt_matches_recording"])

    def test_delay_parser_and_cli_reject_nonfinite_duplicate_or_negative_values(self):
        for value in ("-1", "nan", "inf", "61", "invalid"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                fixed.nonnegative_delay(value)
        self.assertEqual(fixed.nonnegative_delay("0.5"), 0.5)
        for args in (["--delays", "0", "0.0"], ["--repeat", "0"], ["--delays", "nan"],
                     ["--samples", "mesh2d/ag", "--all-public"]):
            with self.subTest(args=args), patch.object(compare, "apply_limits") as limits, \
                    patch.object(compare, "configure_environment") as configure, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                fixed.main(args)
            self.assertEqual(error.exception.code, 2)
            limits.assert_not_called()
            configure.assert_not_called()

    def test_dry_run_never_loads_configuration_applies_limits_or_creates_output(self):
        output = self.folder / "new-output"
        versions = [compare.Version("1.4", compare.PROJECT / "artifacts/v1.4/source/src", False),
                    compare.Version(compare.__version__, compare.PROJECT / "src", True)]
        with patch.object(compare, "resolve_versions", return_value=versions), \
                patch.object(compare, "load_samples", return_value=[self.sample]), \
                patch.object(compare, "check_output", return_value=output), \
                patch.object(compare, "configure_environment") as configure, \
                patch.object(compare, "apply_limits") as limits, \
                redirect_stdout(io.StringIO()) as stdout:
            result = fixed.main(["--recordings-dir", str(self.recordings), "--dry-run", "--repeat", "2"])
        self.assertEqual(result, 0)
        self.assertFalse(output.exists())
        configure.assert_not_called()
        limits.assert_not_called()
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["expected_synthetic_sdk_requests"], 8)
        self.assertEqual(plan["actual_model_calls"], 0)

    def test_network_and_configuration_guards_never_echo_arguments(self):
        guard = fixed.OfflineGuard()
        for action, counter in ((guard.network, "network_attempts"),
                                (guard.configuration, "configuration_attempts")):
            with self.assertRaises(RuntimeError) as error:
                action("secret-valued-argument")
            self.assertNotIn("secret", str(error.exception))
            self.assertEqual(getattr(guard, counter), 1)

    def test_two_small_sdk_versions_preserve_response_and_flag_mismatched_prompt(self):
        """Four 2-rank requests cover rotation, HTTP JSON and SDK trace boundaries."""
        versions = compare.resolve_versions(["1.4", "current"])
        output = self.folder / "sdk-output"
        recording = self.recording()
        with patch.object(compare, "configure_environment", side_effect=AssertionError("no credentials")), \
                patch("src.config.read_env_file", side_effect=AssertionError("no dotenv")), \
                redirect_stdout(io.StringIO()):
            report = asyncio.run(fixed.run(versions, [self.sample], {self.sample.label: recording},
                                          2, [0.0], output, {"scope": "unit test; limits not applied"}))
        self.assertTrue(report["valid"], report)
        self.assertFalse(report["controlled_comparison"])
        self.assertFalse(report["source_files_changed_during_run"])
        self.assertTrue(report["recording_files_unchanged"])
        self.assertEqual(report["blocked_network_attempts"], 0)
        self.assertEqual(report["blocked_configuration_reads"], 0)
        rows = report["rows"]
        self.assertEqual([row["version"] for row in rows], ["1.4", compare.__version__, compare.__version__, "1.4"])
        self.assertEqual(report["summary"]["replay_valid"], 4)
        for row in rows:
            with self.subTest(version=row["version"], repeat=row["repeat"]):
                body = (output / row["response_file"]).read_bytes()
                saved = json.loads(body)
                llm = saved["output"]["diagnostics"]["llm"]
                self.assertEqual(llm["response_evidence"]["content"], self.raw)
                self.assertEqual(llm["request_evidence"]["model"], "recorded-model")
                self.assertEqual(llm["request_evidence"]["max_tokens"], 64 if row["version"] in ("1.5", "1.6", "1.9", "1.11") else 317)
                self.assertEqual(len(body), row["response_bytes"])
                self.assertGreater(row["request_s"], 0)
                self.assertGreater(row["request_cpu_s"], 0)
                self.assertEqual(row["sdk_synthetic_llm_calls"], 1)
                self.assertEqual(row["sdk_synthetic_tokens"], 0)
                self.assertEqual(row["actual_model_calls"], 0)
                self.assertFalse(row["base_matches_recording"])
                self.assertFalse(row["model_accepted"])
                self.assertNotIn("calls", row)
                self.assertNotIn("tokens", row)
        self.assertEqual(compare.read_object(output / "inputs/fixture-ag.json"), self.sample.request())
        # Guard patches must not survive after the comparison context.
        self.assertFalse(hasattr(socket.socket.connect, "assert_called"))


if __name__ == "__main__":
    unittest.main()
