"""Comparison-runner input and evidence checks; no model or network calls."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import compare_versions as compare
from tests.test_solver import make_case


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        folder = compare.PROJECT / "artifacts/v1.3/reports"
        folder.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="compare-test-", dir=folder)
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)

    def source(self, version, *, current=False):
        folder = self.project / ("src" if current else f"artifacts/v{version}/source/src")
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text(f'__version__ = "{version}"\n')
        (folder / "my_agent.py").write_text("class MyAgent: pass\n")
        return folder

    def sample(self, label="fixture/ag"):
        task, topology = make_case(2)
        folder = self.project / "materials/spec/examples/256-rank" / label
        folder.mkdir(parents=True)
        (folder / "task.json").write_text(json.dumps(task))
        (folder / "topology.json").write_text(json.dumps(topology))
        return folder

    def test_versions_bind_to_explicit_archives_and_current_source(self):
        self.source("1.1")
        self.source("1.2")
        current = self.source("1.3", current=True)
        versions = compare.resolve_versions(["v1.1", "1.2", "current"], project=self.project,
                                            current_version="1.3")
        self.assertEqual([version.label for version in versions], ["1.1", "1.2", "1.3"])
        self.assertEqual([version.current for version in versions], [False, False, True])
        self.assertEqual(versions[-1].source, current)

    def test_missing_archive_is_reported_before_any_call(self):
        with self.assertRaisesRegex(ValueError, "source is missing"):
            compare.resolve_versions(["1.1"], project=self.project, current_version="1.3")

    def test_duplicate_aliases_and_unsafe_versions_are_rejected(self):
        self.source("1.3", current=True)
        with self.assertRaisesRegex(ValueError, "duplicate version"):
            compare.resolve_versions(["current", "1.3"], project=self.project, current_version="1.3")
        for label in ["../src", "v1.2/../../src", "1", "latest"]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "invalid version"):
                compare.resolve_versions([label], project=self.project, current_version="1.3")

    def test_every_version_takes_every_order_position(self):
        versions = ["1.1", "1.2", "1.3"]
        orders = [compare.rotating_order(versions, 0, repeat) for repeat in range(3)]
        for index in range(3):
            self.assertEqual({order[index] for order in orders}, set(versions))
        self.assertEqual(compare.rotating_order(versions, 1, 0), orders[1])

    def test_bad_samples_are_rejected_before_environment_setup(self):
        self.sample()
        for label in ["../src/ag", "fixture/unknown", "missing/ag"]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                compare.load_samples([label], project=self.project)
        with self.assertRaisesRegex(ValueError, "distinct"):
            compare.load_samples(["fixture/ag", "fixture/ag"], project=self.project)

    def test_malformed_and_wrong_shape_json_are_rejected(self):
        folder = self.sample()
        for text in ["not json", "[]", '{"operator":NaN}']:
            (folder / "task.json").write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                compare.load_samples(["fixture/ag"], project=self.project)

    def test_bad_task_constraints_and_wrong_operator_are_rejected(self):
        folder = self.sample()
        task = json.loads((folder / "task.json").read_text())
        task["operator"] = "all-reduce"
        (folder / "task.json").write_text(json.dumps(task))
        with self.assertRaisesRegex(ValueError, "operator does not match"):
            compare.load_samples(["fixture/ag"], project=self.project)
        task["operator"] = "all-gather"
        task["split_policy"]["candidates"] = [0]
        (folder / "task.json").write_text(json.dumps(task))
        with self.assertRaisesRegex(ValueError, "validation"):
            compare.load_samples(["fixture/ag"], project=self.project)

    def test_input_preflight_and_request_preserve_actual_task(self):
        self.sample()
        samples = compare.load_samples(all_public=True, project=self.project)
        self.assertEqual([sample.label for sample in samples], ["fixture/ag"])
        request = samples[0].request()
        self.assertEqual(request["input"]["task"], samples[0].task)
        self.assertEqual(request["input"]["topology"], samples[0].topology)
        self.assertEqual(request["constraints"]["timeout_ms"], 180000)

    def test_output_cannot_escape_or_overwrite_a_completed_experiment(self):
        with self.assertRaisesRegex(ValueError, "inside the project"):
            compare.check_output("../outside", project=self.project)
        output = self.project / "existing"
        output.mkdir()
        (output / "summary.json").write_text("preserved")
        with self.assertRaisesRegex(ValueError, "not overwritten"):
            compare.check_output(output, project=self.project)
        self.assertEqual((output / "summary.json").read_text(), "preserved")

    def test_invalid_cli_arguments_do_not_load_credentials(self):
        for args in [["--repeat", "0"], ["--repeat", "1.5"],
                     ["--samples", "mesh2d/ag", "--all-public"], ["--versions", "../../src"]]:
            with self.subTest(args=args), patch.object(compare, "configure_environment") as configure, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                compare.main(args)
            self.assertEqual(error.exception.code, 2)
            configure.assert_not_called()

    def test_dry_run_does_not_load_credentials_apply_limits_or_create_output(self):
        self.source("1.3", current=True)
        self.sample()
        versions = compare.resolve_versions(["current"], project=self.project, current_version="1.3")
        samples = compare.load_samples(["fixture/ag"], project=self.project)
        output = self.project / "new-output"
        with patch.object(compare, "resolve_versions", return_value=versions), \
                patch.object(compare, "load_samples", return_value=samples), \
                patch.object(compare, "check_output", return_value=output), \
                patch.object(compare, "configure_environment") as configure, \
                patch.object(compare, "apply_limits") as limits, redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(compare.main(["--dry-run", "--repeat", "3"]), 0)
        configure.assert_not_called()
        limits.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(json.loads(stream.getvalue())["expected_model_calls"], 3)

    def test_measurements_record_real_metadata_without_inventing_model_adoption(self):
        sample = compare.Sample("fixture/ag", *make_case(2))
        checked = {"valid": True, "total_time_us": 12.0, "nonempty_rounds": 1, "split": 1}
        data = {"status": "success", "metadata": {"llm_calls": 1, "total_tokens": 25},
                "output": {"diagnostics": {"llm": {"duration_ms": 200},
                            "phases_s": {"construct_s": 0.1}, "search": {"model_edit": {
                                "accepted": True, "selected": False}}}}}
        with patch.object(compare, "replay", return_value=checked):
            row = compare.response_measurements(sample, data, 200)
        self.assertTrue(row["valid"])
        self.assertTrue(row["model_accepted"])
        self.assertFalse(row["model_selected"])
        self.assertEqual((row["calls"], row["tokens"], row["time_us"]), (1, 25, 12.0))
        self.assertEqual(row["phases_s"], {"construct_s": 0.1})
        self.assertNotIn("score", row)
        self.assertFalse(compare.response_measurements(sample, data, 500)["valid"])

    def test_source_change_is_visible_in_hashes(self):
        source = self.source("1.3", current=True)
        versions = compare.resolve_versions(["current"], project=self.project, current_version="1.3")
        before = compare.source_hashes(versions, project=self.project)
        (source / "my_agent.py").write_text("class MyAgent: changed = True\n")
        after = compare.source_hashes(versions, project=self.project)
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
