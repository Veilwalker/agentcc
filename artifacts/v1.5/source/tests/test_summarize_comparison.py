"""Small synthetic report fixtures; never execute a solver or model request."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import tempfile
import unittest

from tests import summarize_comparison as analysis


def observation(version, repeat=1, *, sample="mesh2d/ag", request=2.0, communication=100.0,
                tokens=100, selected=False):
    return {"sample": sample, "version": version, "repeat": repeat, "request_s": request,
            "time_us": communication, "tokens": tokens, "valid": True, "calls": 1,
            "status": "success", "http_status": 200, "model_selected": selected}


def summary(rows):
    versions = sorted({row["version"] for row in rows})
    hashes = {version: {"src/service.py": hashlib.sha256(version.encode()).hexdigest()}
              for version in versions}
    return {"schema_version": "agentcc.version-comparison.v1", "complete": True,
            "expected_requests": len(rows), "versions": [{"version": version} for version in versions],
            "rows": rows, "source_sha256_at_start": hashes,
            "source_sha256_at_end": deepcopy(hashes), "source_files_changed_during_run": False}


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        folder = analysis.PROJECT / "artifacts/v1.3/reports"
        folder.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="analysis-test-", dir=folder)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def write(self, name, data):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def test_medians_token_mean_and_geometric_communication_speedup(self):
        rows = []
        for repeat, (duration, tokens) in enumerate([(1, 100), (2, 200), (100, 900)], 1):
            rows.append(observation("1.2", repeat, request=5, communication=200))
            rows.append(observation("1.3", repeat, request=duration, tokens=tokens, selected=repeat==1))
        report = analysis.summarize([self.write("one/summary.json", summary(rows))])
        self.assertEqual(report["status"], "complete")
        group = next(group for group in report["groups"] if group["version"] == "1.3")
        self.assertEqual(group["metrics"]["request_s"]["median"], 2)
        self.assertEqual(group["metrics"]["tokens"]["median"], 200)
        self.assertEqual(group["metrics"]["tokens"]["mean"], 400)
        self.assertAlmostEqual(group["model_selection_rate"], 1/3)
        metrics = report["comparisons"]["1.3"]["1.2"]["metrics"]
        self.assertEqual(metrics["time_us"]["geomean_speedup"], 2)
        self.assertAlmostEqual(metrics["request_s"]["geomean_speedup"], 2.5)
        self.assertFalse(report["official_score_estimate"])

    def test_independent_runs_merge_only_after_matching_source_hashes(self):
        first = self.write("one/summary.json", summary([observation("1.3", request=1)]))
        second = self.write("two/summary.json", summary([observation("1.3", request=3)]))
        report = analysis.summarize([first, second])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["requests"], 2)
        self.assertEqual(report["groups"][0]["metrics"]["request_s"]["median"], 2)
        changed = summary([observation("1.3", request=4)])
        changed["source_sha256_at_start"]["1.3"]["src/service.py"] = "0"*64
        changed["source_sha256_at_end"] = deepcopy(changed["source_sha256_at_start"])
        third = self.write("three/summary.json", changed)
        rejected = analysis.summarize([first, third])
        self.assertEqual(rejected["status"], "rejected")
        self.assertNotIn("groups", rejected)
        self.assertIn("hashes differ", rejected["rejections"][0]["reason"])

    def test_mid_run_source_change_or_unfinished_run_is_rejected(self):
        base = summary([observation("1.3")])
        for key, value in [("source_files_changed_during_run", True), ("complete", False)]:
            data = {**base, key: value}
            result = analysis.summarize([self.write(key+".json", data)])
            self.assertEqual(result["status"], "rejected")
            self.assertNotIn("comparisons", result)
        data = deepcopy(base)
        data["source_sha256_at_end"]["1.3"]["src/service.py"] = "0"*64
        self.assertEqual(analysis.summarize([self.write("mismatch.json", data)])["status"], "rejected")

    def test_different_measurement_conditions_are_not_silently_merged(self):
        old = summary([observation("1.3", request=1)])
        new = summary([observation("1.3", request=2)])
        old["limits"] = {"cpu_affinity": [0], "address_space_limit_bytes": 1024**3}
        new["limits"] = {"cpu_affinity": [0, 1], "address_space_limit_bytes": 1024**3}
        report = analysis.summarize([self.write("one/summary.json", old), self.write("two/summary.json", new)])
        self.assertEqual(report["status"], "rejected")
        self.assertIn("resource limits", report["rejections"][0]["reason"])

    def test_duplicate_paths_content_and_rows_are_not_independent_repeats(self):
        data = summary([observation("1.3")])
        first = self.write("one.json", data)
        copied = self.write("copy.json", data)
        for paths in [[first, first], [first, copied]]:
            self.assertEqual(analysis.summarize(paths)["status"], "rejected")
        duplicated = summary([observation("1.3"), observation("1.3")])
        result = analysis.summarize([self.write("duplicate-row.json", duplicated)])
        self.assertIn("duplicate sample", result["rejections"][0]["reason"])

    def test_fast_failure_does_not_improve_the_latency_comparison(self):
        good = observation("1.3", request=10, tokens=100)
        failed = observation("1.3", 2, request=0.001, tokens=30)
        failed.update(valid=False, status="error", calls=0)
        data = summary([observation("1.2", request=2), good, failed])
        report = analysis.summarize([self.write("failure.json", data)])
        group = next(group for group in report["groups"] if group["version"] == "1.3")
        self.assertEqual(group["valid_rate"], 0.5)
        self.assertEqual(group["metrics"]["request_s"]["median"], 10)
        self.assertEqual(group["tokens_total_all_requests"], 130)
        comparison = report["comparisons"]["1.3"]["1.2"]
        self.assertEqual(comparison["metrics"]["request_s"]["paired_cases"], 0)
        self.assertEqual(len(comparison["validity_regressions"]), 1)

    def test_regressions_and_unrecorded_model_adoption_remain_explicit(self):
        data = summary([observation("1.2", request=1, tokens=50, selected=None),
                        observation("1.3", request=2, tokens=100, selected=False)])
        report = analysis.summarize([self.write("regression.json", data)])
        old = next(group for group in report["groups"] if group["version"] == "1.2")
        self.assertIsNone(old["model_selection_rate"])
        regression = report["comparisons"]["1.3"]["1.2"]["regressions"]
        self.assertEqual({row["metric"] for row in regression}, {"request_s", "tokens"})

    def test_zero_duration_does_not_create_infinite_or_undefined_speedup(self):
        data = summary([observation("1.2", communication=0), observation("1.3", communication=0)])
        report = analysis.summarize([self.write("zero.json", data)])
        metric = report["comparisons"]["1.3"]["1.2"]["metrics"]["time_us"]
        self.assertIsNone(metric["geomean_speedup"])
        self.assertEqual(metric["tied_cases"], 1)
        json.dumps(report, allow_nan=False)

    def test_nonfinite_metrics_and_empty_measurements_are_rejected(self):
        data = summary([observation("1.3", request=float("nan"))])
        self.assertEqual(analysis.summarize([self.write("nonfinite.json", data)])["status"], "rejected")
        data = summary([observation("1.3")])
        data.update(rows=[], expected_requests=0)
        self.assertEqual(analysis.summarize([self.write("empty.json", data)])["status"], "rejected")

    def test_html_escapes_labels_and_has_no_external_dependencies(self):
        label = '</script><img src="https://invalid.test/x">'
        data = summary([observation("1.2", sample=label), observation("1.3", sample=label)])
        report = analysis.summarize([self.write("html.json", data)])
        rendered = analysis.render_html(report, self.folder)
        self.assertIn("本地公开样例，非官方得分", rendered)
        self.assertNotIn(label, rendered)
        self.assertNotIn('<img src=', rendered)
        class Assets(HTMLParser):
            def __init__(self):super().__init__();self.external=[]
            def handle_starttag(self, tag, attrs):
                for name, value in attrs:
                    if name in ("src", "href") and value.startswith(("https:", "http:", "//")):
                        self.external.append(value)
        check=Assets();check.feed(rendered)
        self.assertFalse(check.external)

    def test_cli_writes_both_reports_and_refuses_to_overwrite_an_input(self):
        source = self.write("run/summary.json", summary([observation("1.2"), observation("1.3")]))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(analysis.main([str(source)]), 0)
        self.assertTrue((source.parent / "report.json").is_file())
        self.assertTrue((source.parent / "report.html").is_file())
        original = self.write("report.json", summary([observation("1.3")]))
        content = original.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            analysis.main([str(original)])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(original.read_bytes(), content)

    def test_rejection_is_written_as_a_report_and_returns_failure(self):
        data = summary([observation("1.3")])
        data["source_files_changed_during_run"] = True
        source = self.write("changed/summary.json", data)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(analysis.main([str(source)]), 1)
        saved = json.loads((source.parent / "report.json").read_text())
        self.assertEqual(saved["status"], "rejected")
        self.assertIn("拒绝合并", (source.parent / "report.html").read_text())


if __name__ == "__main__":
    unittest.main()
