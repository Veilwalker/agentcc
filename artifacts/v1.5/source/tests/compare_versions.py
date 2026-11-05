"""Explicit real-model comparison through the unmodified official SDK.

Archived versions and current source receive the same inputs and process-level
configuration. Requests run sequentially in rotating version order, on one CPU
with a 1 GiB address-space limit. This module records measurements, never an
estimated official score. Importing it or using --dry-run makes no model call.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

from src import __version__
from src.config import load_settings, read_env_file
from src.replay import replay
from src.solver import validate_task
from src.topology import Topology


PROJECT = Path(__file__).resolve().parents[1]
TIMEOUT_MS = 180_000
CONFIG_KEYS = ("PLATFORM_LLM_URL", "PLATFORM_API_KEY", "PLATFORM_LLM_MODEL",
               "AGENTCC_LLM_TIMEOUT_S")


@dataclass(frozen=True)
class Version:
    label: str
    source: Path
    current: bool


@dataclass(frozen=True)
class Sample:
    label: str
    task: dict
    topology: dict

    def request(self):
        return {"task_id": self.task["task_id"], "task_type": "agentcc",
                "input": {"instance_id": self.task["instance_id"],
                          "task": self.task, "topology": self.topology},
                "constraints": {"timeout_ms": TIMEOUT_MS}}


def within_project(path: str | Path, project: Path = PROJECT) -> Path:
    root = project.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("benchmark paths must stay inside the project")
    return resolved


def resolve_versions(labels, *, project=PROJECT, current_version=__version__):
    """Bind version names to explicit archived or current source directories."""
    versions, seen = [], set()
    for requested in labels:
        label = current_version if requested == "current" else requested.removeprefix("v")
        if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", label):
            raise ValueError(f"invalid version label: {requested!r}")
        if label in seen:
            raise ValueError(f"duplicate version: {label}")
        seen.add(label)
        current = label == current_version
        relative = "src" if current else f"artifacts/v{label}/source/src"
        source = within_project(relative, project)
        if not all((source / name).is_file() for name in ("__init__.py", "my_agent.py")):
            raise ValueError(f"version {label} source is missing: {relative}")
        versions.append(Version(label, source, current))
    if not versions:
        raise ValueError("select at least one version")
    return versions


def read_object(path: Path) -> dict:
    def reject_constant(_value):
        raise ValueError("non-finite JSON number")

    try:
        value = json.loads(path.read_text(), parse_constant=reject_constant)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read valid JSON from {path.name}: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def load_samples(labels=None, *, all_public=False, project=PROJECT):
    public = within_project("materials/spec/examples/256-rank", project)
    if all_public:
        labels = [str(path.parent.relative_to(public))
                  for path in sorted(public.glob("*/*/task.json"))]
    labels = labels if labels is not None else ["mesh2d/ag", "mesh2d/ar"]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("select one or more distinct samples")
    samples = []
    for label in labels:
        if not re.fullmatch(r"[A-Za-z0-9_-]+/(ag|ar)", label):
            raise ValueError(f"invalid sample label: {label!r}; expected family/ag or family/ar")
        folder = within_project(public / label, project)
        if not all((folder / name).is_file() for name in ("task.json", "topology.json")):
            raise ValueError(f"unknown or incomplete public sample: {label}")
        task, topology = read_object(folder / "task.json"), read_object(folder / "topology.json")
        if not all(isinstance(task.get(key), str) and task[key] for key in ("task_id", "instance_id")):
            raise ValueError(f"sample {label} needs nonempty task_id and instance_id")
        expected_operator = "all-gather" if label.endswith("/ag") else "all-reduce"
        if task.get("operator") != expected_operator:
            raise ValueError(f"sample {label} operator does not match its label")
        try:
            validate_task(task, Topology.from_data(topology))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"sample {label} failed task/topology validation: {type(exc).__name__}") from None
        samples.append(Sample(label, task, topology))
    return samples


def check_output(path, *, project=PROJECT):
    output = within_project(path, project)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty; completed experiments are not overwritten")
    return output


def rotating_order(versions, sample_index, repeat_index):
    offset = (sample_index + repeat_index) % len(versions)
    return versions[offset:] + versions[:offset]


def source_hashes(versions, *, project=PROJECT):
    hashes = {}
    for version in versions:
        files = sorted(version.source.rglob("*.py"))
        dockerfile = version.source.parent / "Dockerfile"
        if dockerfile.is_file():
            files.append(dockerfile)
        hashes[version.label] = {
            str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }
    return hashes


def load_agent(version):
    if version.current:
        return importlib.import_module("src.my_agent").MyAgent()
    name = "agentcc_compare_v" + version.label.replace(".", "_")
    specification = importlib.util.spec_from_file_location(
        name, version.source / "__init__.py", submodule_search_locations=[str(version.source)])
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return importlib.import_module(f"{name}.my_agent").MyAgent()


def configure_environment(project=PROJECT):
    """Share existing configuration through this process, not archive files."""
    for name, value in read_env_file(project / ".env").items():
        if name in CONFIG_KEYS:
            os.environ.setdefault(name, value)
    load_settings(env_path=project / ".env")  # Validate before making any request.


def apply_limits():
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(allowed)})
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    memory = 1024**3 if hard == resource.RLIM_INFINITY else min(1024**3, hard)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    return {"cpu_affinity": sorted(os.sched_getaffinity(0)), "address_space_limit_bytes": memory,
            "python": sys.version.split()[0],
            "scope": "Process affinity and address space; not Docker CPU/memory/pids cgroups"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def response_measurements(sample, data, http_status):
    """Independent Replay happens after the measured HTTP/JSON interval."""
    if not isinstance(data, dict):
        return {"valid": False, "error": "SDK response was not an object"}
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    record = {"status": data.get("status"), "calls": metadata.get("llm_calls"),
              "tokens": metadata.get("total_tokens"), "valid": False}
    if http_status != 200 or data.get("status") != "success":
        record["error"] = "SDK request did not succeed; see saved response"
        return record
    try:
        checked = replay(sample.task, sample.topology, data)
        diagnostics = data["output"]["diagnostics"]
        search = diagnostics.get("search", {})
        edit = search.get("model_edit", {})
        record.update(valid=checked["valid"], time_us=checked["total_time_us"],
                      rounds=checked["nonempty_rounds"], split=checked["split"],
                      llm_s=diagnostics.get("llm", {}).get("duration_ms", 0) / 1000,
                      model_selected=edit.get("selected"), model_accepted=edit.get("accepted"),
                      phases_s=diagnostics.get("phases_s", {}),
                      diagnostic_version=diagnostics.get("version"),
                      selected_candidates=[candidate for candidate in search.get("candidates", [])
                                           if candidate.get("selected")])
    except (ValueError, TypeError, KeyError) as exc:
        record.update(valid=False, error=f"response validation failed: {type(exc).__name__}")
    return record


def update_summary(report):
    rows = report["rows"]
    report["summary"] = {"completed": len(rows), "expected": report["expected_requests"],
                         "valid": sum(row.get("valid", False) for row in rows),
                         "one_call_per_request": all(row.get("calls") == 1 for row in rows),
                         "failed": sum(not row.get("valid", False) for row in rows)}


async def run(versions, samples, repeat, output, limits):
    import httpx
    from agenthack import AgentRuntime

    before = source_hashes(versions)
    report = {"schema_version": "agentcc.version-comparison.v1", "score_estimates": False,
              "purpose": "Raw same-input official-SDK/model measurements in rotating version order",
              "versions": [{"version": version.label, "current": version.current,
                            "source": str(version.source.relative_to(PROJECT))} for version in versions],
              "samples": [sample.label for sample in samples], "repeat": repeat,
              "expected_requests": len(versions) * len(samples) * repeat,
              "limits": limits, "timeout_ms": TIMEOUT_MS,
              "timing": "request_s includes HTTP request and complete response JSON parsing; excludes independent Replay and disk writes",
              "source_sha256_at_start": before, "rows": []}
    output.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        write_json(output / "inputs" / (sample.label.replace("/", "-") + ".json"), sample.request())
    update_summary(report)
    write_json(output / "summary.json", report)
    credential = os.environ.get("PLATFORM_API_KEY", "").strip()
    # No authentication header or configuration value is recorded. Also reject
    # and redact an unexpected credential echo before writing response evidence.
    async with AsyncExitStack() as stack:
        clients = {}
        for version in versions:
            runtime = AgentRuntime(load_agent(version))
            await stack.enter_async_context(runtime.app.router.lifespan_context(runtime.app))
            client = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=runtime.app), base_url="http://official-sdk-test"))
            if (await client.get("/health")).status_code != 200:
                raise RuntimeError(f"official SDK health check failed for version {version.label}")
            clients[version.label] = client
        for repeat_index in range(repeat):
            for sample_index, sample in enumerate(samples):
                order = rotating_order(versions, sample_index, repeat_index)
                for position, version in enumerate(order):
                    gc.collect()
                    record = {"sample": sample.label, "repeat": repeat_index + 1,
                              "version": version.label, "order": [item.label for item in order],
                              "position": position + 1}
                    stem = f"{sample.label.replace('/', '-')}-r{repeat_index + 1:02d}"
                    destination = output / f"v{version.label}" / f"{stem}.json"
                    started, cpu_started = time.monotonic(), time.process_time()
                    try:
                        response = await clients[version.label].post("/solve", json=sample.request())
                        data = response.json()
                        elapsed = time.monotonic() - started
                        record.update(request_s=elapsed, request_cpu_s=time.process_time() - cpu_started,
                                      http_status=response.status_code,
                                      response_bytes=len(response.content))
                        leaked = bool(credential and credential.encode() in response.content)
                        if leaked:
                            sanitized = response.text.replace(credential, "[REDACTED]")
                            data = json.loads(sanitized)
                            record.update(valid=False, error="credential echo redacted from response",
                                          credential_redacted=True)
                        else:
                            record.update(response_measurements(sample, data, response.status_code))
                        write_json(destination, data)
                    except Exception as exc:
                        # Exception strings from HTTP clients can contain URLs;
                        # only the class is recorded, never arbitrary credentials.
                        record.update(valid=False, request_s=time.monotonic() - started,
                                      request_cpu_s=time.process_time() - cpu_started,
                                      error=f"benchmark request failed: {type(exc).__name__}")
                        write_json(destination, {"benchmark_error": record["error"]})
                    record["response_file"] = str(destination.relative_to(output))
                    record["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    report["rows"].append(record)
                    update_summary(report)
                    write_json(output / "summary.json", report)
                    print(json.dumps({key: record.get(key) for key in (
                        "sample", "repeat", "version", "valid", "request_s", "time_us", "calls",
                        "tokens", "model_selected", "error")}), flush=True)
    report["source_sha256_at_end"] = source_hashes(versions)
    report["source_files_changed_during_run"] = before != report["source_sha256_at_end"]
    report["complete"] = len(report["rows"]) == report["expected_requests"]
    report["valid"] = (report["complete"] and report["summary"]["failed"] == 0
                       and report["summary"]["one_call_per_request"]
                       and not report["source_files_changed_during_run"])
    write_json(output / "summary.json", report)
    return 0 if report["valid"] else 1


def positive_integer(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", default=["1.3", "current"],
                        help="version numbers or current; default: 1.3 current")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--samples", nargs="+", help="family/ag and family/ar labels; default: mesh2d/ag mesh2d/ar")
    selection.add_argument("--all-public", action="store_true", help="all 14 public 256-rank inputs")
    parser.add_argument("--repeat", type=positive_integer, default=1)
    parser.add_argument("--output-dir", default=f"artifacts/v{__version__}/runs/comparison")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan without configuration or network access")
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        versions = resolve_versions(args.versions)
        samples = load_samples(args.samples, all_public=args.all_public)
        output = check_output(args.output_dir)
        if args.dry_run:
            print(json.dumps({"versions": [version.label for version in versions],
                              "samples": [sample.label for sample in samples],
                              "expected_model_calls": len(versions) * len(samples) * args.repeat,
                              "output": str(output.relative_to(PROJECT)), "network_calls": 0}, indent=2))
            return 0
        configure_environment()
    except ValueError as exc:
        parser.error(str(exc))
    limits = apply_limits()
    return asyncio.run(run(versions, samples, args.repeat, output, limits))


if __name__ == "__main__":
    raise SystemExit(main())
