"""Compare complete SDK requests using unchanged saved model replies, offline.

The first v1.4 reply for each selected public input is replayed byte-for-byte.
Each version constructs its own prompt: a changed base or prompt is reported,
never repaired. Simulated waits expose overlap with the deterministic search.
SDK model spans are synthetic; actual network/model calls and token use are zero.
This measures local HTTP/JSON execution, not real model latency or official scores.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import resource
import socket
import time
from unittest.mock import patch

from tests import compare_versions as compare
from tests.model_stability import load_prompts


PROJECT = compare.PROJECT
DEFAULT_RECORDINGS = "artifacts/v1.4/runs/public-narrow"


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def payload_bytes(payload: dict) -> bytes:
    # Identical encoding to llm._chat_completion, including whitespace and Unicode.
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class Recording:
    sample: str
    response_path: Path
    input_path: Path
    response_file_sha256: str
    input_file_sha256: str
    payload: dict
    base: str
    raw: str
    finish_reason: str | None
    historical_usage: dict

    def reference(self):
        return {"sample": self.sample,
                "source_response_file": str(self.response_path.relative_to(PROJECT)),
                "source_response_file_sha256": self.response_file_sha256,
                "source_input_file": str(self.input_path.relative_to(PROJECT)),
                "source_input_file_sha256": self.input_file_sha256,
                "recorded_request_sha256": sha256(payload_bytes(self.payload)),
                "recorded_raw_sha256": sha256(self.raw.encode()),
                "recorded_base": self.base, "historical_usage_not_measured": self.historical_usage}


def load_recordings(folder, samples, *, project=PROJECT):
    """Select by report order, verify exact inputs, and retain the original reply."""
    folder = compare.within_project(folder, project)
    summary = compare.read_object(folder / "summary.json")
    if summary.get("source_files_changed_during_run"):
        raise ValueError("recording source changed during its original comparison")
    rows = summary.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("recording summary needs an array of row objects")
    first = {}
    for row in rows:
        if row.get("version") == "1.4":
            first.setdefault(row.get("sample"), row)
    recordings = {}
    for sample in samples:
        row = first.get(sample.label)
        if not row or not isinstance(row.get("response_file"), str):
            raise ValueError(f"no first v1.4 recording for {sample.label}")
        path = compare.within_project(folder / row["response_file"], project)
        if not path.is_relative_to(folder) or path.suffix != ".json":
            raise ValueError("recorded response must be a JSON file within its run directory")
        input_path = folder / "inputs" / (sample.label.replace("/", "-") + ".json")
        if compare.read_object(input_path) != sample.request():
            raise ValueError(f"public input differs from the saved input for {sample.label}")
        prompt = load_prompts([path], project=project)[0]
        data = compare.read_object(path)
        llm = data["output"]["diagnostics"]["llm"]
        evidence = llm.get("response_evidence", {})
        raw = evidence.get("content")
        if (not isinstance(raw, str) or evidence.get("credential_redacted") is not False
                or llm.get("synthetic")):
            raise ValueError("recording must contain an unredacted original real-model reply")
        if llm.get("response_sha256") != sha256(raw.encode()):
            raise ValueError("recorded response hash does not match its raw reply")
        temperature = prompt.payload["temperature"]
        if type(temperature) not in (int, float) or not math.isfinite(temperature):
            raise ValueError("recorded temperature must be finite")
        recordings[sample.label] = Recording(
            sample.label, path, input_path, prompt.file_sha256, sha256(input_path.read_bytes()),
            prompt.payload, prompt.base, raw, llm.get("finish_reason"), llm.get("usage", {}))
    return recordings


class OfflineGuard:
    """Fail closed on a transport or dotenv read, without exposing its arguments."""

    def __init__(self):
        self.network_attempts = 0
        self.configuration_attempts = 0

    def network(self, *_args, **_kwargs):
        self.network_attempts += 1
        raise RuntimeError("network access is forbidden in the fixed-reply comparison")

    def configuration(self, *_args, **_kwargs):
        self.configuration_attempts += 1
        raise RuntimeError("reading model configuration is forbidden in this comparison")


def recorded_completion(recording, wait_s, observations):
    """Replace transport only: leave prompt creation, parsing and validation intact."""
    def complete(system_prompt, user_content, settings, *, allow_truncated=False, temperature=0.2):
        started = time.monotonic()
        payload = {"model": settings.model, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}],
            "temperature": temperature, "max_tokens": settings.max_tokens, "stream": False}
        observations.append(payload)
        if wait_s:
            time.sleep(wait_s)
        return recording.raw, {
            "called": True, "synthetic": True, "mode": "unchanged_saved_reply",
            "actual_network_calls": 0, "actual_model_calls": 0, "actual_tokens": 0,
            "model": settings.model, "usage": {},
            "sampling": {"temperature": temperature, "guaranteed_deterministic": False},
            "duration_ms": (time.monotonic() - started) * 1000,
            "simulated_wait_s": wait_s, "request_evidence": payload,
            "input_sha256": sha256(user_content.encode()),
            "request_sha256": sha256(payload_bytes(payload)),
            "response_sha256": sha256(recording.raw.encode()),
            "finish_reason": recording.finish_reason,
        }
    return complete


def compare_prompt(recording, payload):
    try:
        base = json.loads(payload["messages"][1]["content"])["problem"]["base"]
    except (ValueError, TypeError, KeyError):
        base = None
    return {"prompt_matches_recording": payload["messages"] == recording.payload["messages"],
            "request_payload_matches_recording": payload == recording.payload,
            "model_matches_recording": payload["model"] == recording.payload["model"],
            "max_tokens_matches_recording": payload["max_tokens"] == recording.payload["max_tokens"],
            "base_matches_recording": base == recording.base,
            "recorded_base": recording.base, "current_base": base,
            "request_payload_sha256": sha256(payload_bytes(payload)),
            "recorded_request_payload_sha256": sha256(payload_bytes(recording.payload))}


def update_summary(report):
    rows = report["rows"]
    report["summary"] = {
        "completed": len(rows), "expected": report["expected_requests"],
        "replay_valid": sum(row.get("valid") is True for row in rows),
        "failed": sum(row.get("valid") is not True for row in rows),
        "identical_recorded_payloads": sum(row.get("request_payload_matches_recording") is True for row in rows),
        "one_synthetic_completion_per_request": all(row.get("synthetic_completion_count") == 1 for row in rows),
        "one_synthetic_sdk_span_per_request": all(row.get("sdk_synthetic_llm_calls") == 1 for row in rows),
        "all_raw_replies_unchanged": all(row.get("raw_reply_unchanged") is True for row in rows),
        "actual_network_calls": 0, "actual_model_calls": 0, "actual_tokens": 0,
    }


def version_modules(agent):
    package = type(agent).__module__.rsplit(".", 1)[0]
    modules = {name: importlib.import_module(f"{package}.{name}") for name in ("config", "llm")}
    adapter = importlib.import_module(type(agent).__module__)
    modules["service"] = importlib.import_module(adapter.solve_async.__module__)
    return modules


async def run(versions, samples, recordings, repeat, delays, output, limits):
    import httpx
    from agenthack import AgentRuntime
    from agenthack.telemetry import Telemetry
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    before = compare.source_hashes(versions)
    guard = OfflineGuard()
    report = {
        "schema_version": "agentcc.fixed-reply-sdk-comparison.v1", "synthetic": True,
        "score_estimates": False, "credentials_loaded": False,
        "actual_network_calls": 0, "actual_model_calls": 0, "actual_tokens": 0,
        "sdk_telemetry": "Model spans are synthetic, with zero token usage; no remote telemetry exporter",
        "purpose": "Same saved original v1.4 raw reply and simulated wait through each complete official SDK request",
        "recording_policy": "First v1.4 row per sample in the saved summary; never rebind base or change counts",
        "timing": "request_s and request_cpu_s include ASGI HTTP request and full response JSON parsing; exclude independent Replay, evidence hashing and writes",
        "versions": [{"version": version.label, "current": version.current,
                      "source": str(version.source.relative_to(PROJECT))} for version in versions],
        "samples": [sample.label for sample in samples], "repeat": repeat,
        "simulated_waits_s": delays, "limits": limits, "timeout_ms": compare.TIMEOUT_MS,
        "expected_requests": len(versions) * len(samples) * repeat * len(delays),
        "source_sha256_at_start": before,
        "tool_sha256_at_start": sha256(Path(__file__).read_bytes()),
        "recording_sources": [recordings[sample.label].reference() for sample in samples], "rows": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        compare.write_json(output / "inputs" / (sample.label.replace("/", "-") + ".json"), sample.request())
    update_summary(report)
    compare.write_json(output / "summary.json", report)
    async with AsyncExitStack() as stack:
        for name in ("connect", "connect_ex"):
            stack.enter_context(patch.object(socket.socket, name, side_effect=guard.network))
        for name in ("create_connection", "getaddrinfo"):
            stack.enter_context(patch.object(socket, name, side_effect=guard.network))
        clients, modules = {}, {}
        for version in versions:
            agent = compare.load_agent(version)
            modules[version.label] = version_modules(agent)
            config = modules[version.label]["config"]
            stack.enter_context(patch.object(config, "read_env_file", side_effect=guard.configuration))
            # Keep SDK traces and counters, but supply an exporter-free provider
            # so host telemetry environment variables cannot enable networking.
            provider = TracerProvider(resource=Resource({"service.name": "agentcc-fixed-reply-test"}))
            stack.callback(provider.shutdown)
            runtime = AgentRuntime(agent, telemetry=Telemetry(provider=provider))
            await stack.enter_async_context(runtime.app.router.lifespan_context(runtime.app))
            client = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=runtime.app), base_url="http://official-sdk-test"))
            if (await client.get("/health")).status_code != 200:
                raise RuntimeError(f"official SDK health check failed for version {version.label}")
            clients[version.label] = client
        for repeat_index in range(repeat):
            for sample_index, sample in enumerate(samples):
                recording = recordings[sample.label]
                # Rotate both version and wait order, while giving every version
                # the identical saved content and delay for a matched comparison.
                offset = (sample_index + repeat_index) % len(delays)
                for delay_index in list(range(len(delays)))[offset:] + list(range(len(delays)))[:offset]:
                    wait_s = delays[delay_index]
                    order = compare.rotating_order(versions, sample_index, repeat_index + delay_index)
                    delay_folder = f"delay-{delay_index + 1:02d}-{wait_s:g}s"
                    for position, version in enumerate(order):
                        observations = []
                        config, llm, service = (modules[version.label][key] for key in ("config", "llm", "service"))
                        settings = config.Settings("http://unused.invalid/v1/chat/completions", "",
                                                   recording.payload["model"],
                                                   max_tokens=recording.payload["max_tokens"])
                        row = {"sample": sample.label, "repeat": repeat_index + 1, "version": version.label,
                               "simulated_wait_s": wait_s, "delay_index": delay_index + 1,
                               "position": position + 1, "order": [item.label for item in order],
                               "actual_network_calls": 0, "actual_model_calls": 0, "actual_tokens": 0}
                        stem = f"{sample.label.replace('/', '-')}-r{repeat_index + 1:02d}.json"
                        destination = output / delay_folder / f"v{version.label}" / stem
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with ExitStack() as request_stack:
                            for module in (config, llm, service):
                                request_stack.enter_context(patch.object(module, "load_settings", return_value=settings))
                            request_stack.enter_context(patch.object(llm, "_chat_completion", side_effect=
                                recorded_completion(recording, wait_s, observations)))
                            request_stack.enter_context(patch.object(llm, "build_opener", side_effect=guard.network))
                            gc.collect()
                            started, cpu_started = time.monotonic(), time.process_time()
                            try:
                                response = await clients[version.label].post("/solve", json=sample.request())
                                data = response.json()
                                row.update(request_s=time.monotonic() - started,
                                           request_cpu_s=time.process_time() - cpu_started,
                                           http_status=response.status_code, response_bytes=len(response.content))
                                measured = compare.response_measurements(sample, data, response.status_code)
                                row.update(sdk_synthetic_llm_calls=measured.pop("calls", None),
                                           sdk_synthetic_tokens=measured.pop("tokens", None), **measured)
                                diagnostics = data.get("output", {}).get("diagnostics", {})
                                returned_llm = diagnostics.get("llm", {})
                                row.update(raw_reply_unchanged=(returned_llm.get("response_evidence", {}).get("content")
                                                               == recording.raw),
                                           returned_llm_is_synthetic=returned_llm.get("synthetic") is True,
                                           control_time_us=diagnostics.get("search", {}).get("model_edit", {}).get("control_time_us"),
                                           raw_reply_sha256=sha256(recording.raw.encode()))
                                # Preserve exact HTTP response bytes, including SDK telemetry.
                                destination.write_bytes(response.content)
                            except Exception as exc:
                                row.update(valid=False, request_s=time.monotonic() - started,
                                           request_cpu_s=time.process_time() - cpu_started,
                                           error=f"fixed-reply SDK request failed: {type(exc).__name__}")
                                compare.write_json(destination, {"benchmark_error": row["error"]})
                        row["synthetic_completion_count"] = len(observations)
                        if len(observations) == 1:
                            row.update(compare_prompt(recording, observations[0]))
                        row.update(response_file=str(destination.relative_to(output)),
                                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
                        report["rows"].append(row)
                        update_summary(report)
                        compare.write_json(output / "summary.json", report)
                        print(json.dumps({key: row.get(key) for key in (
                            "sample", "repeat", "version", "simulated_wait_s", "valid", "request_s",
                            "request_cpu_s", "time_us", "request_payload_matches_recording", "error")}), flush=True)
    report["source_sha256_at_end"] = compare.source_hashes(versions)
    report["source_files_changed_during_run"] = before != report["source_sha256_at_end"]
    report["tool_sha256_at_end"] = sha256(Path(__file__).read_bytes())
    report["recording_files_unchanged"] = all(
        sha256(recording.response_path.read_bytes()) == recording.response_file_sha256
        and sha256(recording.input_path.read_bytes()) == recording.input_file_sha256
        for recording in recordings.values())
    report.update(blocked_network_attempts=guard.network_attempts,
                  blocked_configuration_reads=guard.configuration_attempts,
                  complete=len(report["rows"]) == report["expected_requests"])
    summary = report["summary"]
    report["valid"] = bool(report["complete"] and not summary["failed"]
                           and summary["one_synthetic_completion_per_request"]
                           and summary["one_synthetic_sdk_span_per_request"] and summary["all_raw_replies_unchanged"]
                           and all(row.get("sdk_synthetic_tokens") == 0 and row.get("returned_llm_is_synthetic")
                                   for row in report["rows"])
                           and not report["source_files_changed_during_run"]
                           and report["tool_sha256_at_start"] == report["tool_sha256_at_end"]
                           and report["recording_files_unchanged"]
                           and not guard.network_attempts and not guard.configuration_attempts)
    # Valid schedules alone do not establish that this is a controlled model-
    # input comparison. Keep mismatched prompts and replies visible, never fix them.
    report["controlled_comparison"] = bool(report["valid"]
        and summary["identical_recorded_payloads"] == report["expected_requests"])
    compare.write_json(output / "summary.json", report)
    return report


def nonnegative_delay(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("delay must be a finite number from 0 to 60 seconds") from None
    if not math.isfinite(number) or not 0 <= number <= 60:
        raise argparse.ArgumentTypeError("delay must be a finite number from 0 to 60 seconds")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", default=["1.4", "current"])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--samples", nargs="+")
    selection.add_argument("--all-public", action="store_true")
    parser.add_argument("--repeat", type=compare.positive_integer, default=1)
    parser.add_argument("--delays", nargs="+", type=nonnegative_delay, default=[0.0, 2.0])
    parser.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS)
    parser.add_argument("--output-dir", default=f"artifacts/v{compare.__version__}/runs/fixed-version-comparison")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if len(set(args.delays)) != len(args.delays):
            raise ValueError("delays must be distinct")
        versions = compare.resolve_versions(args.versions)
        samples = compare.load_samples(args.samples, all_public=args.all_public)
        recordings = load_recordings(args.recordings_dir, samples)
        output = compare.check_output(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({"versions": [version.label for version in versions],
                          "samples": [sample.label for sample in samples], "delays_s": args.delays,
                          "expected_synthetic_sdk_requests": len(versions) * len(samples) * args.repeat * len(args.delays),
                          "actual_network_calls": 0, "actual_model_calls": 0, "credentials_loaded": False,
                          "output": str(output.relative_to(PROJECT))}, indent=2))
        return 0
    limits = compare.apply_limits()
    report = asyncio.run(run(versions, samples, recordings, args.repeat, args.delays, output, limits))
    return 0 if report["controlled_comparison"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
