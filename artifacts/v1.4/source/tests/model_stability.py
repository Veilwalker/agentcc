"""Repeat frozen real-model prompts; default mode never reads credentials or calls out.

This is a prompt/transport experiment, not an SDK solve benchmark. Protocol
validation checks only base/counts; it cannot establish schedule feasibility.
Run explicitly with ``python -m tests.model_stability --run``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from src.config import ConfigurationError, load_settings
from src.llm import LLMError, _chat_completion, _parse_edit_json


PROJECT = Path(__file__).resolve().parents[1]
TEMPERATURES = (0.0, 0.2)
DEFAULT_RESPONSES = (
    "artifacts/v1.4/runs/pilot-01/v1.4/cloudmatrix-ag-r01.json",
    "artifacts/v1.4/runs/pilot-01/v1.4/nvlink_like-ar-r01.json",
)
DEFAULT_OUTPUT = "artifacts/v1.4/runs/model-stability"


@dataclass(frozen=True)
class Prompt:
    label: str
    path: Path
    file_sha256: str
    payload: dict
    base: str
    maxima: tuple[int, ...]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def local_path(value, project=PROJECT):
    root = project.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("experiment paths must stay inside the project")
    return path


def load_prompts(paths, *, project=PROJECT):
    """Freeze the exact two prompt strings and recorded model/token limit."""
    prompts, seen = [], set()
    required = {"model", "messages", "temperature", "max_tokens", "stream"}
    for index, value in enumerate(paths):
        path = local_path(value, project)
        if path in seen:
            raise ValueError("duplicate response input")
        seen.add(path)
        try:
            body = path.read_bytes()
            saved = json.loads(body)
            llm = saved["output"]["diagnostics"]["llm"]
            payload = llm["request_evidence"]
            if saved.get("status") != "success" or llm.get("called") is not True:
                raise ValueError("expected a successful real-model SDK response")
            if saved.get("metadata", {}).get("llm_calls") != 1:
                raise ValueError("expected one recorded SDK model call")
            if not isinstance(payload, dict) or set(payload) != required:
                raise ValueError("unsupported recorded request fields")
            if not isinstance(payload["model"], str) or not payload["model"]:
                raise ValueError("missing recorded model")
            if type(payload["max_tokens"]) is not int or not 1 <= payload["max_tokens"] <= 16384:
                raise ValueError("invalid recorded output-token limit")
            if payload["stream"] is not False:
                raise ValueError("streaming requests are unsupported")
            messages = payload["messages"]
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError("expected exactly system and user messages")
            for message, role in zip(messages, ("system", "user")):
                if (not isinstance(message, dict) or set(message) != {"role", "content"}
                        or message["role"] != role or not isinstance(message["content"], str)):
                    raise ValueError("unsupported recorded message")
            expected_sha = digest(json.dumps(payload, ensure_ascii=False).encode())
            if llm.get("request_sha256") not in (None, expected_sha):
                raise ValueError("recorded request hash does not match its payload")
            problem = _parse_edit_json(messages[1]["content"])["problem"]
            base, groups = problem["base"], problem["groups"]
            if not isinstance(base, str) or not base or not isinstance(groups, list):
                raise ValueError("missing base or editable groups")
            maxima = []
            for group in groups:
                maximum = (group.get("max_chunks") if isinstance(group, dict)
                           else group[1] if isinstance(group, list) and len(group) >= 2 else None)
                if type(maximum) is not int or maximum < 0:
                    raise ValueError("expected v2/v3 groups with integer count bounds")
                maxima.append(maximum)
        except (OSError, KeyError, IndexError, TypeError, ValueError, RecursionError) as error:
            # Do not echo arbitrary source-file contents or JSON fragments.
            raise ValueError(f"invalid saved response {path.name}: {type(error).__name__}") from None
        label = re.sub(r"[^a-zA-Z0-9_-]", "_", re.sub(r"-r\d+$", "", path.stem))
        prompts.append(Prompt(f"prompt-{index + 1:02d}-{label}", path,
                              digest(body), payload, base, tuple(maxima)))
    if not prompts:
        raise ValueError("select at least one saved response")
    return prompts


def validate_reply(content, prompt, *, finish_reason="stop"):
    """Check the wire protocol only; no route, resource or Replay validation."""
    if finish_reason == "length":
        raise ValueError("model response was truncated")
    proposal = _parse_edit_json(content)
    if set(proposal) != {"base", "counts"} or proposal["base"] != prompt.base:
        raise ValueError("reply must contain the exact base and counts fields")
    counts = proposal["counts"]
    if not isinstance(counts, list) or len(counts) != len(prompt.maxima):
        raise ValueError("reply needs exactly one count per group")
    if any(type(count) is not int or not 0 <= count <= maximum
           for count, maximum in zip(counts, prompt.maxima)):
        raise ValueError("reply counts must be integers within the offered bounds")
    return {"base": prompt.base, "counts": counts}


def request_order(prompts, repeat):
    """Alternate temperature order per prompt/repeat; never retry a request."""
    for repetition in range(1, repeat + 1):
        for index, prompt in enumerate(prompts):
            temperatures = TEMPERATURES if (repetition + index) % 2 else TEMPERATURES[::-1]
            for position, temperature in enumerate(temperatures, 1):
                yield prompt, repetition, position, temperature


def distribution(values):
    return ({"count": len(values), "min": min(values), "median": statistics.median(values),
             "max": max(values)} if values else {"count": 0, "min": None, "median": None, "max": None})


def summarize(rows):
    groups = []
    for label in dict.fromkeys(row["prompt"] for row in rows):
        for temperature in TEMPERATURES:
            subset = [row for row in rows if row["prompt"] == label and row["temperature"] == temperature]
            valid = [row for row in subset if row["protocol_valid"]]
            variants = Counter(json.dumps(row["canonical_reply"], sort_keys=True, separators=(",", ":"))
                               for row in valid)
            tokens = [row["usage"]["total_tokens"] for row in subset if "total_tokens" in row["usage"]]
            group = {"prompt": label, "temperature": temperature, "attempted": len(subset),
                     "responses": sum(row["response_received"] for row in subset),
                     "protocol_valid": len(valid), "canonical_answer_count": len(variants),
                     "raw_answer_count": len({row["response_sha256"] for row in subset
                                               if row["response_received"]}),
                     "variants": [{"reply": json.loads(reply), "occurrences": count}
                                  for reply, count in sorted(variants.items())],
                     "request_s": distribution([row["request_s"] for row in subset]),
                     "tokens": distribution(tokens), "known_total_tokens": sum(tokens),
                     "token_observations": len(tokens),
                     "count_values_per_group": [sorted({row["canonical_reply"]["counts"][index]
                                                         for row in valid})
                                                for index in range(len(valid[0]["canonical_reply"]["counts"]))]
                                                if valid else [],
                     "variation_observed": len(variants) > 1}
            groups.append(group)
    return groups


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def redact_credential(value, credential):
    """Remove literal and JSON-escaped credentials before storing evidence."""
    if isinstance(value, str):
        safe = value
        if credential:
            forms = {credential, json.dumps(credential, ensure_ascii=False)[1:-1],
                     json.dumps(credential, ensure_ascii=True)[1:-1]}
            # The assistant text can itself contain a JSON string. Redacting
            # metadata recursively alone would miss its escaped spelling.
            for form in sorted(forms, key=len, reverse=True):
                safe = safe.replace(form, "[REDACTED]")
        return safe, safe != value
    if isinstance(value, list):
        children = [redact_credential(item, credential) for item in value]
        return [item for item, _ in children], any(changed for _, changed in children)
    if isinstance(value, dict):
        children = [(redact_credential(key, credential), redact_credential(item, credential))
                    for key, item in value.items()]
        return {key[0]: item[0] for key, item in children}, any(key[1] or item[1] for key, item in children)
    return value, False


def check_output(value, *, project=PROJECT):
    path = local_path(value, project)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("output directory must be new or empty; prior experiments are not overwritten")
    return path


def run(prompts, repeat, output, settings, *, completion=None):
    completion = completion or _chat_completion
    # Keep every recorded request field except temperature fixed. Authentication
    # and the authorized endpoint come from the normal configuration interface.
    for prompt in prompts:
        if prompt.payload["model"] != settings.model:
            raise ValueError("configured model differs from a recorded request; cannot isolate temperature")
        if redact_credential(prompt.payload, settings.api_key)[1]:
            raise ValueError("unexpected credential in recorded prompt; experiment refused")
    sources = [{"prompt": p.label, "response_file": str(p.path.relative_to(PROJECT)),
                "source_file_sha256": p.file_sha256, "base": p.base, "count_bounds": list(p.maxima)}
               for p in prompts]
    report = {"schema_version": "agentcc.model-stability.v1", "scope": "Fixed-prompt direct LLM experiment",
              "sdk_benchmark": False, "replay_performed": False, "score_estimates": False,
              "validation_scope": "base/counts JSON and integer bounds only; not schedule feasibility",
              "notes": ["Temperature zero does not guarantee identical backend responses.",
                        "Three repeats cannot establish P95 latency or statistical significance.",
                        "Raw response means the complete assistant text returned by _chat_completion; the HTTP JSON envelope is not exposed by that interface.",
                        "Failed attempts remain in latency counts; missing token usage is unknown, never zero.",
                        "No retries, SDK solves, production sampling changes or schedule scoring are performed."],
              "temperatures": list(TEMPERATURES), "repeat": repeat, "expected_requests": len(prompts) * repeat * 2,
              "inputs": sources, "rows": [], "groups": [], "complete": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "responses").mkdir()
    write_json(output / "summary.json", report)
    for prompt, repetition, position, temperature in request_order(prompts, repeat):
        expected = {**prompt.payload, "temperature": temperature}
        filename = f"{prompt.label}-t{temperature:g}-r{repetition:02d}.json"
        row = {"prompt": prompt.label, "repeat": repetition, "position": position, "temperature": temperature,
               "response_received": False, "protocol_valid": False, "usage": {},
               "canonical_reply": None, "response_sha256": None, "evidence_file": "responses/" + filename}
        evidence = {"request_payload": expected, "raw_response_scope": "complete assistant message text",
                    "raw_response": None, "metadata": None, "credential_redacted": False}
        started = time.monotonic()
        try:
            content, metadata = completion(expected["messages"][0]["content"],
                                           expected["messages"][1]["content"],
                                           replace(settings, max_tokens=expected["max_tokens"]),
                                           allow_truncated=True, temperature=temperature)
            row["request_s"] = time.monotonic() - started
            sanitized, redacted = redact_credential({"content": content, "metadata": metadata}, settings.api_key)
            content, metadata = sanitized["content"], sanitized["metadata"]
            evidence.update(raw_response=content, metadata=metadata, credential_redacted=redacted)
            row.update(response_received=True, usage=metadata.get("usage", {}),
                       response_sha256=digest(content.encode()),
                       transport_duration_ms=metadata.get("duration_ms"),
                       payload_matches_recorded=metadata.get("request_evidence") == expected)
            if redacted:
                row["error"] = "unexpected credential in model evidence; redacted and rejected"
            elif not row["payload_matches_recorded"]:
                row["error"] = "actual request differs from the frozen payload"
            else:
                try:
                    row["canonical_reply"] = validate_reply(content, prompt, finish_reason=metadata.get("finish_reason"))
                    row["protocol_valid"] = True
                except ValueError as error:
                    row["error"] = str(error)
        except LLMError as error:
            row["request_s"] = time.monotonic() - started
            row["error"] = str(error).replace(settings.api_key, "[REDACTED]") if settings.api_key else str(error)
        except Exception as error:
            row["request_s"] = time.monotonic() - started
            row["error"] = f"unexpected request failure: {type(error).__name__}"
        write_json(output / "responses" / filename, {**row, **evidence})
        report["rows"].append(row)
        report["groups"] = summarize(report["rows"])
        write_json(output / "summary.json", report)
        print(json.dumps({key: row[key] for key in ("prompt", "repeat", "temperature", "protocol_valid", "request_s")}), flush=True)
    report["complete"] = len(report["rows"]) == report["expected_requests"]
    report["source_files_unchanged"] = all(p.path.is_file() and digest(p.path.read_bytes()) == p.file_sha256 for p in prompts)
    report["protocol_valid_requests"] = sum(row["protocol_valid"] for row in report["rows"])
    report["known_total_tokens"] = sum(row["usage"].get("total_tokens", 0) for row in report["rows"])
    report["token_observations"] = sum("total_tokens" in row["usage"] for row in report["rows"])
    write_json(output / "summary.json", report)
    return report


def positive_integer(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("repeat must be a positive integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("repeat must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", nargs="+", default=DEFAULT_RESPONSES,
                        help="Saved real SDK responses containing llm.request_evidence")
    parser.add_argument("--repeat", type=positive_integer, default=3)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Explicitly authorize this real-model experiment")
    mode.add_argument("--dry-run", action="store_true", help="Validate saved inputs only (the default)")
    args = parser.parse_args(argv)
    try:
        prompts = load_prompts(args.responses)
        output = check_output(args.output_dir)
        if not args.run:
            print(json.dumps({"mode": "dry-run", "expected_requests": len(prompts) * args.repeat * 2,
                              "temperatures": TEMPERATURES, "repeat": args.repeat,
                              "prompts": [{"id": p.label, "groups": len(p.maxima), "source_file_sha256": p.file_sha256}
                                          for p in prompts], "output": str(output.relative_to(PROJECT)),
                              "credentials_loaded": False, "network_calls": 0}, ensure_ascii=False, indent=2))
            return 0
        settings = load_settings(env_path=PROJECT / ".env")
        report = run(prompts, args.repeat, output, settings)
    except (ValueError, ConfigurationError) as error:
        parser.error(str(error))
    return 0 if report["source_files_unchanged"] and report["protocol_valid_requests"] == report["expected_requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
