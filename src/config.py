"""Configuration for the official model service; secrets are never logged."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import shlex
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit


class ConfigurationError(ValueError):
    """A safe-to-display configuration error without secret values."""


def read_env_file(path: Path) -> dict[str, str]:
    """Read simple dotenv assignments without executing or expanding values."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for number, original in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ConfigurationError(f"Invalid .env assignment on line {number}")
        value = value.strip()
        if value.startswith(("'", '"')):
            try:
                parts = shlex.split(value, comments=True, posix=True)
            except ValueError:
                raise ConfigurationError(f"Invalid .env quoting on line {number}") from None
            if len(parts) != 1:
                raise ConfigurationError(f"Invalid .env value on line {number}")
            value = parts[0]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[key] = value
    return values


def completion_endpoint(value: str) -> str:
    """Accept a complete chat endpoint or a conventional OpenAI-compatible base."""
    try:
        parsed = urlsplit(value.strip())
        valid_host = parsed.hostname
        parsed.port  # Validate malformed ports without including the original URL.
    except ValueError:
        raise ConfigurationError("PLATFORM_LLM_URL is not a valid HTTP(S) URL") from None
    if parsed.scheme not in ("http", "https") or not valid_host:
        raise ConfigurationError("PLATFORM_LLM_URL must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ConfigurationError("PLATFORM_LLM_URL must not contain user info or a fragment")
    if any(character.isspace() for character in value):
        raise ConfigurationError("PLATFORM_LLM_URL must not contain whitespace")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1/chat/completions"
    elif path.endswith("/v1"):
        path += "/chat/completions"
    elif not path.endswith("/chat/completions"):
        raise ConfigurationError(
            "PLATFORM_LLM_URL must be a /v1 base or complete /chat/completions endpoint"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


@dataclass(frozen=True)
class Settings:
    endpoint: str = field(repr=False)
    api_key: str = field(repr=False)
    model: str
    timeout_s: float = 15.0
    max_tokens: int = 256


def load_settings(
    env_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Process environment overrides the project's optional .env file."""
    path = Path(env_path) if env_path is not None else Path(__file__).resolve().parents[1] / ".env"
    values = read_env_file(path)
    values.update(os.environ if environ is None else environ)
    for name in ("PLATFORM_LLM_URL", "PLATFORM_API_KEY", "PLATFORM_LLM_MODEL"):
        if not values.get(name, "").strip():
            raise ConfigurationError(f"Missing configuration: {name}")
    key = values["PLATFORM_API_KEY"].strip()
    model = values["PLATFORM_LLM_MODEL"].strip()
    if any(character in key for character in "\r\n"):
        raise ConfigurationError("PLATFORM_API_KEY must not contain line breaks")
    if len(model) > 256 or any(character in model for character in "\r\n"):
        raise ConfigurationError("PLATFORM_LLM_MODEL is invalid")
    try:
        timeout = float(values.get("AGENTCC_LLM_TIMEOUT_S", "15"))
    except ValueError:
        raise ConfigurationError("AGENTCC_LLM_TIMEOUT_S must be a number") from None
    if not math.isfinite(timeout) or not 1 <= timeout <= 60:
        raise ConfigurationError("AGENTCC_LLM_TIMEOUT_S must be between 1 and 60 seconds")
    return Settings(
        endpoint=completion_endpoint(values["PLATFORM_LLM_URL"]),
        api_key=key,
        model=model,
        timeout_s=timeout,
    )
