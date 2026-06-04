"""Experimental live endpoint collection helpers for adapters.

This module is intentionally adapter-side only.  It lets an adapter validate a
small ``parameters["live_collection"]`` contract, collect model responses during
``JobPhase.LOADING_DATA``, and write a normalized JSONL dataset for its existing
benchmark loader.

Trust boundary: this helper assumes the adapter operator controls the endpoint
configuration. Hosted runtimes that accept this config from untrusted users
should add their own egress policy or endpoint allowlist before enabling it.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


class LiveEndpointConfig(BaseModel):
    """Configuration for a live endpoint used during data collection."""

    type: Literal["openai_chat_completions"] = Field(
        default="openai_chat_completions",
        description="Endpoint protocol supported by the collector.",
    )
    base_url: str = Field(
        ..., description="Base URL for an OpenAI-compatible API, excluding the route."
    )
    model: str = Field(..., description="Model name sent to the endpoint.")
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable containing the endpoint bearer token.",
    )
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must use http or https")
        if not parsed.netloc:
            raise ValueError("base_url must include a host")
        return value.rstrip("/")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model cannot be empty")
        return cleaned

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_env(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("api_key_env cannot be empty")
        return cleaned

    @model_validator(mode="after")
    def require_https_for_api_keys(self) -> LiveEndpointConfig:
        if self.api_key_env and urlparse(self.base_url).scheme != "https":
            raise ValueError("api_key_env requires an https base_url")
        return self

    @property
    def chat_completions_url(self) -> str:
        """Return the OpenAI-compatible chat completions route."""
        return f"{self.base_url}/chat/completions"

    @property
    def api_key(self) -> str | None:
        """Read the bearer token from the configured environment variable."""
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


class LiveCollectionConfig(BaseModel):
    """Experimental adapter parameter contract for live response collection."""

    input_path: Path = Field(..., description="CSV or JSONL file with questions.")
    output_path: Path = Field(..., description="JSONL file to write collected rows.")
    question_field: str = Field(default="question")
    id_field: str | None = Field(default=None)
    endpoint: LiveEndpointConfig

    @field_validator("question_field")
    @classmethod
    def validate_question_field(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question_field cannot be empty")
        return cleaned

    @field_validator("id_field")
    @classmethod
    def validate_id_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("id_field cannot be empty")
        return cleaned

    @property
    def manifest_path(self) -> Path:
        """Return the sidecar manifest path for ``output_path``."""
        return self.output_path.with_suffix(f"{self.output_path.suffix}.manifest.json")


class LiveCollectionSummary(BaseModel):
    """Summary returned after a collection run."""

    rows_total: int
    rows_succeeded: int
    rows_failed: int
    output_path: Path
    manifest_path: Path


def load_live_collection_config(parameters: dict[str, Any]) -> LiveCollectionConfig:
    """Load ``parameters["live_collection"]`` into a typed config model."""
    raw_config = parameters.get("live_collection")
    if raw_config is None:
        raise ValueError('Missing parameters["live_collection"]')
    if not isinstance(raw_config, dict):
        raise ValueError('parameters["live_collection"] must be an object')
    return LiveCollectionConfig(**raw_config)


def run_live_collection(config: LiveCollectionConfig) -> LiveCollectionSummary:
    """Collect live endpoint responses and write output JSONL plus a manifest."""
    rows = load_input_rows(config.input_path)
    collected = list(collect_live_responses(rows, config))
    write_jsonl(config.output_path, collected)

    rows_failed = sum(1 for row in collected if row["error"] is not None)
    summary = LiveCollectionSummary(
        rows_total=len(collected),
        rows_succeeded=len(collected) - rows_failed,
        rows_failed=rows_failed,
        output_path=config.output_path,
        manifest_path=config.manifest_path,
    )
    write_manifest(config.manifest_path, config, summary)
    return summary


def load_input_rows(path: Path) -> list[dict[str, Any]]:
    """Load CSV or JSONL input rows."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} is not an object")
                rows.append(value)
        return rows
    raise ValueError(f"Unsupported input format: {path.suffix}")


def collect_live_responses(
    rows: Iterable[dict[str, Any]],
    config: LiveCollectionConfig,
) -> Iterable[dict[str, Any]]:
    """Yield source rows with collected response fields attached."""
    for index, row in enumerate(rows):
        output = dict(row)
        output["response"] = None
        output["response_metadata"] = {"source_index": index}
        if config.id_field and config.id_field in row:
            output["response_metadata"]["source_id"] = row[config.id_field]
        output["error"] = None

        question = str(row.get(config.question_field, "")).strip()
        if not question:
            output["error"] = {
                "type": "missing_question",
                "message": f"Missing or empty question field: {config.question_field}",
            }
            yield output
            continue

        try:
            response = call_openai_chat_completion(config.endpoint, question)
        except Exception as exc:  # noqa: BLE001 - preserve per-row failures.
            output["error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
        else:
            output["response"] = response["text"]
            output["response_metadata"].update(response["metadata"])
            if response["text"] is None:
                output["error"] = {
                    "type": "empty_response_content",
                    "message": "Endpoint returned no message content",
                }
        yield output


def call_openai_chat_completion(
    endpoint: LiveEndpointConfig,
    question: str,
) -> dict[str, Any]:
    """Call one OpenAI-compatible chat completions endpoint."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "live endpoint collection requires httpx; install eval-hub-sdk[core]"
        ) from exc

    payload = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": question}],
    }
    headers = {"content-type": "application/json"}
    if endpoint.api_key:
        headers["authorization"] = f"Bearer {endpoint.api_key}"

    last_error: Exception | None = None
    for attempt in range(endpoint.max_retries + 1):
        try:
            response = httpx.post(
                endpoint.chat_completions_url,
                json=payload,
                headers=headers,
                timeout=endpoint.timeout_seconds,
                follow_redirects=False,
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            return {
                "text": message.get("content"),
                "metadata": {
                    "status_code": response.status_code,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage"),
                },
            }
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code < 500 or attempt >= endpoint.max_retries:
                raise
        except (httpx.RequestError, TimeoutError) as exc:
            last_error = exc
            if attempt >= endpoint.max_retries:
                raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Unexpected chat completions response shape") from exc

        time.sleep(min(2**attempt, 8))

    if last_error is not None:
        raise last_error
    raise RuntimeError("chat completion failed without an exception")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically write JSONL rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def write_manifest(
    path: Path,
    config: LiveCollectionConfig,
    summary: LiveCollectionSummary,
) -> None:
    """Write a small manifest next to the collected dataset."""
    manifest = {
        "experimental": True,
        "endpoint_type": config.endpoint.type,
        "input_path": str(config.input_path),
        "output_path": str(summary.output_path),
        "question_field": config.question_field,
        "id_field": config.id_field,
        "rows_total": summary.rows_total,
        "rows_succeeded": summary.rows_succeeded,
        "rows_failed": summary.rows_failed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
