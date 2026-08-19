"""Experimental live endpoint response collection for adapters.

Lets adapters collect chatbot/model responses during ``JobPhase.LOADING_DATA``
by querying a live endpoint with a set of test questions. Supports
OpenAI-compatible chat completions and generic HTTP endpoints (MCP, Langflow,
LangGraph, custom APIs).

Config is passed through ``JobSpec.parameters["live_collection"]`` while the
server-side API shape (``test_data_ref``) is still being validated.

Trust boundary: the adapter operator controls the endpoint configuration.
Hosted runtimes that accept config from untrusted users should add their own
egress policy or endpoint allowlist before enabling this.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from .auth import ModelCredentials, resolve_model_credentials

logger = logging.getLogger(__name__)

_PARAM_KEY = "live_collection"

_CA_BUNDLE_CANDIDATES = [
    Path("/etc/pki/ca-trust/source/anchors/service-ca.crt"),
    Path("/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"),
    Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"),
]


class CollectorProtocol(StrEnum):
    OPENAI_CHAT = "openai_chat_completions"
    GENERIC_HTTP = "generic_http"


class CollectorError(Exception):
    """Raised when collection fails in fail_fast mode."""

    def __init__(
        self,
        message: str,
        question_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.question_id = question_id
        self.__cause__ = cause
        super().__init__(message)


class CollectorConfig(BaseModel):
    """Configuration for live endpoint response collection."""

    questions_path: Path = Field(
        ..., description="CSV, JSON, or JSONL file with test questions"
    )
    output_dir: Path = Field(
        ..., description="Directory for responses.jsonl and manifest.json"
    )
    endpoint_url: str = Field(
        ..., description="Endpoint URL (OpenAI-compatible base or full HTTP URL)"
    )
    protocol: CollectorProtocol = Field(default=CollectorProtocol.OPENAI_CHAT)

    question_column: str = Field(default="question")
    id_column: str = Field(default="id")

    model: str | None = Field(
        default=None, description="Model name for OpenAI chat requests"
    )
    system_prompt: str | None = Field(default=None)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    request_template: dict[str, Any] | None = Field(
        default=None, description="Request body template for generic_http protocol"
    )
    response_path: str | None = Field(
        default=None,
        description="Dot-separated path to extract answer from response",
    )
    extra_response_paths: dict[str, str] | None = Field(
        default=None,
        description="Additional fields to extract from response (output_name -> dot-path)",
    )

    api_key_env: str | None = Field(
        default=None, description="Env var containing a bearer token"
    )
    request_headers: dict[str, str] = Field(default_factory=dict)
    use_model_credentials: bool = Field(
        default=True,
        description="Use resolve_model_credentials() for auth and TLS",
    )

    ca_bundle: str | None = Field(default=None, description="CA bundle path")
    insecure: bool = Field(default=False, description="Skip TLS verification")

    fail_fast: bool = Field(
        default=False, description="Raise on first collection error"
    )
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=0, ge=0)
    retry_backoff_seconds: float = Field(default=0.25, ge=0.0)
    max_retry_backoff_seconds: float = Field(default=5.0, ge=0.0)

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> CollectorConfig:
        raw = parameters.get(_PARAM_KEY)
        if raw is None:
            raise ValueError(f'JobSpec.parameters["{_PARAM_KEY}"] is required')
        if not isinstance(raw, dict):
            raise ValueError(f'parameters["{_PARAM_KEY}"] must be an object')
        return cls.model_validate(raw)

    @field_validator("endpoint_url")
    @classmethod
    def _validate_endpoint_url(cls, value: str) -> str:
        if not value.lower().startswith(("http://", "https://")):
            raise ValueError("endpoint_url must use http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _validate_protocol_requirements(self) -> Self:
        if self.protocol == CollectorProtocol.OPENAI_CHAT and not self.model:
            raise ValueError("model is required for openai_chat_completions protocol")
        if self.protocol == CollectorProtocol.GENERIC_HTTP:
            if not self.request_template:
                raise ValueError(
                    "request_template is required for generic_http protocol"
                )
            if not self.response_path:
                raise ValueError("response_path is required for generic_http protocol")
        return self


class LiveQuestion(BaseModel):
    question: str
    question_id: str | None = None
    source_row: dict[str, Any] = Field(default_factory=dict)


class CollectedRecord(BaseModel):
    """One collected response row.

    ``source_fields`` preserves all original input columns as-is.
    Collector-added fields (``response``, ``error``, etc.) are merged
    on top when serialized to JSONL.
    """

    source_fields: dict[str, Any] = Field(default_factory=dict)
    response: str | None = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float | None = None


class CollectionManifest(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    experimental: bool = True
    protocol: str
    endpoint_url: str
    model: str | None = None
    questions_path: str
    output_path: str
    total: int
    completed: int
    failed: int


def is_configured(parameters: dict[str, Any] | None) -> bool:
    """Check if parameters contain live collection config."""
    if not parameters:
        return False
    raw = parameters.get(_PARAM_KEY)
    return isinstance(raw, dict) and bool(raw)


def collect_responses(
    config: CollectorConfig,
    *,
    client: Any | None = None,
    credentials: ModelCredentials | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> CollectionManifest:
    """Collect responses from a live endpoint.

    Reads test questions, queries the endpoint, and writes an evaluation-ready
    JSONL dataset plus a manifest summary.
    """
    if credentials is None and config.use_model_credentials:
        try:
            credentials = resolve_model_credentials()
        except Exception:
            logger.debug("Could not resolve model credentials", exc_info=True)
            credentials = None

    headers = _resolve_auth_headers(config, credentials)
    verify = _resolve_verify(config, credentials)
    questions = load_questions(
        config.questions_path,
        question_column=config.question_column,
        id_column=config.id_column,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "responses.jsonl"

    active_client = client
    owns_client = active_client is None
    if active_client is None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Live collection requires httpx. "
                "Install eval-hub-sdk[adapter] or eval-hub-sdk[core]."
            ) from exc
        active_client = httpx.Client(
            follow_redirects=False,
            timeout=config.timeout_seconds,
            verify=verify,
        )

    completed = 0
    failed = 0
    total = len(questions)

    try:
        with output_path.open("w", encoding="utf-8") as out:
            for i, question in enumerate(questions):
                record = _collect_one(config, active_client, headers, question)
                if record.error:
                    failed += 1
                    if config.fail_fast:
                        raise CollectorError(
                            record.error,
                            question_id=question.question_id,
                        )
                else:
                    completed += 1
                out.write(json.dumps(_flatten_record(record)) + "\n")
                if progress_callback and total > 0:
                    progress_callback((i + 1) / total)
    finally:
        if owns_client and hasattr(active_client, "close"):
            active_client.close()

    manifest = CollectionManifest(
        protocol=config.protocol.value,
        endpoint_url=config.endpoint_url,
        model=config.model,
        questions_path=str(config.questions_path),
        output_path=str(output_path),
        total=total,
        completed=completed,
        failed=failed,
    )
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def collect_responses_from_parameters(
    parameters: Mapping[str, Any],
    **kwargs: Any,
) -> CollectionManifest:
    """Convenience wrapper: load config from parameters and collect."""
    config = CollectorConfig.from_parameters(parameters)
    return collect_responses(config, **kwargs)


def load_questions(
    path: Path | str,
    *,
    question_column: str = "question",
    id_column: str = "id",
) -> list[LiveQuestion]:
    """Load questions from CSV, JSON, or JSONL."""
    qpath = Path(path)
    if not qpath.exists():
        raise FileNotFoundError(f"Question file not found: {qpath}")

    suffix = qpath.suffix.lower()
    if suffix == ".csv":
        questions = _load_csv(qpath, question_column, id_column)
    elif suffix == ".jsonl":
        questions = _load_jsonl(qpath, question_column, id_column)
    elif suffix == ".json":
        questions = _load_json(qpath, question_column, id_column)
    else:
        raise ValueError(
            f"Unsupported question file type '{suffix}'. Use CSV, JSON, or JSONL."
        )

    if not questions:
        raise ValueError(f"No questions found in {qpath}")
    return questions


def extract_by_path(data: Any, path: str) -> Any:
    """Extract a value from nested data using a dot-separated path."""
    current = data
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        elif isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


def substitute_template(template: Any, variables: dict[str, str]) -> Any:
    """Recursively substitute ``{key}`` placeholders in string values."""
    if isinstance(template, str):
        try:
            return template.format_map(variables)
        except KeyError as exc:
            raise ValueError(
                f"Unknown placeholder {exc} in template. "
                f"Available: {', '.join(variables.keys())}"
            ) from exc
    if isinstance(template, dict):
        return {k: substitute_template(v, variables) for k, v in template.items()}
    if isinstance(template, list):
        return [substitute_template(item, variables) for item in template]
    return template


def _resolve_auth_headers(
    config: CollectorConfig,
    credentials: ModelCredentials | None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if credentials and credentials.api_key:
        headers["Authorization"] = f"Bearer {credentials.api_key}"
    if config.api_key_env:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {config.api_key_env!r} is required")
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(config.request_headers)
    return headers


def _resolve_verify(
    config: CollectorConfig,
    credentials: ModelCredentials | None,
) -> bool | str:
    if config.insecure:
        return False
    if config.ca_bundle:
        return config.ca_bundle
    if credentials and credentials.ca_cert_path:
        return str(credentials.ca_cert_path)
    for candidate in _CA_BUNDLE_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return True


def _collect_one(
    config: CollectorConfig,
    client: Any,
    headers: dict[str, str],
    question: LiveQuestion,
) -> CollectedRecord:
    if config.protocol == CollectorProtocol.OPENAI_CHAT:
        return _collect_openai(config, client, headers, question)
    return _collect_generic_http(config, client, headers, question)


def _collect_openai(
    config: CollectorConfig,
    client: Any,
    headers: dict[str, str],
    question: LiveQuestion,
) -> CollectedRecord:
    messages: list[dict[str, str]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": question.question})

    body: dict[str, Any] = dict(config.extra_body)
    body["model"] = config.model
    body["messages"] = messages

    url = f"{config.endpoint_url}/chat/completions"
    return _send_request(config, client, headers, question, url, body, _extract_openai)


def _collect_generic_http(
    config: CollectorConfig,
    client: Any,
    headers: dict[str, str],
    question: LiveQuestion,
) -> CollectedRecord:
    request_template = config.request_template
    response_path = config.response_path
    if not request_template or not response_path:
        raise ValueError(
            "request_template and response_path are required for generic_http"
        )

    variables = {
        "question": question.question,
        "question_id": question.question_id or "",
    }
    body = substitute_template(request_template, variables)
    return _send_request(
        config,
        client,
        headers,
        question,
        config.endpoint_url,
        body,
        lambda resp: _extract_generic(resp, response_path),
    )


def _send_request(
    config: CollectorConfig,
    client: Any,
    headers: dict[str, str],
    question: LiveQuestion,
    url: str,
    body: dict[str, Any],
    extractor: Callable[[dict[str, Any]], str | None],
) -> CollectedRecord:
    last_error: str | None = None
    for attempt in range(config.max_retries + 1):
        start = time.monotonic()
        try:
            response = client.post(
                url,
                json=body,
                headers=dict(headers),
                timeout=config.timeout_seconds,
            )
            latency_ms = (time.monotonic() - start) * 1000

            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int) and 300 <= status_code < 400:
                raise ValueError(f"Redirect not allowed: HTTP {status_code}")

            response.raise_for_status()
            raw_response = response.json()
            if not isinstance(raw_response, dict):
                raise ValueError("Response must be a JSON object")

            text = extractor(cast(dict[str, Any], raw_response))
            extra_fields = _extract_extra_fields(
                cast(dict[str, Any], raw_response), config.extra_response_paths
            )
            return CollectedRecord(
                source_fields=question.source_row,
                response=text,
                extra_fields=extra_fields,
                raw_response=cast(dict[str, Any], raw_response),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < config.max_retries and config.retry_backoff_seconds > 0:
                delay = min(
                    config.retry_backoff_seconds * (2**attempt),
                    config.max_retry_backoff_seconds,
                )
                time.sleep(delay)

    return CollectedRecord(
        source_fields=question.source_row,
        error=last_error,
    )


_RESERVED_FIELDS = {"response", "raw_response", "error", "latency_ms"}


def _flatten_record(record: CollectedRecord) -> dict[str, Any]:
    """Flatten a record for JSONL output.

    Starts with the original input columns (``source_fields``), then adds
    collector-produced fields (``response``, ``error``, etc.) and any
    ``extra_fields`` on top.
    """
    data = dict(record.source_fields)
    collisions = _RESERVED_FIELDS & record.source_fields.keys()
    if collisions:
        logger.warning(
            "Source fields %s will be overwritten by collector output", collisions
        )
    data["response"] = record.response
    data["raw_response"] = record.raw_response
    data["error"] = record.error
    data["latency_ms"] = record.latency_ms
    if record.extra_fields:
        data.update(record.extra_fields)
    return data


def _extract_extra_fields(
    raw_response: dict[str, Any],
    extra_paths: dict[str, str] | None,
) -> dict[str, Any]:
    if not extra_paths:
        return {}
    result: dict[str, Any] = {}
    for output_name, path in extra_paths.items():
        result[output_name] = extract_by_path(raw_response, path)
    return result


def _extract_openai(raw_response: dict[str, Any]) -> str | None:
    value = extract_by_path(raw_response, "choices.0.message.content")
    return str(value) if value is not None else None


def _extract_generic(raw_response: dict[str, Any], response_path: str) -> str | None:
    value = extract_by_path(raw_response, response_path)
    if value is None:
        return None
    return str(value) if not isinstance(value, str) else value


def _load_csv(path: Path, question_column: str, id_column: str) -> list[LiveQuestion]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or question_column not in reader.fieldnames:
            raise ValueError(f"CSV must include a '{question_column}' column")

        questions: list[LiveQuestion] = []
        for i, row in enumerate(reader, start=1):
            text = (row.get(question_column) or "").strip()
            if not text:
                continue
            qid = (row.get(id_column) or "").strip() or str(i)
            questions.append(
                LiveQuestion(question=text, question_id=qid, source_row=dict(row))
            )
        return questions


def _load_jsonl(path: Path, question_column: str, id_column: str) -> list[LiveQuestion]:
    questions: list[LiveQuestion] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"JSONL row {i} must be an object")
            q = _question_from_row(raw, question_column, id_column, i)
            if q:
                questions.append(q)
    return questions


def _load_json(path: Path, question_column: str, id_column: str) -> list[LiveQuestion]:
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_data, dict) and "questions" in raw_data:
        raw_data = raw_data["questions"]
    if not isinstance(raw_data, list):
        raise ValueError("JSON input must be a list or contain a 'questions' list")

    questions: list[LiveQuestion] = []
    for i, raw in enumerate(raw_data, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"JSON row {i} must be an object")
        q = _question_from_row(raw, question_column, id_column, i)
        if q:
            questions.append(q)
    return questions


def _question_from_row(
    row: dict[str, Any],
    question_column: str,
    id_column: str,
    row_index: int,
) -> LiveQuestion | None:
    raw_q = row.get(question_column)
    if raw_q is None:
        raise ValueError(f"Row {row_index} must include '{question_column}'")
    if not isinstance(raw_q, str):
        raise ValueError(f"Row {row_index} '{question_column}' must be a string")
    text = raw_q.strip()
    if not text:
        return None

    raw_id = row.get(id_column)
    qid = str(raw_id).strip() if raw_id is not None else str(row_index)

    return LiveQuestion(question=text, question_id=qid, source_row=dict(row))
