"""Experimental helpers for collecting live endpoint responses in adapters."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator


class LiveCollectionConfig(BaseModel):
    """Configuration for adapter-side live response collection.

    The shape is intentionally small and can be passed through
    ``JobSpec.parameters["live_collection"]`` while the server/API contract is
    still being validated.
    """

    questions_path: Path = Field(
        ..., description="CSV, JSON, or JSONL file containing input questions"
    )
    output_dir: Path = Field(
        ..., description="Directory where responses.jsonl and manifest.json are written"
    )
    endpoint_url: str = Field(
        ..., description="OpenAI-compatible /v1/chat/completions endpoint URL"
    )
    model: str = Field(..., description="Model name to send in each request")
    endpoint_type: Literal["openai_chat_completions"] = Field(
        default="openai_chat_completions",
        description="Live endpoint collector implementation to use",
    )
    question_column: str = Field(
        default="question", description="CSV/JSON field containing the prompt text"
    )
    id_column: str = Field(default="id", description="Optional question id field")
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable containing a bearer token for the endpoint",
    )
    request_headers: dict[str, str] = Field(
        default_factory=dict, description="Additional headers sent with every request"
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system message sent before each question"
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict, description="Additional JSON body fields for each request"
    )
    timeout_seconds: float = Field(
        default=30.0, gt=0.0, description="Per-request timeout"
    )
    max_retries: int = Field(
        default=0, ge=0, description="Number of retries after the initial request"
    )

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> LiveCollectionConfig:
        """Load config from ``JobSpec.parameters["live_collection"]``."""
        raw_config = parameters.get("live_collection")
        if raw_config is None:
            raise ValueError('JobSpec.parameters["live_collection"] is required')
        return cls.model_validate(raw_config)

    @field_validator("endpoint_url")
    @classmethod
    def _endpoint_url_must_be_http(cls, value: str) -> str:
        if not value.lower().startswith(("http://", "https://")):
            raise ValueError("endpoint_url must use http:// or https://")
        return value


class LiveQuestion(BaseModel):
    """One question loaded from a CSV/JSON/JSONL input file."""

    question: str
    question_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiveCollectionRecord(BaseModel):
    """One collected response row."""

    question: str
    question_id: str | None = None
    response: str | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiveCollectionManifest(BaseModel):
    """Summary of a live collection run."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    endpoint_type: str
    endpoint_url: str
    model: str
    questions_path: Path
    output_path: Path
    total: int
    completed: int
    failed: int


def load_live_questions(
    path: Path | str,
    *,
    question_column: str = "question",
    id_column: str = "id",
) -> list[LiveQuestion]:
    """Load live collection questions from CSV, JSON, or JSONL."""
    question_path = Path(path)
    if not question_path.exists():
        raise FileNotFoundError(f"Question file not found: {question_path}")

    suffix = question_path.suffix.lower()
    if suffix == ".csv":
        questions = _load_csv_questions(question_path, question_column, id_column)
    elif suffix == ".jsonl":
        questions = _load_jsonl_questions(question_path, question_column, id_column)
    elif suffix == ".json":
        questions = _load_json_questions(question_path, question_column, id_column)
    else:
        raise ValueError(
            f"Unsupported question file type '{suffix}'. Use CSV, JSON, or JSONL."
        )

    if not questions:
        raise ValueError(f"No questions found in {question_path}")
    return questions


def collect_openai_chat_completions(
    config: LiveCollectionConfig,
    *,
    client: Any | None = None,
) -> LiveCollectionManifest:
    """Collect responses from an OpenAI-compatible chat-completions endpoint.

    Adapters can call this during ``JobPhase.LOADING_DATA`` and pass the
    returned ``responses.jsonl`` path into their evaluation framework loader.
    Row-level request failures are recorded in the output JSONL instead of
    aborting the whole collection run.
    """
    questions = load_live_questions(
        config.questions_path,
        question_column=config.question_column,
        id_column=config.id_column,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = config.output_dir / "responses.jsonl"
    headers = _build_headers(config)
    active_client = client
    owns_client = active_client is None

    if active_client is None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Live collection requires httpx. Install eval-hub-sdk[adapter] "
                "or eval-hub-sdk[core]."
            ) from exc

        active_client = httpx.Client(
            follow_redirects=False,
            timeout=config.timeout_seconds,
        )

    completed = 0
    failed = 0
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            for question in questions:
                record = _collect_one_question(config, active_client, headers, question)
                if record.error:
                    failed += 1
                else:
                    completed += 1
                output_file.write(json.dumps(record.model_dump(mode="json")) + "\n")
    finally:
        if owns_client and hasattr(active_client, "close"):
            active_client.close()

    manifest = LiveCollectionManifest(
        endpoint_type=config.endpoint_type,
        endpoint_url=config.endpoint_url,
        model=config.model,
        questions_path=config.questions_path,
        output_path=output_path,
        total=len(questions),
        completed=completed,
        failed=failed,
    )
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def collect_live_responses_from_parameters(
    parameters: Mapping[str, Any],
    *,
    client: Any | None = None,
) -> LiveCollectionManifest:
    """Collect responses from ``JobSpec.parameters["live_collection"]`` config."""
    config = LiveCollectionConfig.from_parameters(parameters)
    return collect_openai_chat_completions(config, client=client)


def _load_csv_questions(
    path: Path,
    question_column: str,
    id_column: str,
) -> list[LiveQuestion]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or question_column not in reader.fieldnames:
            raise ValueError(f"CSV file must include a '{question_column}' column")

        questions: list[LiveQuestion] = []
        for row_index, row in enumerate(reader, start=1):
            question = (row.get(question_column) or "").strip()
            if not question:
                continue
            question_id = (row.get(id_column) or "").strip() or None
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {question_column, id_column} and value not in (None, "")
            }
            if question_id is None:
                question_id = str(row_index)
            questions.append(
                LiveQuestion(
                    question=question,
                    question_id=question_id,
                    metadata=metadata,
                )
            )
    return questions


def _load_jsonl_questions(
    path: Path,
    question_column: str,
    id_column: str,
) -> list[LiveQuestion]:
    questions: list[LiveQuestion] = []
    with path.open(encoding="utf-8") as jsonl_file:
        for row_index, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            raw_row = json.loads(line)
            if not isinstance(raw_row, dict):
                raise ValueError("Each JSONL line must be an object")
            questions.append(
                _question_from_mapping(
                    cast(Mapping[str, Any], raw_row),
                    question_column,
                    id_column,
                    row_index,
                )
            )
    return questions


def _load_json_questions(
    path: Path,
    question_column: str,
    id_column: str,
) -> list[LiveQuestion]:
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_data, dict) and "questions" in raw_data:
        raw_data = raw_data["questions"]
    if not isinstance(raw_data, list):
        raise ValueError(
            "JSON question input must be a list or contain a questions list"
        )

    questions: list[LiveQuestion] = []
    for row_index, raw_row in enumerate(raw_data, start=1):
        if not isinstance(raw_row, dict):
            raise ValueError("Each JSON question row must be an object")
        questions.append(
            _question_from_mapping(
                cast(Mapping[str, Any], raw_row),
                question_column,
                id_column,
                row_index,
            )
        )
    return questions


def _question_from_mapping(
    row: Mapping[str, Any],
    question_column: str,
    id_column: str,
    row_index: int,
) -> LiveQuestion:
    raw_question = row.get(question_column)
    if not isinstance(raw_question, str) or not raw_question.strip():
        raise ValueError(f"Question row {row_index} must include '{question_column}'")

    raw_id = row.get(id_column)
    question_id = str(raw_id) if raw_id not in (None, "") else str(row_index)

    metadata: dict[str, Any] = {}
    explicit_metadata = row.get("metadata")
    if isinstance(explicit_metadata, dict):
        metadata.update(cast(dict[str, Any], explicit_metadata))
    for key, value in row.items():
        if key not in {question_column, id_column, "metadata"}:
            metadata[key] = value

    return LiveQuestion(
        question=raw_question.strip(),
        question_id=question_id,
        metadata=metadata,
    )


def _build_headers(config: LiveCollectionConfig) -> dict[str, str]:
    headers = dict(config.request_headers)
    if config.api_key_env:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {config.api_key_env} is not set")
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _collect_one_question(
    config: LiveCollectionConfig,
    client: Any,
    headers: Mapping[str, str],
    question: LiveQuestion,
) -> LiveCollectionRecord:
    last_error: str | None = None
    for _attempt in range(config.max_retries + 1):
        try:
            response = client.post(
                config.endpoint_url,
                json=_chat_completion_body(config, question),
                headers=dict(headers),
                timeout=config.timeout_seconds,
            )
            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int) and 300 <= status_code < 400:
                raise ValueError(f"Redirect response not allowed: HTTP {status_code}")
            response.raise_for_status()
            raw_response = response.json()
            if not isinstance(raw_response, dict):
                raise ValueError("Chat completion response must be a JSON object")
            return LiveCollectionRecord(
                question=question.question,
                question_id=question.question_id,
                response=_extract_chat_content(cast(dict[str, Any], raw_response)),
                raw_response=cast(dict[str, Any], raw_response),
                metadata=question.metadata,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return LiveCollectionRecord(
        question=question.question,
        question_id=question.question_id,
        error=last_error,
        metadata=question.metadata,
    )


def _chat_completion_body(
    config: LiveCollectionConfig,
    question: LiveQuestion,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": question.question})

    body: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
    }
    body.update(config.extra_body)
    return body


def _extract_chat_content(raw_response: Mapping[str, Any]) -> str | None:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts) if parts else None
    return None
