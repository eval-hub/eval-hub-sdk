"""Tests for adapter-side live response collection helpers."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from evalhub.adapter import (
    LiveCollectionConfig,
    collect_live_responses_from_parameters,
    collect_openai_chat_completions,
    load_live_questions,
)


class StubResponse:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        status_code: int = 200,
    ) -> None:
        self.payload = payload or {"choices": [{"message": {"content": "stub answer"}}]}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 300:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self.payload


class StubClient:
    def __init__(self, responses: list[StubResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> StubResponse:
        self.calls.append(
            {
                "url": url,
                "json": dict(json),
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        return self.responses.pop(0)


def test_load_live_questions_from_csv(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text(
        "id,question,topic\nq1,What is EvalHub?,sdk\n",
        encoding="utf-8",
    )

    questions = load_live_questions(questions_path)

    assert len(questions) == 1
    assert questions[0].question_id == "q1"
    assert questions[0].question == "What is EvalHub?"
    assert questions[0].metadata == {"topic": "sdk"}


def test_load_live_questions_from_jsonl(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "question": "Summarize the release.",
                "metadata": {"source": "golden"},
                "category": "summary",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    questions = load_live_questions(questions_path)

    assert questions[0].question_id == "row-1"
    assert questions[0].metadata == {
        "source": "golden",
        "category": "summary",
    }


def test_collect_openai_chat_completions_writes_jsonl_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text(
        "id,question\nq1,What is EvalHub?\nq2,What is BYOF?\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    monkeypatch.setenv("CHAT_API_KEY", "secret-token")
    client = StubClient(
        [
            StubResponse({"choices": [{"message": {"content": "answer 1"}}]}),
            StubResponse({"choices": [{"message": {"content": "answer 2"}}]}),
        ]
    )
    config = LiveCollectionConfig(
        questions_path=questions_path,
        output_dir=output_dir,
        endpoint_url="https://example.test/v1/chat/completions",
        model="test-model",
        api_key_env="CHAT_API_KEY",
        system_prompt="Answer briefly.",
        extra_body={"temperature": 0},
    )

    manifest = collect_openai_chat_completions(config, client=client)

    assert manifest.total == 2
    assert manifest.completed == 2
    assert manifest.failed == 0
    assert client.calls[0]["headers"]["Authorization"] == "Bearer secret-token"
    assert client.calls[0]["json"]["model"] == "test-model"
    assert client.calls[0]["json"]["temperature"] == 0
    assert client.calls[0]["json"]["messages"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What is EvalHub?"},
    ]

    rows = [
        json.loads(line)
        for line in (output_dir / "responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["response"] == "answer 1"
    assert rows[0]["error"] is None

    manifest_data = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_data["completed"] == 2
    assert manifest_data["output_path"].endswith("responses.jsonl")


def test_collect_openai_chat_completions_records_redirect_errors(
    tmp_path: Path,
) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps([{"question": "Will this redirect?"}]),
        encoding="utf-8",
    )
    config = LiveCollectionConfig(
        questions_path=questions_path,
        output_dir=tmp_path / "out",
        endpoint_url="https://example.test/v1/chat/completions",
        model="test-model",
    )
    client = StubClient([StubResponse(status_code=302)])

    manifest = collect_openai_chat_completions(config, client=client)

    assert manifest.completed == 0
    assert manifest.failed == 1
    row = json.loads((tmp_path / "out" / "responses.jsonl").read_text())
    assert "HTTP 302" in row["error"]


def test_collect_openai_chat_completions_retries_row_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps([{"question": "Retry this?"}]),
        encoding="utf-8",
    )
    config = LiveCollectionConfig(
        questions_path=questions_path,
        output_dir=tmp_path / "out",
        endpoint_url="https://example.test/v1/chat/completions",
        model="test-model",
        max_retries=1,
        retry_backoff_seconds=0.5,
    )
    client = StubClient(
        [
            StubResponse(status_code=500),
            StubResponse({"choices": [{"message": {"content": "recovered"}}]}),
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr("evalhub.adapter.live_collection.time.sleep", sleeps.append)

    manifest = collect_openai_chat_completions(config, client=client)

    assert manifest.completed == 1
    assert manifest.failed == 0
    assert len(client.calls) == 2
    assert sleeps == [0.5]


def test_collect_openai_chat_completions_requires_configured_auth_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text("question\nHello?\n", encoding="utf-8")
    monkeypatch.delenv("MISSING_CHAT_KEY", raising=False)
    config = LiveCollectionConfig(
        questions_path=questions_path,
        output_dir=tmp_path / "out",
        endpoint_url="https://example.test/v1/chat/completions",
        model="test-model",
        api_key_env="MISSING_CHAT_KEY",
    )

    with pytest.raises(ValueError, match="MISSING_CHAT_KEY"):
        collect_openai_chat_completions(config, client=StubClient([]))


def test_live_collection_config_rejects_non_http_endpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="endpoint_url"):
        LiveCollectionConfig(
            questions_path=tmp_path / "questions.csv",
            output_dir=tmp_path / "out",
            endpoint_url="file:///tmp/socket",
            model="test-model",
        )


def test_collect_live_responses_from_parameters(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text("question\nHello?\n", encoding="utf-8")
    client = StubClient(
        [StubResponse({"choices": [{"message": {"content": "hello"}}]})]
    )

    manifest = collect_live_responses_from_parameters(
        {
            "live_collection": {
                "questions_path": questions_path,
                "output_dir": tmp_path / "out",
                "endpoint_url": "https://example.test/v1/chat/completions",
                "model": "test-model",
            }
        },
        client=client,
    )

    assert manifest.total == 1
    assert manifest.completed == 1
