"""Unit tests for experimental live endpoint collection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from evalhub.adapter import (
    LiveCollectionConfig,
    LiveEndpointConfig,
    load_input_rows,
    load_live_collection_config,
    run_live_collection,
)


def _response(status_code: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "https://endpoint.example/v1/chat/completions"),
    )


def test_load_live_collection_config_from_parameters(tmp_path: Path) -> None:
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "responses.jsonl"

    config = load_live_collection_config(
        {
            "live_collection": {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "question_field": "prompt",
                "id_field": "case_id",
                "endpoint": {
                    "base_url": "https://endpoint.example/v1/",
                    "model": "chatbot",
                    "api_key_env": "CHATBOT_API_KEY",
                    "timeout_seconds": 5,
                    "max_retries": 0,
                },
            }
        }
    )

    assert config.input_path == input_path
    assert config.output_path == output_path
    assert config.question_field == "prompt"
    assert config.id_field == "case_id"
    assert config.endpoint.base_url == "https://endpoint.example/v1"
    assert config.endpoint.chat_completions_url == (
        "https://endpoint.example/v1/chat/completions"
    )


def test_load_live_collection_config_requires_object() -> None:
    with pytest.raises(ValueError, match='parameters\\["live_collection"\\]'):
        load_live_collection_config({})

    with pytest.raises(ValueError, match="must be an object"):
        load_live_collection_config({"live_collection": "not-an-object"})


def test_endpoint_config_rejects_non_http_urls() -> None:
    with pytest.raises(ValueError, match="http or https"):
        LiveEndpointConfig(base_url="file:///tmp/model", model="chatbot")


def test_endpoint_config_requires_https_when_api_key_is_used() -> None:
    with pytest.raises(ValueError, match="requires an https base_url"):
        LiveEndpointConfig(
            base_url="http://endpoint.example/v1",
            model="chatbot",
            api_key_env="CHATBOT_API_KEY",
        )


def test_load_input_rows_supports_csv_and_jsonl(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text("id,question\n1,hello\n", encoding="utf-8")
    assert load_input_rows(csv_path) == [{"id": "1", "question": "hello"}]

    jsonl_path = tmp_path / "questions.jsonl"
    jsonl_path.write_text(
        json.dumps({"id": "2", "question": "hi"}) + "\n\n",
        encoding="utf-8",
    )
    assert load_input_rows(jsonl_path) == [{"id": "2", "question": "hi"}]


def test_load_input_rows_rejects_non_object_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row 1 is not an object"):
        load_input_rows(path)


def test_run_live_collection_writes_jsonl_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "questions.csv"
    output_path = tmp_path / "responses.jsonl"
    input_path.write_text("id,question\nrow-1,What is up?\nrow-2,\n", encoding="utf-8")
    monkeypatch.setenv("CHATBOT_API_KEY", "secret-token")

    calls: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        _ = args
        calls.append(kwargs)
        return _response(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "collected answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 7},
            },
        )

    monkeypatch.setattr("httpx.post", fake_post)

    summary = run_live_collection(
        LiveCollectionConfig(
            input_path=input_path,
            output_path=output_path,
            id_field="id",
            endpoint=LiveEndpointConfig(
                base_url="https://endpoint.example/v1",
                model="chatbot",
                api_key_env="CHATBOT_API_KEY",
                max_retries=0,
            ),
        )
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["response"] == "collected answer"
    assert rows[0]["response_metadata"]["source_index"] == 0
    assert rows[0]["response_metadata"]["source_id"] == "row-1"
    assert rows[0]["response_metadata"]["usage"] == {"total_tokens": 7}
    assert rows[0]["error"] is None
    assert rows[1]["error"]["type"] == "missing_question"
    assert len(calls) == 1
    assert calls[0]["headers"]["authorization"] == "Bearer secret-token"
    assert calls[0]["follow_redirects"] is False

    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["experimental"] is True
    assert manifest["rows_total"] == 2
    assert manifest["rows_succeeded"] == 1
    assert manifest["rows_failed"] == 1
    assert summary.rows_total == 2
    assert summary.rows_succeeded == 1
    assert summary.rows_failed == 1


def test_run_live_collection_retries_transient_http_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "responses.jsonl"
    input_path.write_text(json.dumps({"question": "retry?"}) + "\n", encoding="utf-8")
    monkeypatch.setattr("evalhub.adapter.live_collection.time.sleep", lambda _: None)

    responses = [
        _response(500, {"error": "temporarily unavailable"}),
        _response(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "after retry"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
    ]

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        _ = args, kwargs
        return responses.pop(0)

    monkeypatch.setattr("httpx.post", fake_post)

    summary = run_live_collection(
        LiveCollectionConfig(
            input_path=input_path,
            output_path=output_path,
            endpoint=LiveEndpointConfig(
                base_url="https://endpoint.example/v1",
                model="chatbot",
                max_retries=1,
            ),
        )
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["response"] == "after retry"
    assert row["error"] is None
    assert summary.rows_succeeded == 1
    assert responses == []


def test_redirect_response_becomes_row_error_without_following_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "responses.jsonl"
    input_path.write_text(
        json.dumps({"question": "redirect?"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("CHATBOT_API_KEY", "secret-token")

    calls: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        _ = args
        calls.append(kwargs)
        return httpx.Response(
            307,
            headers={"location": "https://other.example/v1/chat/completions"},
            request=httpx.Request(
                "POST",
                "https://endpoint.example/v1/chat/completions",
            ),
        )

    monkeypatch.setattr("httpx.post", fake_post)

    summary = run_live_collection(
        LiveCollectionConfig(
            input_path=input_path,
            output_path=output_path,
            endpoint=LiveEndpointConfig(
                base_url="https://endpoint.example/v1",
                model="chatbot",
                api_key_env="CHATBOT_API_KEY",
                max_retries=0,
            ),
        )
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["response"] is None
    assert row["error"]["type"] == "HTTPStatusError"
    assert len(calls) == 1
    assert calls[0]["headers"]["authorization"] == "Bearer secret-token"
    assert calls[0]["follow_redirects"] is False
    assert summary.rows_failed == 1
