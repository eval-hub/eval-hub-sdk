"""Unit tests for the live endpoint response collector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from evalhub.adapter.collector import (
    CollectorConfig,
    CollectorError,
    CollectorProtocol,
    collect_responses,
    collect_responses_from_parameters,
    extract_by_path,
    is_configured,
    load_questions,
    substitute_template,
)

pytestmark = pytest.mark.unit


def _response(status_code: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "https://endpoint.example/v1/chat/completions"),
    )


def _openai_ok(content: str = "test answer") -> httpx.Response:
    return _response(
        200,
        {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        },
    )


# -- is_configured -----------------------------------------------------------


class TestIsConfigured:
    def test_configured_with_valid_dict(self) -> None:
        assert is_configured({"live_collection": {"endpoint_url": "http://x"}})

    def test_not_configured_empty_params(self) -> None:
        assert not is_configured({})

    def test_not_configured_none(self) -> None:
        assert not is_configured(None)

    def test_not_configured_empty_dict(self) -> None:
        assert not is_configured({"live_collection": {}})

    def test_not_configured_non_dict(self) -> None:
        assert not is_configured({"live_collection": "string"})


# -- CollectorConfig validation -----------------------------------------------


class TestCollectorConfigValidation:
    def test_rejects_non_http_endpoint(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="http:// or https://"):
            CollectorConfig(
                questions_path=tmp_path / "q.csv",
                output_dir=tmp_path / "out",
                endpoint_url="file:///etc/passwd",
                model="chatbot",
            )

    def test_openai_requires_model(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="model is required"):
            CollectorConfig(
                questions_path=tmp_path / "q.csv",
                output_dir=tmp_path / "out",
                endpoint_url="https://api.example/v1",
                protocol=CollectorProtocol.OPENAI_CHAT,
            )

    def test_generic_http_requires_template(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="request_template is required"):
            CollectorConfig(
                questions_path=tmp_path / "q.csv",
                output_dir=tmp_path / "out",
                endpoint_url="https://api.example",
                protocol=CollectorProtocol.GENERIC_HTTP,
                response_path="result.text",
            )

    def test_generic_http_requires_response_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="response_path is required"):
            CollectorConfig(
                questions_path=tmp_path / "q.csv",
                output_dir=tmp_path / "out",
                endpoint_url="https://api.example",
                protocol=CollectorProtocol.GENERIC_HTTP,
                request_template={"input": "{question}"},
            )

    def test_from_parameters_loads_config(self, tmp_path: Path) -> None:
        config = CollectorConfig.from_parameters(
            {
                "live_collection": {
                    "questions_path": str(tmp_path / "q.csv"),
                    "output_dir": str(tmp_path / "out"),
                    "endpoint_url": "https://api.example/v1",
                    "model": "chatbot",
                }
            }
        )
        assert config.model == "chatbot"
        assert config.protocol == CollectorProtocol.OPENAI_CHAT

    def test_from_parameters_missing_key(self) -> None:
        with pytest.raises(ValueError, match="live_collection"):
            CollectorConfig.from_parameters({})


# -- Auth resolution ----------------------------------------------------------


class TestAuthResolution:
    def test_api_key_env_sets_bearer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_KEY", "secret")
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        calls: list[dict[str, Any]] = []

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            calls.append(kwargs)
            return _openai_ok()

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            api_key_env="TEST_KEY",
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        collect_responses(config, client=mock_client, credentials=None)

        assert calls[0]["headers"]["Authorization"] == "Bearer secret"

    def test_missing_api_key_env_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MISSING_KEY", raising=False)
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            api_key_env="MISSING_KEY",
            use_model_credentials=False,
        )
        with pytest.raises(ValueError, match="MISSING_KEY"):
            collect_responses(config, client=MagicMock(), credentials=None)

    def test_request_headers_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_KEY", "from-env")
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        calls: list[dict[str, Any]] = []

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            calls.append(kwargs)
            return _openai_ok()

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            api_key_env="TEST_KEY",
            request_headers={"Authorization": "Custom override"},
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        collect_responses(config, client=mock_client, credentials=None)

        assert calls[0]["headers"]["Authorization"] == "Custom override"


# -- Question loading ---------------------------------------------------------


class TestLoadQuestions:
    def test_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "q.csv"
        p.write_text("id,question\n1,hello\n2,world\n", encoding="utf-8")
        qs = load_questions(p)
        assert len(qs) == 2
        assert qs[0].question == "hello"
        assert qs[0].question_id == "1"

    def test_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "q.jsonl"
        p.write_text(
            json.dumps({"question": "hi", "id": "a"})
            + "\n"
            + json.dumps({"question": "bye", "id": "b"})
            + "\n",
            encoding="utf-8",
        )
        qs = load_questions(p)
        assert len(qs) == 2
        assert qs[1].question == "bye"

    def test_json_array(self, tmp_path: Path) -> None:
        p = tmp_path / "q.json"
        p.write_text(json.dumps([{"question": "test"}]), encoding="utf-8")
        qs = load_questions(p)
        assert len(qs) == 1

    def test_json_wrapper(self, tmp_path: Path) -> None:
        p = tmp_path / "q.json"
        p.write_text(
            json.dumps({"questions": [{"question": "test"}]}), encoding="utf-8"
        )
        qs = load_questions(p)
        assert len(qs) == 1

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "q.csv"
        p.write_text("question\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No questions"):
            load_questions(p)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_questions(tmp_path / "missing.csv")

    def test_skips_empty_questions(self, tmp_path: Path) -> None:
        p = tmp_path / "q.csv"
        p.write_text("question\nhello\n\nworld\n", encoding="utf-8")
        qs = load_questions(p)
        assert len(qs) == 2

    def test_preserves_source_row(self, tmp_path: Path) -> None:
        p = tmp_path / "q.csv"
        p.write_text(
            "question,category,difficulty\nhello,greet,easy\n", encoding="utf-8"
        )
        qs = load_questions(p)
        assert qs[0].source_row == {
            "question": "hello",
            "category": "greet",
            "difficulty": "easy",
        }


# -- Template substitution and path extraction --------------------------------


class TestSubstituteTemplate:
    def test_string(self) -> None:
        assert substitute_template("Hi {question}", {"question": "hello"}) == "Hi hello"

    def test_nested_dict(self) -> None:
        result = substitute_template({"a": {"b": "{question}"}}, {"question": "test"})
        assert result == {"a": {"b": "test"}}

    def test_list(self) -> None:
        result = substitute_template(["{question}", 42], {"question": "hi"})
        assert result == ["hi", 42]

    def test_non_string_passthrough(self) -> None:
        assert substitute_template(42, {"question": "hi"}) == 42
        assert substitute_template(True, {"question": "hi"}) is True


class TestExtractByPath:
    def test_simple_key(self) -> None:
        assert extract_by_path({"a": "val"}, "a") == "val"

    def test_nested(self) -> None:
        assert extract_by_path({"a": {"b": {"c": "val"}}}, "a.b.c") == "val"

    def test_array_index(self) -> None:
        assert extract_by_path({"a": [10, 20, 30]}, "a.1") == 20

    def test_missing_returns_none(self) -> None:
        assert extract_by_path({"a": 1}, "b") is None

    def test_openai_path(self) -> None:
        data = {"choices": [{"message": {"content": "answer"}}]}
        assert extract_by_path(data, "choices.0.message.content") == "answer"

    def test_mcp_path(self) -> None:
        data = {"result": {"content": [{"text": "answer"}]}}
        assert extract_by_path(data, "result.content.0.text") == "answer"


# -- OpenAI chat collection ---------------------------------------------------


class TestCollectOpenAI:
    def test_collects_responses(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\nworld\n", encoding="utf-8")

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            return _openai_ok("answer")

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        manifest = collect_responses(config, client=mock_client, credentials=None)

        assert manifest.total == 2
        assert manifest.completed == 2
        assert manifest.failed == 0

        output = tmp_path / "out" / "responses.jsonl"
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert rows[0]["response"] == "answer"
        assert rows[0]["error"] is None
        assert rows[0]["latency_ms"] is not None

    def test_writes_manifest(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=_openai_ok())

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            use_model_credentials=False,
        )
        collect_responses(config, client=mock_client, credentials=None)

        manifest_path = tmp_path / "out" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["experimental"] is True
        assert manifest["protocol"] == "openai_chat_completions"
        assert manifest["total"] == 1
        assert manifest["completed"] == 1

    def test_system_prompt_included(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        calls: list[dict[str, Any]] = []

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            calls.append(kwargs)
            return _openai_ok()

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            system_prompt="You are a helpful assistant.",
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        collect_responses(config, client=mock_client, credentials=None)

        body = calls[0]["json"]
        assert body["messages"][0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert body["messages"][1] == {"role": "user", "content": "hello"}

    def test_redirect_recorded_as_error(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                307,
                headers={"location": "https://other.example"},
                request=httpx.Request(
                    "POST", "https://api.example/v1/chat/completions"
                ),
            )

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        manifest = collect_responses(config, client=mock_client, credentials=None)

        assert manifest.failed == 1
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "responses.jsonl").read_text().splitlines()
        ]
        assert "Redirect" in rows[0]["error"]

    def test_fail_fast_raises(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\nworld\n", encoding="utf-8")

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            return _response(500, {"error": "server error"})

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            fail_fast=True,
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        with pytest.raises(CollectorError):
            collect_responses(config, client=mock_client, credentials=None)

    def test_best_effort_records_errors(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\nworld\n", encoding="utf-8")

        call_count = 0

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _response(500, {"error": "fail"})
            return _openai_ok()

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            fail_fast=False,
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        manifest = collect_responses(config, client=mock_client, credentials=None)

        assert manifest.failed == 1
        assert manifest.completed == 1

    def test_retries_transient_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")
        monkeypatch.setattr("evalhub.adapter.collector.time.sleep", lambda _: None)

        responses = [
            _response(500, {"error": "temporary"}),
            _openai_ok("after retry"),
        ]

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            return responses.pop(0)

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            max_retries=1,
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        manifest = collect_responses(config, client=mock_client, credentials=None)

        assert manifest.completed == 1
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "responses.jsonl").read_text().splitlines()
        ]
        assert rows[0]["response"] == "after retry"

    def test_progress_callback(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\na\nb\nc\n", encoding="utf-8")

        progress: list[float] = []

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/v1",
            model="chatbot",
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=_openai_ok())
        collect_responses(
            config,
            client=mock_client,
            credentials=None,
            progress_callback=progress.append,
        )

        assert len(progress) == 3
        assert progress[-1] == pytest.approx(1.0)


# -- Generic HTTP collection --------------------------------------------------


class TestCollectGenericHTTP:
    def test_template_substitution(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        calls: list[dict[str, Any]] = []

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            calls.append(kwargs)
            return _response(200, {"result": {"text": "answer"}})

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example/invoke",
            protocol=CollectorProtocol.GENERIC_HTTP,
            request_template={"input": "{question}", "id": "{question_id}"},
            response_path="result.text",
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        manifest = collect_responses(config, client=mock_client, credentials=None)

        assert calls[0]["json"]["input"] == "hello"
        assert manifest.completed == 1

        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "responses.jsonl").read_text().splitlines()
        ]
        assert rows[0]["response"] == "answer"

    def test_mcp_style_request(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.jsonl"
        qpath.write_text(
            json.dumps({"question": "What is K8s?"}) + "\n", encoding="utf-8"
        )

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            return _response(
                200,
                {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [{"type": "text", "text": "Kubernetes is..."}]
                    },
                    "id": "1",
                },
            )

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://mcp-chatbot.example/mcp",
            protocol=CollectorProtocol.GENERIC_HTTP,
            request_template={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "chat", "arguments": {"message": "{question}"}},
                "id": "{question_id}",
            },
            response_path="result.content.0.text",
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        manifest = collect_responses(config, client=mock_client, credentials=None)

        assert manifest.completed == 1
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "responses.jsonl").read_text().splitlines()
        ]
        assert rows[0]["response"] == "Kubernetes is..."

    def test_extra_response_paths_extracted(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            return _response(
                200,
                {
                    "output": {
                        "answer": "the answer",
                        "context": [{"source": "doc1", "text": "relevant text"}],
                        "score": 0.95,
                    }
                },
            )

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example",
            protocol=CollectorProtocol.GENERIC_HTTP,
            request_template={"input": {"question": "{question}"}},
            response_path="output.answer",
            extra_response_paths={
                "contexts": "output.context",
                "confidence": "output.score",
            },
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        collect_responses(config, client=mock_client, credentials=None)

        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "responses.jsonl").read_text().splitlines()
        ]
        assert rows[0]["response"] == "the answer"
        assert rows[0]["contexts"] == [{"source": "doc1", "text": "relevant text"}]
        assert rows[0]["confidence"] == 0.95

    def test_input_metadata_flattened_to_top_level(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text(
            "question,reference,category\nWhat is X?,X is Y,general\n",
            encoding="utf-8",
        )

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            return _response(200, {"output": {"answer": "X is Y indeed"}})

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example",
            protocol=CollectorProtocol.GENERIC_HTTP,
            request_template={"input": {"question": "{question}"}},
            response_path="output.answer",
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        collect_responses(config, client=mock_client, credentials=None)

        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "responses.jsonl").read_text().splitlines()
        ]
        assert rows[0]["response"] == "X is Y indeed"
        assert rows[0]["reference"] == "X is Y"
        assert rows[0]["category"] == "general"
        assert "metadata" not in rows[0]

    def test_response_path_missing_returns_none(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        def fake_post(*_: Any, **kwargs: Any) -> httpx.Response:
            return _response(200, {"unexpected": "shape"})

        config = CollectorConfig(
            questions_path=qpath,
            output_dir=tmp_path / "out",
            endpoint_url="https://api.example",
            protocol=CollectorProtocol.GENERIC_HTTP,
            request_template={"q": "{question}"},
            response_path="result.text",
            use_model_credentials=False,
        )
        mock_client = MagicMock()
        mock_client.post = fake_post
        collect_responses(config, client=mock_client, credentials=None)

        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "responses.jsonl").read_text().splitlines()
        ]
        assert rows[0]["response"] is None


# -- collect_responses_from_parameters ----------------------------------------


class TestCollectFromParameters:
    def test_convenience_wrapper(self, tmp_path: Path) -> None:
        qpath = tmp_path / "q.csv"
        qpath.write_text("question\nhello\n", encoding="utf-8")

        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=_openai_ok())

        manifest = collect_responses_from_parameters(
            {
                "live_collection": {
                    "questions_path": str(qpath),
                    "output_dir": str(tmp_path / "out"),
                    "endpoint_url": "https://api.example/v1",
                    "model": "chatbot",
                    "use_model_credentials": False,
                }
            },
            client=mock_client,
            credentials=None,
        )

        assert manifest.completed == 1
