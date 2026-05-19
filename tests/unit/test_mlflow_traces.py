"""MLflow trace search, parsing, and materialization via TracesNamespace."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from evalhub.adapter.mlflow import (
    MlflowClient,
    TracesNamespace,
    _parse_trace,
)

# -- TracesNamespace.is_source_configured -----------------------------------


def test_configured_with_experiment_id() -> None:
    assert TracesNamespace.is_source_configured({"mlflow_traces_experiment_id": "42"})


def test_configured_with_experiment_name() -> None:
    assert TracesNamespace.is_source_configured(
        {"mlflow_traces_experiment_name": "my-exp"}
    )


def test_not_configured_empty() -> None:
    assert not TracesNamespace.is_source_configured({})


def test_not_configured_none() -> None:
    assert not TracesNamespace.is_source_configured(None)


def test_not_configured_blank_values() -> None:
    assert not TracesNamespace.is_source_configured(
        {"mlflow_traces_experiment_id": "", "mlflow_traces_experiment_name": "  "}
    )


# -- _parse_trace -----------------------------------------------------------


def test_parse_trace_basic() -> None:
    raw = {
        "info": {
            "request_id": "tr-abc123",
            "experiment_id": "1",
            "timestamp_ms": 1700000000000,
            "execution_time_ms": 500,
            "status": "OK",
            "tags": [
                {"key": "framework", "value": "langgraph"},
            ],
            "request_metadata": [
                {"key": "source", "value": "test"},
            ],
        },
        "data": {"spans": [{"name": "root"}]},
    }
    trace = _parse_trace(raw)
    assert trace.info.request_id == "tr-abc123"
    assert trace.info.experiment_id == "1"
    assert trace.info.status == "OK"
    assert trace.info.tags == {"framework": "langgraph"}
    assert trace.info.request_metadata == {"source": "test"}
    assert trace.data == {"spans": [{"name": "root"}]}


def test_parse_trace_missing_fields() -> None:
    trace = _parse_trace({})
    assert trace.info.request_id == ""
    assert trace.info.tags == {}
    assert trace.data == {}


# -- TracesNamespace.search (mocked HTTP) -----------------------------------


def test_search_traces_parses_response() -> None:
    mock_response = {
        "traces": [
            {
                "info": {
                    "request_id": "tr-001",
                    "experiment_id": "5",
                    "timestamp_ms": 1700000000000,
                    "execution_time_ms": 200,
                    "status": "OK",
                    "tags": [],
                    "request_metadata": [],
                },
                "data": {},
            },
            {
                "info": {
                    "request_id": "tr-002",
                    "experiment_id": "5",
                    "timestamp_ms": 1700000001000,
                    "execution_time_ms": 150,
                    "status": "ERROR",
                    "tags": [{"key": "k", "value": "v"}],
                    "request_metadata": [],
                },
                "data": {"spans": []},
            },
        ],
        "next_page_token": "tok123",
    }
    mock_client = MagicMock(spec=MlflowClient)
    mock_client._get = MagicMock(return_value=mock_response)
    ns = TracesNamespace(mock_client)

    traces, token = ns.search(experiment_ids=["5"], max_results=10)
    assert len(traces) == 2
    assert traces[0].info.request_id == "tr-001"
    assert traces[1].info.status == "ERROR"
    assert traces[1].info.tags == {"k": "v"}
    assert token == "tok123"


def test_search_traces_no_results() -> None:
    mock_client = MagicMock(spec=MlflowClient)
    mock_client._get = MagicMock(return_value={"traces": []})
    ns = TracesNamespace(mock_client)

    traces, token = ns.search(experiment_ids=["1"])
    assert traces == []
    assert token is None


# -- TracesNamespace.materialize --------------------------------------------


def test_materialize_writes_files(tmp_path: Path) -> None:
    mock_client = MagicMock(spec=MlflowClient)
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="10")
    ns = TracesNamespace(mock_client)
    mock_client._get = MagicMock(
        return_value={
            "traces": [
                {
                    "info": {
                        "request_id": "abc",
                        "experiment_id": "10",
                        "timestamp_ms": 1700000000000,
                        "execution_time_ms": 300,
                        "status": "OK",
                        "tags": [],
                        "request_metadata": [],
                    },
                    "data": {"spans": [{"name": "root"}]},
                },
                {
                    "info": {
                        "request_id": "tr-def",
                        "experiment_id": "10",
                        "timestamp_ms": 0,
                        "execution_time_ms": 0,
                        "status": "ERROR",
                        "tags": [],
                        "request_metadata": [],
                    },
                    "data": {},
                },
            ],
            "next_page_token": None,
        }
    )

    out = ns.materialize(
        parameters={
            "mlflow_traces_experiment_name": "test-exp",
            "mlflow_traces_max_results": 10,
        },
        output_dir=tmp_path / "traces",
    )

    files = sorted(out.iterdir())
    assert len(files) == 2
    names = [f.name for f in files]
    assert "tr-abc.json" in names
    assert "tr-def.json" in names

    content = json.loads((out / "tr-abc.json").read_text())
    assert content["info"]["request_id"] == "abc"
    assert content["data"]["spans"] == [{"name": "root"}]


def test_materialize_resolves_experiment_name(tmp_path: Path) -> None:
    mock_client = MagicMock(spec=MlflowClient)
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="42")
    ns = TracesNamespace(mock_client)
    mock_client._get = MagicMock(
        return_value={
            "traces": [
                {
                    "info": {
                        "request_id": "x",
                        "experiment_id": "42",
                        "timestamp_ms": 0,
                        "execution_time_ms": 0,
                        "status": "OK",
                        "tags": [],
                        "request_metadata": [],
                    },
                    "data": {},
                },
            ],
            "next_page_token": None,
        }
    )

    ns.materialize(
        parameters={"mlflow_traces_experiment_name": "my-exp"},
        output_dir=tmp_path,
    )
    mock_client.get_experiment_by_name.assert_called_once_with("my-exp")


def test_materialize_no_results_raises(tmp_path: Path) -> None:
    mock_client = MagicMock(spec=MlflowClient)
    mock_client._get = MagicMock(return_value={"traces": [], "next_page_token": None})
    ns = TracesNamespace(mock_client)

    with pytest.raises(ValueError, match="No traces returned"):
        ns.materialize(
            parameters={"mlflow_traces_experiment_id": "99"},
            output_dir=tmp_path,
        )
