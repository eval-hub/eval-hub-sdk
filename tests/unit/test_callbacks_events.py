"""Tests for DefaultCallbacks POST /events payload (mlflow_run_id)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from evalhub.adapter.callbacks import DefaultCallbacks
from evalhub.adapter.models.job import (
    JobResults,
    JobSpec,
)
from evalhub.models.api import EvaluationResult, JobStatus, ModelConfig

pytestmark = pytest.mark.unit


def _results(mlflow_run_id: str | None = None) -> JobResults:
    return JobResults(
        id="job-1",
        benchmark_id="arc_easy",
        benchmark_index=0,
        model_name="m",
        results=[
            EvaluationResult(metric_name="acc", metric_value=0.9, metric_type="float")
        ],
        num_examples_evaluated=1,
        duration_seconds=1.0,
        completed_at=datetime.now(UTC),
        mlflow_run_id=mlflow_run_id,
    )


def _job_spec(experiment_name: str = "exp") -> JobSpec:
    return JobSpec(
        id="job-1",
        provider_id="lm_evaluation_harness",
        benchmark_id="arc_easy",
        benchmark_index=0,
        model=ModelConfig(url="http://localhost/v1", name="m"),
        parameters={},
        callback_url="http://evalhub:8080",
        experiment_name=experiment_name,
    )


def _make_callbacks(
    provider_id: str | None = "lm_evaluation_harness",
) -> tuple[DefaultCallbacks, MagicMock]:
    """Create a DefaultCallbacks with a mocked HTTP client."""
    mock_http = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    mock_http.post.return_value = resp

    with patch.object(DefaultCallbacks, "_create_http_client", return_value=mock_http):
        callbacks = DefaultCallbacks(
            job_id="job-1",
            benchmark_id="arc_easy",
            provider_id=provider_id,
            benchmark_index=0,
            sidecar_url="http://evalhub:8080",
            insecure=True,
        )
    return callbacks, mock_http


def test_report_results_sends_mlflow_run_id_when_set_on_job_results() -> None:
    callbacks, mock_http = _make_callbacks()

    callbacks.report_results(_results(mlflow_run_id="mlflow-run-abc"))

    mock_http.post.assert_called_once()
    body = mock_http.post.call_args.kwargs["json"]
    assert body["benchmark_status_event"]["mlflow_run_id"] == "mlflow-run-abc"


def test_report_results_omits_mlflow_run_id_when_not_set() -> None:
    callbacks, mock_http = _make_callbacks()

    callbacks.report_results(_results())

    body = mock_http.post.call_args.kwargs["json"]
    assert "mlflow_run_id" not in body["benchmark_status_event"]


def test_mlflow_save_returns_run_id_from_odh_path() -> None:
    """Regression: save() must return _save_odh/_save_upstream result (not None)."""
    from evalhub.adapter.callbacks import _MlflowOps
    from evalhub.adapter.config import MlflowBackend
    from evalhub.adapter.models.job import JobResults, JobSpec
    from evalhub.models.api import EvaluationResult, ModelConfig

    spec = JobSpec(
        id="j1",
        provider_id="p",
        benchmark_id="b",
        benchmark_index=0,
        model=ModelConfig(url="http://localhost/v1", name="m"),
        parameters={},
        callback_url="http://localhost/",
        experiment_name="exp",
    )
    results = JobResults(
        id="j1",
        benchmark_id="b",
        benchmark_index=0,
        model_name="m",
        results=[
            EvaluationResult(metric_name="acc", metric_value=1.0, metric_type="float")
        ],
        num_examples_evaluated=1,
        duration_seconds=1.0,
        completed_at=datetime.now(UTC),
    )
    ops = _MlflowOps(backend=MlflowBackend.ODH)
    with patch.object(_MlflowOps, "_save_odh", return_value="run-from-odh") as m:
        rid = ops.save(results, spec)
    assert rid == "run-from-odh"
    m.assert_called_once()


def test_mlflow_save_returns_run_id_from_upstream_path() -> None:
    from evalhub.adapter.callbacks import _MlflowOps
    from evalhub.adapter.config import MlflowBackend
    from evalhub.adapter.models.job import JobResults, JobSpec
    from evalhub.models.api import EvaluationResult, ModelConfig

    spec = JobSpec(
        id="j1",
        provider_id="p",
        benchmark_id="b",
        benchmark_index=0,
        model=ModelConfig(url="http://localhost/v1", name="m"),
        parameters={},
        callback_url="http://localhost/",
        experiment_name="exp",
    )
    results = JobResults(
        id="j1",
        benchmark_id="b",
        benchmark_index=0,
        model_name="m",
        results=[
            EvaluationResult(metric_name="acc", metric_value=1.0, metric_type="float")
        ],
        num_examples_evaluated=1,
        duration_seconds=1.0,
        completed_at=datetime.now(UTC),
    )
    ops = _MlflowOps(backend=MlflowBackend.UPSTREAM)
    with patch.object(_MlflowOps, "_save_upstream", return_value="run-upstream") as m:
        rid = ops.save(results, spec)
    assert rid == "run-upstream"
    m.assert_called_once()


def test_mlflow_save_posts_failed_event_on_mlflow_error() -> None:
    callbacks, mock_http = _make_callbacks()

    with patch.object(
        callbacks.mlflow,
        "_save_odh",
        side_effect=RuntimeError("mlflow offline"),
    ):
        with pytest.raises(RuntimeError, match="MLflow save failed: mlflow offline"):
            callbacks.mlflow.save(_results(), _job_spec())

    body = mock_http.post.call_args.kwargs["json"]["benchmark_status_event"]
    assert body["status"] == JobStatus.FAILED.value
    assert (
        body["error_message"]["message"]
        == "Failed to save evaluation results to MLflow."
    )
    assert body["error_message"]["message_code"] == "mlflow_save_failed"
    assert body["error_message"]["message_origin"] == "sdk"
    assert "mlflow offline" not in body["error_message"]["message"]
    assert "warning_message" not in body


def test_build_run_name_format() -> None:
    from evalhub.adapter.callbacks import _MlflowOps
    from evalhub.adapter.models.job import JobSpec
    from evalhub.models.api import ModelConfig

    spec = JobSpec(
        id="job-42",
        provider_id="p",
        benchmark_id="mmlu",
        benchmark_index=3,
        model=ModelConfig(url="http://localhost/v1", name="m"),
        parameters={},
        callback_url="http://localhost/",
    )
    assert _MlflowOps._build_run_name(spec) == "job-42_3"


def test_build_params_metrics_base_params_and_metrics() -> None:
    from evalhub.adapter.callbacks import _MlflowOps
    from evalhub.adapter.models.job import JobResults, JobSpec
    from evalhub.models.api import EvaluationResult, ModelConfig

    spec = JobSpec(
        id="j1",
        provider_id="lm_evaluation_harness",
        benchmark_id="mmlu",
        benchmark_index=0,
        model=ModelConfig(url="http://localhost/v1", name="m"),
        parameters={},
        callback_url="http://localhost/",
    )
    results = JobResults(
        id="j1",
        benchmark_id="mmlu",
        benchmark_index=0,
        model_name="gpt-4",
        results=[
            EvaluationResult(metric_name="acc", metric_value=0.9, metric_type="float"),
            EvaluationResult(
                metric_name="f1", metric_value="N/A", metric_type="string"
            ),
        ],
        num_examples_evaluated=100,
        duration_seconds=42.5,
        overall_score=0.85,
        completed_at=datetime.now(UTC),
    )
    params, metrics = _MlflowOps._build_params_metrics(results, spec)
    param_dict = {p.key: p.value for p in params}

    assert param_dict == {
        "benchmark_id": "mmlu",
        "provider_id": "lm_evaluation_harness",
        "model_name": "gpt-4",
        "num_examples_evaluated": "100",
        "duration_seconds": "42.5",
    }

    metric_dict = {m.key: m.value for m in metrics}
    assert metric_dict["acc"] == 0.9
    assert metric_dict["overall_score"] == 0.85
    assert "f1" not in metric_dict


# ---------------------------------------------------------------------------
# report_status payload tests
# ---------------------------------------------------------------------------


def test_report_status_sends_error_message() -> None:
    from evalhub.adapter.models.job import JobStatusUpdate, MessageInfo
    from evalhub.models.api import JobStatus

    callbacks, mock_http = _make_callbacks()
    callbacks.report_status(
        JobStatusUpdate(
            status=JobStatus.FAILED,
            error_message=MessageInfo(
                message="boom",
                message_code="kaboom",
            ),
        )
    )

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert event["error_message"] == {
        "message": "boom",
        "message_code": "kaboom",
        "message_origin": "adapter",
    }


def test_report_status_sends_warning_message() -> None:
    from evalhub.adapter.models.job import JobStatusUpdate, MessageInfo
    from evalhub.models.api import JobStatus

    callbacks, mock_http = _make_callbacks()
    callbacks.report_status(
        JobStatusUpdate(
            status=JobStatus.RUNNING,
            warning_message=MessageInfo(
                message="slow",
                message_code="slow_response",
            ),
        )
    )

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert event["warning_message"] == {
        "message": "slow",
        "message_code": "slow_response",
        "message_origin": "adapter",
    }


def test_report_status_preserves_explicit_message_origin() -> None:
    from evalhub.adapter.models.job import JobStatusUpdate, MessageInfo
    from evalhub.models.api import JobStatus, MessageOrigin

    callbacks, mock_http = _make_callbacks()
    callbacks.report_status(
        JobStatusUpdate(
            status=JobStatus.FAILED,
            error_message=MessageInfo(
                message="sdk boom",
                message_code="sdk_err",
                message_origin=MessageOrigin.SDK,
            ),
        )
    )

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert event["error_message"]["message_origin"] == "sdk"


def test_report_status_always_includes_provider_id() -> None:
    from evalhub.adapter.models.job import JobStatusUpdate
    from evalhub.models.api import JobStatus

    callbacks, mock_http = _make_callbacks(provider_id=None)
    callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING))

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert "provider_id" in event
    assert event["provider_id"] == ""


def test_report_status_does_not_send_state_or_message() -> None:
    from evalhub.adapter.models.job import JobStatusUpdate
    from evalhub.models.api import JobStatus

    callbacks, mock_http = _make_callbacks()
    callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING))

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert "state" not in event
    assert "message" not in event


def test_report_status_sends_phase_when_set() -> None:
    from evalhub.adapter.models.job import JobPhase, JobStatusUpdate
    from evalhub.models.api import JobStatus

    callbacks, mock_http = _make_callbacks()
    callbacks.report_status(
        JobStatusUpdate(
            status=JobStatus.RUNNING,
            phase=JobPhase.RUNNING_EVALUATION,
        )
    )

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert event["phase"] == "running_evaluation"


def test_report_status_omits_phase_when_not_set() -> None:
    from evalhub.adapter.models.job import JobStatusUpdate
    from evalhub.models.api import JobStatus

    callbacks, mock_http = _make_callbacks()
    callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING))

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert "phase" not in event


# ---------------------------------------------------------------------------
# report_results payload tests
# ---------------------------------------------------------------------------


def test_report_results_does_not_send_state_message_or_duration() -> None:
    callbacks, mock_http = _make_callbacks()
    callbacks.report_results(_results())

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert "state" not in event
    assert "message" not in event
    assert "duration_seconds" not in event


def test_report_results_always_includes_provider_id() -> None:
    callbacks, mock_http = _make_callbacks(provider_id=None)
    callbacks.report_results(_results())

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert "provider_id" in event
    assert event["provider_id"] == ""


# ---------------------------------------------------------------------------
# additional_info payload tests
# ---------------------------------------------------------------------------


def test_report_results_includes_additional_info_when_set() -> None:
    callbacks, mock_http = _make_callbacks()
    results = _results()
    results.additional_info = {
        "dataset_sha": "abc123",
        "zero_shot": "0.85",
    }
    callbacks.report_results(results)

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert event["additional_info"] == {
        "dataset_sha": "abc123",
        "zero_shot": "0.85",
    }


def test_report_results_omits_additional_info_when_not_set() -> None:
    callbacks, mock_http = _make_callbacks()
    callbacks.report_results(_results())

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert "additional_info" not in event


def test_report_results_additional_info_passes_arbitrary_keys() -> None:
    callbacks, mock_http = _make_callbacks()
    results = _results()
    results.additional_info = {
        "alt_prompting": "0.91",
        "alt_prompting_description": "5-Shot CoT",
        "custom_key": 42,
    }
    callbacks.report_results(results)

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert event["additional_info"] == {
        "alt_prompting": "0.91",
        "alt_prompting_description": "5-Shot CoT",
        "custom_key": 42,
    }


def test_report_results_additional_info_passes_nested_values() -> None:
    callbacks, mock_http = _make_callbacks()
    results = _results()
    results.additional_info = {
        "config": {"shots": 5, "cot": True},
        "tags": ["a", "b"],
    }
    callbacks.report_results(results)

    body = mock_http.post.call_args.kwargs["json"]
    event = body["benchmark_status_event"]
    assert event["additional_info"] == {
        "config": {"shots": 5, "cot": True},
        "tags": ["a", "b"],
    }
