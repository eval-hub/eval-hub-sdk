"""Tests for DefaultCallbacks POST /events payload (mlflow_run_id)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from evalhub.adapter.callbacks import DefaultCallbacks
from evalhub.adapter.models.job import JobResults
from evalhub.models.api import EvaluationResult


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


def test_report_results_sends_mlflow_run_id_when_set_on_job_results() -> None:
    mock_http = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    mock_http.post.return_value = resp

    with patch.object(DefaultCallbacks, "_create_http_client", return_value=mock_http):
        callbacks = DefaultCallbacks(
            job_id="job-1",
            benchmark_id="arc_easy",
            provider_id="lm_evaluation_harness",
            benchmark_index=0,
            sidecar_url="http://evalhub:8080",
            insecure=True,
        )

    callbacks.report_results(_results(mlflow_run_id="mlflow-run-abc"))

    mock_http.post.assert_called_once()
    body = mock_http.post.call_args.kwargs["json"]
    assert body["benchmark_status_event"]["mlflow_run_id"] == "mlflow-run-abc"


def test_report_results_omits_mlflow_run_id_when_not_set() -> None:
    mock_http = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    mock_http.post.return_value = resp

    with patch.object(DefaultCallbacks, "_create_http_client", return_value=mock_http):
        callbacks = DefaultCallbacks(
            job_id="job-1",
            benchmark_id="arc_easy",
            benchmark_index=0,
            sidecar_url="http://evalhub:8080",
            insecure=True,
        )

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
