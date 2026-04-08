"""Tests for DefaultCallbacks POST /events payload (mlflow_run_id) and from_adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from evalhub.adapter.callbacks import DefaultCallbacks
from evalhub.adapter.config import EvalHubMode
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


def test_report_results_includes_x_tenant_when_job_tenant_set() -> None:
    """job.json ``tenant`` is forwarded as X-Tenant (aligns with eval-hub job spec)."""
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
            tenant="team-a",
            sidecar_url="http://evalhub:8080",
            insecure=True,
        )

    callbacks.report_results(_results())

    mock_http.post.assert_called_once()
    headers = mock_http.post.call_args.kwargs["headers"]
    assert headers["X-Tenant"] == "team-a"


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


class TestFromAdapterCA:
    """AdapterSettings.ca_bundle_path is passed through from_adapter."""

    def _mock_adapter(self, mode: EvalHubMode) -> MagicMock:
        adapter = MagicMock()
        adapter.settings.mode = mode
        adapter.settings.evalhub_insecure = False
        adapter.settings.oci_auth_config_path = None
        adapter.settings.oci_insecure = False
        adapter.settings.mlflow_backend = "odh"
        adapter.job_spec.id = "job-1"
        adapter.job_spec.provider_id = "provider-1"
        adapter.job_spec.benchmark_id = "bench-1"
        adapter.job_spec.benchmark_index = 0
        adapter.job_spec.tenant = None
        adapter.job_spec.callback_url = None
        adapter.settings.ca_bundle_path = None
        return adapter

    @patch("evalhub.adapter.callbacks.OCIArtifactPersister")
    def test_from_adapter_forwards_ca_bundle_path(
        self, _mock_persister: MagicMock, tmp_path: Path
    ) -> None:
        ca_file = tmp_path / "ca.pem"
        ca_file.write_text("fake-ca-bundle")
        adapter = self._mock_adapter(EvalHubMode.K8S)
        adapter.settings.ca_bundle_path = ca_file
        cb = DefaultCallbacks.from_adapter(adapter)
        assert cb._ca_bundle == ca_file
