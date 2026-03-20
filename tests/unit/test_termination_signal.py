"""Unit tests for termination file signal in DefaultCallbacks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # noqa: F401 (used by pytest fixtures)
from evalhub.adapter.callbacks import DefaultCallbacks
from evalhub.adapter.config import DEFAULT_TERMINATION_FILE_PATH, EvalHubMode
from evalhub.adapter.models import JobResults
from evalhub.adapter.models.job import EvaluationResult


def _make_callbacks(termination_file_path: str | None = None) -> DefaultCallbacks:
    """Create a minimal DefaultCallbacks for testing."""
    with patch("evalhub.adapter.callbacks.OCIArtifactPersister"):
        return DefaultCallbacks(
            job_id="job-1",
            benchmark_id="bench-1",
            termination_file_path=termination_file_path,
        )


def _make_results() -> JobResults:
    """Create a minimal JobResults for testing."""
    return JobResults(
        id="job-1",
        benchmark_id="bench-1",
        benchmark_index=0,
        model_name="test-model",
        results=[EvaluationResult(metric_name="accuracy", metric_value=0.95)],
        num_examples_evaluated=100,
        duration_seconds=10.0,
        completed_at=datetime.now(tz=UTC),
        evaluation_metadata={},
    )


class TestSignalTermination:
    """Tests for _signal_termination method."""

    def test_writes_success_file(self, tmp_path: Path) -> None:
        term_file = tmp_path / "terminated"
        cb = _make_callbacks(str(term_file))
        cb._signal_termination()
        assert term_file.read_text() == "0"

    def test_writes_error_file(self, tmp_path: Path) -> None:
        term_file = tmp_path / "terminated"
        cb = _make_callbacks(str(term_file))
        cb._signal_termination(error="something went wrong")
        assert term_file.read_text() == "something went wrong"

    def test_noop_when_path_is_none(self) -> None:
        cb = _make_callbacks(termination_file_path=None)
        # Should not raise
        cb._signal_termination()

    def test_logs_on_write_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = _make_callbacks("/nonexistent/dir/terminated")
        with caplog.at_level(logging.ERROR):
            cb._signal_termination()
        assert "Failed to write termination file" in caplog.text


class TestReportResultsTermination:
    """Tests for termination signal integration in report_results."""

    def test_report_results_calls_signal_termination(self, tmp_path: Path) -> None:
        term_file = tmp_path / "terminated"
        cb = _make_callbacks(str(term_file))
        cb.report_results(_make_results())
        assert term_file.exists()
        assert term_file.read_text() == "0"

    def test_report_results_captures_sidecar_error(self, tmp_path: Path) -> None:
        term_file = tmp_path / "terminated"
        cb = _make_callbacks(str(term_file))
        # Simulate sidecar being available but failing
        cb.sidecar_url = "http://localhost:8080"
        cb._httpx_available = True
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        http_error = type("HTTPStatusError", (Exception,), {})
        exc = http_error("Service Unavailable")
        exc.response = mock_response
        mock_client.post.side_effect = exc
        cb._http_client = mock_client
        cb.httpx = MagicMock()
        cb.httpx.HTTPStatusError = http_error

        cb.report_results(_make_results())
        content = term_file.read_text()
        assert "Unavailable" in content
        assert content != "0"


class TestFromAdapterTermination:
    """Tests for termination_file_path in from_adapter factory."""

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
        adapter.job_spec.callback_url = None
        return adapter

    @patch("evalhub.adapter.callbacks.OCIArtifactPersister")
    def test_k8s_mode_sets_termination_path(self, _mock_persister: MagicMock) -> None:
        adapter = self._mock_adapter(EvalHubMode.K8S)
        cb = DefaultCallbacks.from_adapter(adapter)
        assert cb._termination_file_path == DEFAULT_TERMINATION_FILE_PATH

    @patch("evalhub.adapter.callbacks.OCIArtifactPersister")
    def test_local_mode_no_termination_path(self, _mock_persister: MagicMock) -> None:
        adapter = self._mock_adapter(EvalHubMode.LOCAL)
        cb = DefaultCallbacks.from_adapter(adapter)
        assert cb._termination_file_path is None
