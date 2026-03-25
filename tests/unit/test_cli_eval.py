"""Unit tests for EvalHub CLI eval commands."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from evalhub.cli.main import main
from evalhub.models.api import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkStatus,
    EvaluationJob,
    EvaluationJobResource,
    EvaluationJobResults,
    EvaluationJobStatus,
    JobStatus,
    ModelConfig,
)

NOW = datetime(2026, 3, 23, 12, 0, 0, tzinfo=UTC)


def _make_job(
    job_id: str = "eval-123",
    name: str = "test-eval",
    state: JobStatus = JobStatus.PENDING,
    benchmarks_cfg: list[BenchmarkConfig] | None = None,
    results: EvaluationJobResults | None = None,
    benchmark_statuses: list[BenchmarkStatus] | None = None,
) -> EvaluationJob:
    if benchmarks_cfg is None:
        benchmarks_cfg = [BenchmarkConfig(id="mmlu", provider_id="lm_eval")]
    return EvaluationJob(
        resource=EvaluationJobResource(id=job_id, created_at=NOW),
        status=EvaluationJobStatus(
            state=state,
            benchmarks=benchmark_statuses or [],
        ),
        results=results,
        name=name,
        model=ModelConfig(url="http://vllm:8000/v1", name="llama3"),
        benchmarks=benchmarks_cfg,
    )


@pytest.fixture()
def config_file(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "config.yaml"
    os.environ["EVALHUB_CONFIG"] = str(path)
    yield path
    os.environ.pop("EVALHUB_CONFIG", None)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    client.jobs = MagicMock()
    return client


# --- eval run ---


class TestEvalRun:
    def test_run_with_config_file(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg = {
            "name": "my-eval",
            "model": {"url": "http://vllm:8000/v1", "name": "llama3"},
            "benchmarks": [{"id": "mmlu", "provider_id": "lm_eval"}],
        }
        cfg_path = tmp_path / "eval.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        mock_client.jobs.submit.return_value = _make_job()
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "run", "--config", str(cfg_path)])
        assert result.exit_code == 0
        assert "Job submitted: eval-123" in result.output
        mock_client.jobs.submit.assert_called_once()

    def test_run_with_json_config(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg = {
            "name": "my-eval",
            "model": {"url": "http://vllm:8000/v1", "name": "llama3"},
            "benchmarks": [{"id": "mmlu", "provider_id": "lm_eval"}],
        }
        cfg_path = tmp_path / "eval.json"
        cfg_path.write_text(json.dumps(cfg))

        mock_client.jobs.submit.return_value = _make_job()
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "run", "--config", str(cfg_path)])
        assert result.exit_code == 0
        assert "Job submitted: eval-123" in result.output

    def test_run_with_inline_flags(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.jobs.submit.return_value = _make_job()
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "eval",
                    "run",
                    "--name",
                    "inline-eval",
                    "--model-url",
                    "http://vllm:8000/v1",
                    "--model-name",
                    "llama3",
                    "--provider",
                    "lm_eval",
                    "-b",
                    "mmlu",
                    "-b",
                    "hellaswag",
                ],
            )
        assert result.exit_code == 0
        assert "Job submitted: eval-123" in result.output
        req = mock_client.jobs.submit.call_args[0][0]
        assert req.name == "inline-eval"
        assert len(req.benchmarks) == 2

    def test_run_missing_required_flags(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "eval",
                    "run",
                    "--name",
                    "incomplete",
                ],
            )
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "Error" in result.output

    def test_run_with_wait(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg = {
            "name": "my-eval",
            "model": {"url": "http://vllm:8000/v1", "name": "llama3"},
            "benchmarks": [{"id": "mmlu", "provider_id": "lm_eval"}],
        }
        cfg_path = tmp_path / "eval.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        submitted = _make_job()
        completed = _make_job(state=JobStatus.COMPLETED)
        mock_client.jobs.submit.return_value = submitted
        mock_client.jobs.wait_for_completion.return_value = completed

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "eval",
                    "run",
                    "--config",
                    str(cfg_path),
                    "--wait",
                ],
            )
        assert result.exit_code == 0
        assert "Waiting for job" in result.output
        assert "completed" in result.output
        mock_client.jobs.wait_for_completion.assert_called_once()

    def test_run_with_wait_failed(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg = {
            "name": "my-eval",
            "model": {"url": "http://vllm:8000/v1", "name": "llama3"},
            "benchmarks": [{"id": "mmlu", "provider_id": "lm_eval"}],
        }
        cfg_path = tmp_path / "eval.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        submitted = _make_job()
        failed = _make_job(state=JobStatus.FAILED)
        mock_client.jobs.submit.return_value = submitted
        mock_client.jobs.wait_for_completion.return_value = failed

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "eval",
                    "run",
                    "--config",
                    str(cfg_path),
                    "--wait",
                ],
            )
        assert result.exit_code == 1

    def test_run_json_output(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg = {
            "name": "my-eval",
            "model": {"url": "http://vllm:8000/v1", "name": "llama3"},
            "benchmarks": [{"id": "mmlu", "provider_id": "lm_eval"}],
        }
        cfg_path = tmp_path / "eval.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        mock_client.jobs.submit.return_value = _make_job()
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "eval",
                    "run",
                    "--config",
                    str(cfg_path),
                    "--format",
                    "json",
                ],
            )
        assert result.exit_code == 0
        assert "eval-123" in result.output


# --- eval status ---


class TestEvalStatus:
    def test_list_all_jobs(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.jobs.list.return_value = [
            _make_job(job_id="eval-1", name="eval-one", state=JobStatus.RUNNING),
            _make_job(job_id="eval-2", name="eval-two", state=JobStatus.COMPLETED),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "status"])
        assert result.exit_code == 0
        assert "running" in result.output
        assert "completed" in result.output or "complet" in result.output

    def test_list_with_status_filter(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.jobs.list.return_value = [
            _make_job(job_id="eval-1", name="eval-one", state=JobStatus.RUNNING),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "status", "--status", "running"])
        assert result.exit_code == 0
        assert "running" in result.output
        mock_client.jobs.list.assert_called_once_with(
            status=JobStatus.RUNNING, limit=None
        )

    def test_list_with_limit(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.jobs.list.return_value = []
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "status", "--limit", "5"])
        assert result.exit_code == 0
        mock_client.jobs.list.assert_called_once_with(status=None, limit=5)

    def test_single_job_detail(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.jobs.get.return_value = _make_job(
            state=JobStatus.RUNNING,
            benchmark_statuses=[
                BenchmarkStatus(
                    id="mmlu",
                    provider_id="lm_eval",
                    status=JobStatus.RUNNING,
                ),
            ],
        )
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "status", "eval-123"])
        assert result.exit_code == 0
        assert "eval-123" in result.output
        assert "running" in result.output
        assert "llama3" in result.output

    def test_single_job_json(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.jobs.get.return_value = _make_job(state=JobStatus.COMPLETED)
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "eval",
                    "status",
                    "eval-123",
                    "--format",
                    "json",
                ],
            )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed[0]["name"] == "test-eval"

    def test_list_empty(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.jobs.list.return_value = []
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "status"])
        assert result.exit_code == 0
        assert "no data" in result.output


# --- eval results ---


class TestEvalResults:
    def test_results_table(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        job = _make_job(
            state=JobStatus.COMPLETED,
            results=EvaluationJobResults(
                benchmarks=[
                    BenchmarkResult(
                        id="mmlu",
                        provider_id="lm_eval",
                        metrics={"accuracy": 0.85, "f1": 0.82},
                    ),
                ],
            ),
        )
        mock_client.jobs.get.return_value = job
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "results", "eval-123"])
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert "0.85" in result.output

    def test_results_json(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        job = _make_job(
            state=JobStatus.COMPLETED,
            results=EvaluationJobResults(
                benchmarks=[
                    BenchmarkResult(
                        id="mmlu",
                        provider_id="lm_eval",
                        metrics={"accuracy": 0.85},
                    ),
                ],
            ),
        )
        mock_client.jobs.get.return_value = job
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "eval",
                    "results",
                    "eval-123",
                    "--format",
                    "json",
                ],
            )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed[0]["id"] == "mmlu"
        assert parsed[0]["metrics"]["accuracy"] == 0.85

    def test_results_no_results(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        job = _make_job(state=JobStatus.RUNNING, results=None)
        mock_client.jobs.get.return_value = job
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "results", "eval-123"])
        assert result.exit_code == 0
        assert "No results available" in result.output

    def test_results_warns_incomplete(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        job = _make_job(
            state=JobStatus.RUNNING,
            results=EvaluationJobResults(
                benchmarks=[
                    BenchmarkResult(
                        id="mmlu", provider_id="lm_eval", metrics={"accuracy": 0.5}
                    ),
                ],
            ),
        )
        mock_client.jobs.get.return_value = job
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "results", "eval-123"])
        assert result.exit_code == 0
        assert "accuracy" in result.output

    def test_results_mlflow_url(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        job = _make_job(
            state=JobStatus.COMPLETED,
            results=EvaluationJobResults(
                benchmarks=[
                    BenchmarkResult(
                        id="mmlu", provider_id="lm_eval", metrics={"accuracy": 0.9}
                    ),
                ],
                mlflow_experiment_url="https://mlflow.example.com/exp/1",
            ),
        )
        mock_client.jobs.get.return_value = job
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "results", "eval-123"])
        assert result.exit_code == 0
        assert "https://mlflow.example.com/exp/1" in result.output

    def test_results_multiple_benchmarks(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        job = _make_job(
            state=JobStatus.COMPLETED,
            results=EvaluationJobResults(
                benchmarks=[
                    BenchmarkResult(
                        id="mmlu", provider_id="lm_eval", metrics={"accuracy": 0.85}
                    ),
                    BenchmarkResult(
                        id="hellaswag",
                        provider_id="lm_eval",
                        metrics={"accuracy": 0.78},
                    ),
                ],
            ),
        )
        mock_client.jobs.get.return_value = job
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "results", "eval-123"])
        assert result.exit_code == 0
        assert "mmlu" in result.output
        assert "hellaswag" in result.output


# --- eval cancel ---


class TestEvalCancel:
    def test_cancel_job(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.jobs.cancel.return_value = True
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "cancel", "eval-123"], input="y\n")
        assert result.exit_code == 0
        assert "cancelled" in result.output
        mock_client.jobs.cancel.assert_called_once_with("eval-123", hard_delete=False)

    def test_cancel_hard_delete(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.jobs.cancel.return_value = True
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["eval", "cancel", "eval-123", "--hard-delete"], input="y\n"
            )
        assert result.exit_code == 0
        assert "deleted" in result.output
        mock_client.jobs.cancel.assert_called_once_with("eval-123", hard_delete=True)

    def test_cancel_aborted(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["eval", "cancel", "eval-123"], input="n\n")
        assert result.exit_code != 0
        mock_client.jobs.cancel.assert_not_called()


# --- eval help ---


class TestEvalHelp:
    def test_eval_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["eval", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "status" in result.output
        assert "results" in result.output
        assert "cancel" in result.output

    def test_eval_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["eval", "run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--model-url" in result.output
        assert "--wait" in result.output

    def test_eval_status_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["eval", "status", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output
        assert "--watch" in result.output

    def test_eval_results_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["eval", "results", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
