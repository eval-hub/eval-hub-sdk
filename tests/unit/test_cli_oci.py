"""OCI reporting preserves evaluation and structured-output contracts."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from evalhub.cli.main import main
from evalhub.client.job_logs import JobLogUpdate
from evalhub.models import EvaluationJob, JobStatus

from .test_oci_job_api import ARTIFACT, job_data, request_data

pytestmark = pytest.mark.unit


@pytest.fixture()
def cli_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    monkeypatch.setenv("EVALHUB_CONFIG", str(tmp_path / "profile.yaml"))
    client = MagicMock()
    job = EvaluationJob.model_validate(job_data("failed"))
    client.jobs.get.return_value = job
    client.jobs.submit.return_value = job
    client.jobs.wait_for_completion.return_value = job
    client.jobs.list.return_value = [job]
    with patch("evalhub.cli.main.get_client", return_value=client):
        yield client


@pytest.mark.parametrize("extension", ["yaml", "json"])
@pytest.mark.parametrize(
    "signing", [None, {"type": "key-pair"}, {"type": "key-pair", "secret_ref": " key "}]
)
def test_config_signing_and_precedence(
    cli_client: MagicMock, tmp_path: Path, extension: str, signing: Any
) -> None:
    path = tmp_path / f"request.{extension}"
    data = request_data(signing)
    path.write_text(json.dumps(data) if extension == "json" else yaml.safe_dump(data))
    result = CliRunner().invoke(
        main,
        [
            "eval",
            "run",
            "--config",
            str(path),
            "--oci-host",
            "ignored",
            "--model-name",
            "ignored",
        ],
    )
    assert result.exit_code == 0, result.output
    request = cli_client.jobs.submit.call_args.args[0]
    assert request.model.name == "test"
    body = request.model_dump(exclude_none=True)["exports"]["oci"]
    assert body["coordinates"]["oci_host"] == "quay.io"
    if signing is None:
        assert "signing" not in body
    else:
        assert body["signing"] == signing


@pytest.mark.parametrize("extension", ["yaml", "json"])
@pytest.mark.parametrize(
    "signing",
    [
        {},
        {"type": None},
        {"type": "keyless"},
        {"secret_ref": "key"},
        {"type": "key-pair", "secret_ref": ""},
        {"type": "key-pair", "secret_ref": " \t"},
    ],
)
def test_invalid_config_before_submission(
    cli_client: MagicMock, tmp_path: Path, extension: str, signing: Any
) -> None:
    path = tmp_path / f"request.{extension}"
    data = request_data(signing)
    path.write_text(json.dumps(data) if extension == "json" else yaml.safe_dump(data))
    result = CliRunner().invoke(main, ["eval", "run", "--config", str(path)])
    assert result.exit_code == 1
    assert "Invalid job configuration" in result.output
    assert "signing" in result.output
    cli_client.jobs.submit.assert_not_called()


@pytest.mark.parametrize("command", ["status", "results"])
@pytest.mark.parametrize(
    "state",
    ["completed", "failed", "generating_evaluation_card", "future_server_state"],
)
def test_human_oci_output_without_benchmarks(
    cli_client: MagicMock, command: str, state: str
) -> None:
    cli_client.jobs.get.return_value = EvaluationJob.model_validate(job_data(state))
    result = CliRunner().invoke(main, ["eval", command, "job-id"])
    assert result.exit_code == 0
    assert "Server OCI outcome" not in result.output
    assert "Message code: SERVER_CODE" not in result.output
    assert "Origin: server" not in result.output
    assert ARTIFACT["oci_digest"] not in result.output
    assert ARTIFACT["oci_reference"] not in result.output
    assert "Configured signing: key-pair" not in result.output
    assert "Reported signing result" not in result.output
    assert "Evaluation bundle" not in result.output
    if command == "status":
        assert "State:   completed" in result.output
        assert result.output.endswith(f"\nOCI:\n  State: {state}\n")
    else:
        assert "OCI:" not in result.output
        assert result.output == "No benchmark results available.\n"


@pytest.mark.parametrize("command", ["status", "results"])
def test_human_future_outputs_and_metrics_survive_failure(
    cli_client: MagicMock, command: str
) -> None:
    data = job_data("failed")
    data["results"]["oci"].update(
        evaluation_bundle=ARTIFACT, signing={"type": "key-pair"}
    )
    data["results"]["benchmarks"] = [
        {"id": "bench", "provider_id": "provider", "metrics": {"accuracy": 0.9}}
    ]
    data["results"]["mlflow_experiment_url"] = "https://mlflow/experiment"
    cli_client.jobs.get.return_value = EvaluationJob.model_validate(data)
    result = CliRunner().invoke(main, ["eval", command, "job-id"])
    assert result.exit_code == 0, result.output
    if command == "results":
        for text in ["accuracy", "0.9", "https://mlflow/experiment"]:
            assert text in result.output
        assert "OCI:" not in result.output
        assert "State: failed" not in result.output
    else:
        assert result.output.endswith("\nOCI:\n  State: failed\n")
    assert ARTIFACT["oci_digest"] not in result.output
    assert ARTIFACT["oci_reference"] not in result.output
    assert "Reported signing result" not in result.output


@pytest.mark.parametrize("kind", ["configured", "disabled", "empty_results"])
def test_absent_oci_status(cli_client: MagicMock, kind: str) -> None:
    data = job_data()
    data.pop("results")
    if kind == "disabled":
        data.pop("exports")
    if kind == "empty_results":
        data.pop("exports")
        data["results"] = {"oci": {}}
    cli_client.jobs.get.return_value = EvaluationJob.model_validate(data)
    result = CliRunner().invoke(main, ["eval", "status", "job-id"])
    if kind == "disabled":
        assert "OCI:" not in result.output
    else:
        assert "State: unavailable" in result.output
        assert "Reported signing result" not in result.output


@pytest.mark.parametrize("output_format", ["json", "yaml"])
def test_full_job_output_and_run_stdout(
    cli_client: MagicMock, tmp_path: Path, output_format: str
) -> None:
    runner = CliRunner()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request_data()))
    for args in [
        ["eval", "status", "job-id"],
        ["eval", "run", "--config", str(path), "--wait"],
    ]:
        result = runner.invoke(main, [*args, "--format", output_format])
        assert result.exit_code == 0, result.output
        data = (
            json.loads(result.stdout)
            if output_format == "json"
            else yaml.safe_load(result.stdout)
        )
        assert len(data) == 1
        assert data[0]["status"]["state"] == "completed"
        assert data[0]["results"]["oci"] == job_data("failed")["results"]["oci"]
        assert data[0]["exports"]["oci"]["signing"]["secret_ref"] == "signing-key"
        assert "evaluation_bundle" not in data[0]["results"]["oci"]
    assert "Job submitted" in result.stderr
    assert "finished with state: completed" in result.stderr


def test_list_summary_keeps_evaluation_filter(cli_client: MagicMock) -> None:
    absent = job_data()
    absent.pop("results")
    cli_client.jobs.list.return_value.append(EvaluationJob.model_validate(absent))
    result = CliRunner().invoke(
        main, ["eval", "status", "--status", "completed", "--format", "json"]
    )
    data = json.loads(result.stdout)
    assert data[0]["state"] == "completed" and data[0]["oci_state"] == "failed"
    assert data[1]["oci_state"] == ""
    assert "resource" not in data[0] and "results" not in data[0]
    cli_client.jobs.list.assert_called_once_with(status=JobStatus.COMPLETED, limit=None)


@pytest.mark.parametrize("output_format", ["json", "yaml", "csv"])
@pytest.mark.parametrize(
    "benchmarks",
    [
        None,
        [],
        [{"id": "bench", "provider_id": "provider", "metrics": {}}],
        [{"id": "bench", "provider_id": "provider", "metrics": {"accuracy": 0.9}}],
    ],
)
def test_benchmark_structured_outputs_stay_homogeneous(
    cli_client: MagicMock, output_format: str, benchmarks: Any
) -> None:
    data = job_data("failed")
    if benchmarks is not None:
        data["results"]["benchmarks"] = benchmarks
    data["results"]["mlflow_experiment_url"] = "https://mlflow/experiment"
    cli_client.jobs.get.return_value = EvaluationJob.model_validate(data)
    result = CliRunner().invoke(
        main, ["eval", "results", "job-id", "--format", output_format]
    )
    assert result.exit_code == 0, result.output
    if output_format == "csv":
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert len(rows) == (1 if benchmarks and benchmarks[0]["metrics"] else 0)
        if rows:
            assert set(rows[0]) == {"benchmark", "provider", "metric", "value"}
        assert "OCI" not in result.stdout and "MLflow" not in result.stdout
    else:
        decoded = (
            json.loads(result.stdout)
            if output_format == "json"
            else yaml.safe_load(result.stdout)
        )
        assert isinstance(decoded, list)
        assert len(decoded) == len(benchmarks or [])
        assert all("oci" not in benchmark for benchmark in decoded)


@pytest.mark.parametrize("state", ["pending", "generating_evaluation_card", "failed"])
def test_watch_stops_at_evaluation_completion(
    cli_client: MagicMock, state: str
) -> None:
    cli_client.jobs.get.return_value = EvaluationJob.model_validate(job_data(state))
    with patch("evalhub.cli.main.time.sleep") as sleep:
        result = CliRunner().invoke(main, ["eval", "status", "job-id", "--watch"])
    assert result.exit_code == 0
    sleep.assert_not_called()
    assert f"State: {state}" in result.output
    assert "State:   completed" in result.output


def test_log_watch_final_summary_reports_oci(
    cli_client: MagicMock, tmp_path: Path
) -> None:
    job = EvaluationJob.model_validate(job_data("generating_evaluation_card"))
    cli_client.jobs.watch_logs.return_value = [
        JobLogUpdate(logs="adapter log\n", job=job)
    ]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request_data()))
    result = CliRunner().invoke(main, ["eval", "run", "--config", str(path), "--watch"])
    assert result.exit_code == 0
    assert "adapter log" in result.stdout
    assert "finished with state: completed" in result.stderr
    assert result.stderr.endswith("\nOCI:\n  State: generating_evaluation_card\n")
