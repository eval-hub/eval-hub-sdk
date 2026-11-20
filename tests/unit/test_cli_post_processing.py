"""CLI contracts for standalone post-processing."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import click
import httpx
import pytest
import yaml
from click.testing import CliRunner, Result
from evalhub.cli.main import (
    _output_post_processing,
    _output_post_processing_results,
    _post_processing_result_records,
    main,
)
from evalhub.models import PostProcessingResource, StandalonePostProcessingRequest

pytestmark = pytest.mark.unit

CONFIG: dict[str, Any] = {
    "name": "interval-job",
    "hardware_config": {"cpu": {"request": "1"}},
    "operation_order": ["confidence_interval"],
    "operations": {
        "confidence_interval": {
            "results_data_ref": {"eval_job": {"id": "eval-source"}},
            "calibration_data_ref": [
                {
                    "pvc": {"claim_name": "calibration-data"},
                    "data_config": {
                        "format": "jsonl",
                        "columns": {"label": "label", "prediction": "prediction"},
                    },
                }
            ],
            "significance_level": 0.05,
        }
    },
}
INTERVAL = {
    "lower": 0.0,
    "upper": 0.8,
    "point_estimate": 0.4,
    "n_labeled": 0,
    "n_unlabeled": 10,
    "alpha": 0.03,
    "method": "ppi",
    "lambda": 0,
}
BENCHMARK = {
    "id": "accuracy",
    "provider_id": "inspect",
    "benchmark_index": 0,
    "confidence_interval": INTERVAL,
}


def make_job(state: str = "completed", **overrides: Any) -> PostProcessingResource:
    data = {
        **copy.deepcopy(CONFIG),
        "resource": {
            "id": "pp-123",
            "created_at": "2026-11-18T12:00:00Z",
            "updated_at": "2026-11-18T13:00:00Z",
        },
        "status": {"state": state},
        **overrides,
    }
    return PostProcessingResource.model_validate(data)


@pytest.fixture()
def client() -> MagicMock:
    client = MagicMock()
    client.post_processing.submit.return_value = make_job("pending")
    client.post_processing.get.return_value = make_job()
    return client


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "post-processing.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    return path


def invoke(client: MagicMock, config_path: Path, command: str, *options: str) -> Result:
    args = ["post-processing", command]
    args += ["--config", str(config_path)] if command == "run" else ["pp-123"]
    with patch("evalhub.cli.main.get_client", return_value=client):
        return CliRunner().invoke(main, [*args, *options])


@pytest.mark.parametrize("command", ["run", "status", "results", "delete"])
def test_help(command: str) -> None:
    result = CliRunner().invoke(main, ["post-processing", command, "--help"])
    assert result.exit_code == 0
    assert command in result.stdout
    assert "--watch" not in result.stdout
    for option in ("--wait", "--timeout", "--poll-interval"):
        assert (option in result.stdout) == (command == "run")


def test_group_registration() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert "post-processing" in result.stdout


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_submission(client: MagicMock, tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"request{suffix}"
    path.write_text(json.dumps(CONFIG) if suffix == ".json" else yaml.safe_dump(CONFIG))
    result = invoke(client, path, "run", "--format", "json")
    assert result.exit_code == 0, result.output
    request = client.post_processing.submit.call_args.args[0]
    assert isinstance(request, StandalonePostProcessingRequest)
    assert request.model_dump(mode="json", exclude_none=True) == CONFIG
    assert json.loads(result.stdout)[0]["resource"]["id"] == "pp-123"
    assert "submitted: pp-123" in result.stderr
    client.post_processing.get.assert_not_called()


@pytest.mark.parametrize("command", ["run", "status"])
@pytest.mark.parametrize("fmt", ["json", "yaml"])
@pytest.mark.parametrize("external", [False, True])
def test_full_resource(
    client: MagicMock, config_path: Path, command: str, fmt: str, external: bool
) -> None:
    job = make_job(results={"confidence_interval": INTERVAL})
    if external:
        job = external_job("s3", {"bucket": "b", "key": "k", "secret_ref": "s"})
    client.post_processing.submit.return_value = job
    client.post_processing.get.return_value = job
    result = invoke(client, config_path, command, "--format", fmt)
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(result.stdout)
    assert data == [job.model_dump(mode="json")]
    assert data[0]["status"]["warning_message"] is None
    assert data[0]["results"]["confidence_interval"]["lambda"] == 0


@pytest.mark.parametrize(
    ("command", "options"),
    [("run", ["--format", "table"]), ("status", []), ("status", ["--format", "table"])],
)
def test_resource_table(
    client: MagicMock, config_path: Path, command: str, options: list[str]
) -> None:
    job = make_job(
        status={
            "state": "failed",
            "started_at": "2026-11-18T12:01:00Z",
            "completed_at": "2026-11-18T12:02:00Z",
            "error_message": {
                "message_code": "failed",
                "message": "computation failed",
            },
            "warning_message": {"message_code": "partial", "message": "partial data"},
        },
        results={"confidence_interval": INTERVAL},
    )
    client.post_processing.submit.return_value = job
    client.post_processing.get.return_value = job
    result = invoke(client, config_path, command, *options)
    assert result.exit_code == 0, result.output
    for value in (
        "pp-123",
        "interval-job",
        "failed",
        "Created",
        "Started",
        "Completed",
        "confidence_interval",
        "Source evaluation",
        "eval-source",
        "computation failed",
        "partial data",
    ):
        assert value in result.stdout
    for value in (
        "updated",
        "13:00",
        "lower",
        "significance_level",
        "calibration-data",
    ):
        assert value not in result.stdout
    assert "submitted" not in result.stdout


def external_job(source: str, config: dict[str, Any]) -> PostProcessingResource:
    data = copy.deepcopy(CONFIG)
    operation = data["operations"]["confidence_interval"]
    operation["results_data_ref"] = {source: config}
    operation["primary_score"] = {"metric": "accuracy"}
    return make_job(**data, results={"confidence_interval": INTERVAL})


@pytest.mark.parametrize(
    ("source", "config", "label"),
    [
        ("s3", {"bucket": "b", "key": "k", "secret_ref": "s"}, "S3"),
        ("pvc", {"claim_name": "data"}, "PVC"),
        ("git", {"url": "https://example.com/data", "ref": "main"}, "Git"),
        ("hf", {"repo_id": "org/data"}, "Hugging Face"),
        ("mlflow", {"run_id": "r", "artifact_path": "p"}, "MLflow"),
        (
            "oci",
            {
                "coordinates": {"oci_host": "quay.io", "oci_repository": "org/data"},
                "artifact_path": "p",
            },
            "OCI",
        ),
    ],
)
@pytest.mark.parametrize("command", ["run", "status"])
def test_external_source_labels(
    client: MagicMock,
    config_path: Path,
    command: str,
    source: str,
    config: dict[str, Any],
    label: str,
) -> None:
    job = external_job(source, config)
    client.post_processing.submit.return_value = job
    client.post_processing.get.return_value = job
    result = invoke(client, config_path, command, "--format", "table")
    assert result.exit_code == 0, result.output
    assert label in result.stdout
    assert "Source evaluation" not in result.stdout


def test_absent_status_fields(client: MagicMock, config_path: Path) -> None:
    client.post_processing.get.return_value = make_job(name=None)
    result = invoke(client, config_path, "status")
    for label in ("Name", "Started", "Completed", "Error", "Warning"):
        assert label not in result.stdout
    client.post_processing.get.assert_called_once_with("pp-123")


@pytest.mark.parametrize("fmt", ["json", "yaml"])
@pytest.mark.parametrize(
    "shape", ["benchmark", "aggregate", "mixed", "empty-benchmarks"]
)
def test_result_records(
    client: MagicMock, config_path: Path, fmt: str, shape: str
) -> None:
    aggregate = {**INTERVAL, "lower": 0.2, "upper": 0.9}
    results: dict[str, Any] = {}
    expected = []
    if shape in ("benchmark", "mixed"):
        results["benchmarks"] = [BENCHMARK, {**BENCHMARK, "benchmark_index": 1}]
        expected.extend(results["benchmarks"])
    if shape != "benchmark":
        results["confidence_interval"] = aggregate
        expected.append({"confidence_interval": aggregate})
    if shape == "empty-benchmarks":
        results["benchmarks"] = []
    client.post_processing.get.return_value = make_job(results=results)
    result = invoke(client, config_path, "results", "--format", fmt)
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.stdout) == expected
    assert result.stderr == ""
    client.post_processing.get.assert_called_once_with("pp-123")


@pytest.mark.parametrize("fmt", ["table", "json", "yaml"])
@pytest.mark.parametrize(
    "results",
    [
        None,
        {},
        cast(dict[str, Any], {"benchmarks": []}),
        {"confidence_interval": None},
    ],
)
def test_empty_results(
    client: MagicMock, config_path: Path, fmt: str, results: Any
) -> None:
    client.post_processing.get.return_value = make_job(results=results)
    result = invoke(client, config_path, "results", "--format", fmt)
    assert result.exit_code == 0
    assert "no results available" in result.stderr.lower()
    assert "Warning:" not in result.stderr
    if fmt == "table":
        assert result.stdout == ""
        assert result.output == "No results available.\n"
    else:
        assert yaml.safe_load(result.stdout) == []


@pytest.mark.parametrize("fmt", ["table", "json", "yaml"])
def test_partial_results_and_diagnostics(
    client: MagicMock, config_path: Path, fmt: str
) -> None:
    client.post_processing.get.return_value = make_job(
        status={
            "state": "failed",
            "error_message": {"message_code": "failed", "message": "bad computation"},
            "warning_message": {
                "message_code": "partial",
                "message": "limited calibration",
            },
        },
        results={"confidence_interval": {"lower": 0, "upper": 1}},
    )
    result = invoke(client, config_path, "results", "--format", fmt)
    assert result.exit_code == 0
    for text in ("failed", "incomplete", "bad computation", "limited calibration"):
        assert text in result.stderr
        assert text not in result.stdout
    assert "Job state is failed; results may be incomplete." in result.stderr
    assert "Warning: limited calibration" in result.stderr
    assert "Error: bad computation" in result.stderr
    assert "Warning: job state" not in result.stderr
    if fmt != "table":
        assert yaml.safe_load(result.stdout) == [
            {"confidence_interval": {"lower": 0.0, "upper": 1.0}}
        ]
    else:
        assert "point_estimate" not in result.stdout


@pytest.mark.parametrize("shape", ["benchmark", "aggregate", "mixed"])
@pytest.mark.parametrize("options", [[], ["--format", "table"]])
def test_results_table(
    client: MagicMock, config_path: Path, shape: str, options: list[str]
) -> None:
    results: dict[str, Any] = {}
    if shape != "aggregate":
        results["benchmarks"] = [BENCHMARK]
    if shape != "benchmark":
        results["confidence_interval"] = INTERVAL
    client.post_processing.get.return_value = make_job(results=results)
    with patch.dict("os.environ", {"COLUMNS": "160"}):
        result = invoke(client, config_path, "results", *options)
    assert result.exit_code == 0, result.output
    for statistic in INTERVAL:
        assert statistic in result.stdout
    assert "0.03" in result.stdout
    assert "confidence_interval" in result.stdout
    assert ("BENCHMARK" in result.stdout) == (shape != "aggregate" and bool(options))
    assert ("inspect" in result.stdout) == (shape != "aggregate")
    assert ("┏" in result.stdout) == bool(options)
    if not options:
        assert result.stdout.count("confidence_interval:") == (
            2 if shape == "mixed" else 1
        )
        if shape != "aggregate":
            assert "Benchmark: accuracy (provider: inspect, index: 0)" in result.stdout
        else:
            assert "Benchmark:" not in result.stdout


def test_default_run(client: MagicMock, config_path: Path) -> None:
    result = invoke(client, config_path, "run")
    assert result.exit_code == 0, result.output
    assert result.stdout == "Post-processing job submitted: pp-123\n"
    assert result.stderr == ""
    client.post_processing.get.assert_not_called()


def test_default_status(client: MagicMock, config_path: Path) -> None:
    result = invoke(client, config_path, "status")
    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "Job:               pp-123\n"
        "Name:              interval-job\n"
        "State:             completed\n"
        "Created:           2026-11-18T12:00:00Z\n"
        "Operations:        confidence_interval\n"
        "Source evaluation: eval-source\n"
    )
    assert result.stderr == ""


def test_default_results_omits_none_preserves_zero(
    client: MagicMock, config_path: Path
) -> None:
    client.post_processing.get.return_value = make_job(
        results={"confidence_interval": {"lower": 0, "upper": 1, "lambda": 0}}
    )
    result = invoke(client, config_path, "results")
    assert result.exit_code == 0, result.output
    assert (
        result.stdout
        == """confidence_interval:
  lower:  0.0
  upper:  1.0
  lambda: 0.0
"""
    )
    assert result.stderr == ""


@pytest.mark.parametrize("state", ["completed", "failed", "cancelled"])
def test_default_wait(client: MagicMock, config_path: Path, state: str) -> None:
    client.post_processing.get.return_value = make_job(
        state, results={"confidence_interval": INTERVAL}
    )
    result = invoke(client, config_path, "run", "--wait")
    assert result.exit_code == 0, result.output
    assert "Waiting" in result.stderr
    assert "submitted" not in result.stdout
    assert "┏" not in result.stdout
    assert f"State:             {state}" in result.stdout
    assert "submitted: pp-123" in result.stderr


def test_default_wait_timeout_empty_stdout(
    client: MagicMock, config_path: Path
) -> None:
    client.post_processing.get.return_value = make_job("running")
    with patch("evalhub.cli.main.time.time", side_effect=[0, 2]):
        result = invoke(client, config_path, "run", "--wait", "--timeout", "1")
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert "Timed out" in result.stderr


def test_default_empty_results(client: MagicMock, config_path: Path) -> None:
    result = invoke(client, config_path, "results")
    assert result.exit_code == 0, result.output
    assert result.stdout == "No results available.\n"
    assert result.stderr == ""


@pytest.mark.parametrize("state", ["pending", "running", "failed", "cancelled"])
@pytest.mark.parametrize("fmt", [None, "table", "json", "yaml"])
def test_empty_results_state_message(
    client: MagicMock, config_path: Path, state: str, fmt: str | None
) -> None:
    client.post_processing.get.return_value = make_job(state)
    options = [] if fmt is None else ["--format", fmt]
    result = invoke(client, config_path, "results", *options)
    assert result.exit_code == 0, result.output
    message = f"Job state is {state}; no results available.\n"
    assert "Warning:" not in result.output
    assert "incomplete" not in result.output
    if fmt is None:
        assert result.stdout == message
        assert result.stderr == ""
    else:
        assert result.stderr == message
        if fmt == "table":
            assert result.stdout == ""
            assert result.output == message
        else:
            assert yaml.safe_load(result.stdout) == []


def test_default_partial_results_state_message(
    client: MagicMock, config_path: Path
) -> None:
    client.post_processing.get.return_value = make_job(
        "running", results={"confidence_interval": INTERVAL}
    )
    result = invoke(client, config_path, "results")
    assert result.exit_code == 0, result.output
    assert result.stderr == "Job state is running; results may be incomplete.\n"
    assert "confidence_interval:" in result.stdout


@pytest.mark.parametrize("command", ["run", "status", "results"])
@pytest.mark.parametrize(
    "options",
    [
        ["--format", "csv"],
        ["--watch"],
        ["--timeout", "1"],
        ["--poll-interval", "0.25"],
    ],
)
def test_invalid_combinations(
    client: MagicMock, config_path: Path, command: str, options: list[str]
) -> None:
    with patch("evalhub.cli.main.get_client") as get_client:
        args = ["post-processing", command]
        args += ["--config", str(config_path)] if command == "run" else ["pp-123"]
        result = CliRunner().invoke(main, [*args, *options])
    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert result.stderr
    get_client.assert_not_called()
    client.post_processing.submit.assert_not_called()
    client.post_processing.get.assert_not_called()


@pytest.mark.parametrize("option", ["--timeout", "--poll-interval"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "word"])
def test_invalid_polling_numbers(config_path: Path, option: str, value: str) -> None:
    args = ["post-processing", "run", "--config", str(config_path)]
    with patch("evalhub.cli.main.get_client") as get_client:
        result = CliRunner().invoke(main, [*args, "--wait", option, value])
    assert result.exit_code == 2
    assert result.stdout == ""
    get_client.assert_not_called()


@pytest.mark.parametrize("command", ["status", "results"])
@pytest.mark.parametrize(
    "options",
    [
        ["--wait"],
        ["--timeout", "1"],
        ["--poll-interval", "0.25"],
        ["--wait", "--timeout", "1"],
        ["--wait", "--poll-interval", "0.25"],
    ],
)
def test_snapshot_commands_reject_waiting(command: str, options: list[str]) -> None:
    with patch("evalhub.cli.main.get_client") as get_client:
        result = CliRunner().invoke(
            main, ["post-processing", command, "pp-123", *options]
        )
    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert "No such option" in result.stderr
    assert options[0] in result.stderr
    get_client.assert_not_called()


@pytest.mark.parametrize("command", ["status", "results"])
@pytest.mark.parametrize(
    "state", ["pending", "running", "completed", "failed", "cancelled"]
)
@pytest.mark.parametrize("fmt", [None, "table", "json", "yaml"])
def test_snapshot_commands_retrieve_once(
    client: MagicMock, config_path: Path, command: str, state: str, fmt: str | None
) -> None:
    client.post_processing.get.return_value = make_job(
        state, results={"confidence_interval": INTERVAL}
    )
    options = [] if fmt is None else ["--format", fmt]
    with patch("evalhub.cli.main.time.sleep") as sleep:
        result = invoke(client, config_path, command, *options)
    assert result.exit_code == 0, result.output
    client.post_processing.get.assert_called_once_with("pp-123")
    client.post_processing.submit.assert_not_called()
    sleep.assert_not_called()
    assert "Waiting" not in result.output


@pytest.mark.parametrize("command", ["status", "results"])
@pytest.mark.parametrize("ids", [[], ["one", "two"]])
def test_exactly_one_id(command: str, ids: list[str]) -> None:
    with patch("evalhub.cli.main.get_client") as get_client:
        result = CliRunner().invoke(main, ["post-processing", command, *ids])
    assert result.exit_code == 2
    get_client.assert_not_called()


@pytest.mark.parametrize("state", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("fmt", ["table", "json", "yaml"])
@pytest.mark.parametrize("immediate", [False, True])
def test_wait_terminal(
    client: MagicMock,
    config_path: Path,
    state: str,
    fmt: str,
    immediate: bool,
) -> None:
    final = make_job(state, results={"confidence_interval": INTERVAL})
    polls = [] if immediate else [make_job("pending"), make_job("running")]
    client.post_processing.submit.return_value = make_job(
        "completed", name="submission"
    )
    client.post_processing.get.side_effect = [*polls, final]
    with patch("evalhub.cli.main.time.sleep") as sleep:
        result = invoke(client, config_path, "run", "--wait", "--format", fmt)
    assert result.exit_code == 0, result.output
    assert client.post_processing.get.call_count == len(polls) + 1
    assert sleep.call_count == len(polls)
    for call in sleep.call_args_list:
        assert call.args == (5.0,)
    assert "Waiting" in result.stderr
    assert "Waiting" not in result.stdout
    assert "submission" not in result.stdout
    if fmt != "table":
        assert yaml.safe_load(result.stdout) == [final.model_dump(mode="json")]


@pytest.mark.parametrize("fmt", ["table", "json", "yaml"])
@pytest.mark.parametrize("slow_request", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_soft_timeout(
    client: MagicMock,
    config_path: Path,
    fmt: str,
    slow_request: bool,
    terminal: bool,
) -> None:
    now = [0.0]

    def submit(request: StandalonePostProcessingRequest) -> PostProcessingResource:
        now[0] += 100  # Submission time must not count against the wait timeout.
        return make_job("pending")

    count = [0]

    def get(job_id: str) -> PostProcessingResource:
        count[0] += 1
        now[0] += 14 if slow_request else 0.1
        final_poll = slow_request or count[0] == 2
        return make_job("completed" if terminal and final_poll else "running")

    def sleep(seconds: float) -> None:
        now[0] += seconds

    client.post_processing.submit.side_effect = submit
    client.post_processing.get.side_effect = get
    with (
        patch("evalhub.cli.main.time.time", side_effect=lambda: now[0]),
        patch("evalhub.cli.main.time.sleep", side_effect=sleep) as sleeper,
    ):
        result = invoke(
            client,
            config_path,
            "run",
            "--wait",
            "--timeout",
            "2",
            "--poll-interval",
            "5",
            "--format",
            fmt,
        )
    assert result.exit_code == (0 if terminal else 1), result.output
    assert count[0] == (1 if slow_request else 2)
    if slow_request:
        sleeper.assert_not_called()
    else:
        sleeper.assert_called_once_with(5.0)
    if not terminal:
        assert result.stdout == ""
        assert "Timed out" in result.stderr
        assert "may still be running" in result.stderr
        assert "Traceback" not in result.stderr


def test_fractional_polling(client: MagicMock, config_path: Path) -> None:
    client.post_processing.get.side_effect = [make_job("running"), make_job()]
    with patch("evalhub.cli.main.time.sleep") as sleep:
        result = invoke(
            client,
            config_path,
            "run",
            "--wait",
            "--poll-interval",
            "0.25",
            "--timeout",
            "0.5",
        )
    assert result.exit_code == 0, result.output
    sleep.assert_called_once_with(0.25)


def api_error(status: int | None) -> httpx.HTTPError:
    if status is None:
        return httpx.ConnectError("offline")
    response = httpx.Response(
        status,
        json={"detail": "server diagnostic"},
        request=httpx.Request("GET", "http://localhost/post-processing/pp-123"),
    )
    return httpx.HTTPStatusError("error", request=response.request, response=response)


@pytest.mark.parametrize("fmt", ["table", "json", "yaml"])
@pytest.mark.parametrize("after_poll", [False, True])
@pytest.mark.parametrize("error_code", [404, 500, None])
def test_polling_errors(
    client: MagicMock,
    config_path: Path,
    fmt: str,
    after_poll: bool,
    error_code: int | None,
) -> None:
    error = api_error(error_code)
    client.post_processing.get.side_effect = (
        [make_job("running"), error] if after_poll else [error]
    )
    with patch("evalhub.cli.main.time.sleep"):
        result = invoke(client, config_path, "run", "--wait", "--format", fmt)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("answer", ["y\n", "n\n", None])
def test_delete_confirmation(client: MagicMock, answer: str | None) -> None:
    args = ["post-processing", "delete", "pp-123"]
    if answer is None:
        args.append("--yes")
    with patch("evalhub.cli.main.get_client", return_value=client):
        result = CliRunner().invoke(main, args, input=answer)
    assert result.exit_code == (1 if answer == "n\n" else 0)
    if answer is None:
        assert result.stdout == "Post-processing job pp-123 deleted.\n"
    else:
        assert "Are you sure" in result.stdout
    if answer == "n\n":
        assert "Aborted!" in result.stderr
        client.post_processing.delete.assert_not_called()
    else:
        assert "Post-processing job pp-123 deleted." in result.stdout
        assert result.stderr == ""
        client.post_processing.delete.assert_called_once_with("pp-123")


@pytest.mark.parametrize("yes", [False, True])
@pytest.mark.parametrize("error_code", [404, 500, None])
def test_delete_errors(client: MagicMock, yes: bool, error_code: int | None) -> None:
    client.post_processing.delete.side_effect = api_error(error_code)
    args = ["post-processing", "delete", "pp-123", *(["--yes"] if yes else [])]
    with patch("evalhub.cli.main.get_client", return_value=client):
        result = CliRunner().invoke(main, args, input="y\n")
    assert result.exit_code == 1
    assert ("Are you sure" in result.stdout) == (not yes)
    if yes:
        assert result.stdout == ""
    assert "Error:" in result.stderr


@pytest.mark.parametrize("group", ["eval", "post-processing"])
@pytest.mark.parametrize(
    ("suffix", "content"), [(".yaml", "key: ["), (".json", "{"), (".yaml", "[]")]
)
def test_bad_config_content(
    client: MagicMock, tmp_path: Path, group: str, suffix: str, content: str
) -> None:
    path = tmp_path / f"request{suffix}"
    path.write_text(content)
    with patch("evalhub.cli.main.get_client", return_value=client):
        result = CliRunner().invoke(main, [group, "run", "--config", str(path)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert str(path) in result.stderr
    assert "Traceback" not in result.stderr
    client.jobs.submit.assert_not_called()
    client.post_processing.submit.assert_not_called()


@pytest.mark.parametrize("group", ["eval", "post-processing"])
@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_config_decode_failure(
    client: MagicMock, tmp_path: Path, group: str, suffix: str
) -> None:
    path = tmp_path / f"request{suffix}"
    path.write_bytes(b"\xff\xfe\x80")
    with patch("evalhub.cli.main.get_client", return_value=client):
        result = CliRunner().invoke(main, [group, "run", "--config", str(path)])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stdout == ""
    assert "Error: Cannot load configuration" in result.stderr
    assert str(path) in result.stderr
    assert "Traceback" not in result.stderr
    client.jobs.submit.assert_not_called()
    client.post_processing.submit.assert_not_called()


@pytest.mark.parametrize("group", ["eval", "post-processing"])
@pytest.mark.parametrize("error_type", [PermissionError, OSError])
@pytest.mark.parametrize("during_read", [False, True])
def test_config_read_failure(
    client: MagicMock,
    config_path: Path,
    group: str,
    error_type: type[OSError],
    during_read: bool,
) -> None:
    error = error_type("read failure")
    file = MagicMock()
    file.__enter__.return_value.read.side_effect = error
    open_kwargs: dict[str, Any] = (
        {"return_value": file} if during_read else {"side_effect": error}
    )
    with (
        patch("evalhub.cli.main.get_client", return_value=client),
        patch("builtins.open", **open_kwargs),
    ):
        result = CliRunner().invoke(main, [group, "run", "--config", str(config_path)])
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert str(config_path) in result.stderr
    assert str(error) in result.stderr
    assert "Traceback" not in result.stderr
    client.jobs.submit.assert_not_called()
    client.post_processing.submit.assert_not_called()


@pytest.mark.parametrize("group", ["eval", "post-processing"])
@pytest.mark.parametrize("directory", [False, True])
def test_config_path_errors(
    client: MagicMock, tmp_path: Path, group: str, directory: bool
) -> None:
    path = tmp_path if directory else tmp_path / "missing.yaml"
    with patch("evalhub.cli.main.get_client", return_value=client):
        result = CliRunner().invoke(main, [group, "run", "--config", str(path)])
    assert result.exit_code == (1 if directory and group == "eval" else 2)
    assert result.stdout == ""
    client.jobs.submit.assert_not_called()
    client.post_processing.submit.assert_not_called()


@pytest.mark.parametrize("group", ["eval", "post-processing"])
def test_config_path_readability_precheck_disabled(group: str) -> None:
    command_group = main.commands[group]
    assert isinstance(command_group, click.Group)
    config = next(
        p for p in command_group.commands["run"].params if p.name == "config_file"
    )
    assert isinstance(config.type, click.Path)
    assert config.type.readable is False
    assert config.type.dir_okay == (group == "eval")


@pytest.mark.parametrize(
    "content",
    ["{}", "operations: []", "operations: {confidence_interval: {}}"],
)
def test_invalid_request(client: MagicMock, config_path: Path, content: str) -> None:
    config_path.write_text(content)
    result = invoke(client, config_path, "run")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Invalid post-processing configuration" in result.stderr
    client.post_processing.submit.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        """
name: evaluation-confidence-interval
hardware_config:
  cpu:
    request: "1"
  memory:
    request: 2Gi
operation_order:
  - confidence_interval
operations:
  confidence_interval:
    results_data_ref:
      eval_job:
        id: eval-123
    calibration_data_ref:
      - s3:
          bucket: evaluation-data
          key: calibration.jsonl
          secret_ref: evaluation-s3-credentials
        data_config:
          format: jsonl
          columns:
            label: human_score
            prediction: model_score
            sample_id: id
    significance_level: 0.05
""",
        """
name: external-confidence-interval
operations:
  confidence_interval:
    results_data_ref:
      s3:
        bucket: evaluation-data
        key: predictions.jsonl
        secret_ref: evaluation-s3-credentials
      data_config:
        format: jsonl
        columns:
          prediction: model_score
          sample_id: id
    calibration_data_ref:
      - s3:
          bucket: evaluation-data
          key: calibration.jsonl
          secret_ref: evaluation-s3-credentials
        data_config:
          format: jsonl
          columns:
            label: human_score
            prediction: model_score
            sample_id: id
    significance_level: 0.05
    primary_score:
      metric: accuracy
      lower_is_better: false
""",
    ],
    ids=["evaluation-job", "external"],
)
def test_example_requests(payload: str) -> None:
    StandalonePostProcessingRequest.model_validate(yaml.safe_load(payload))


def test_operation_discovery_and_order() -> None:
    """Shared status rendering must not treat all operations as intervals."""
    job = MagicMock()
    data = make_job().model_dump(mode="json")
    data["operations"]["future_operation"] = {"input": {"id": "not-an-evaluation"}}
    data["operation_order"] = ["future_operation", "confidence_interval"]
    job.model_dump.return_value = data
    with patch("evalhub.cli.main.output") as render:
        _output_post_processing(job, "table")
    fields = {row["field"]: row["value"] for row in render.call_args.args[0]}
    assert fields["Operations"] == "future_operation, confidence_interval"
    assert fields["confidence_interval: Source evaluation"] == "eval-source"
    assert "future_operation: Source evaluation" not in fields

    data["operations"] = {"future_operation": {"lower": 1, "upper": 2}}
    data["operation_order"] = None
    with patch("evalhub.cli.main.output") as render:
        _output_post_processing(job, "table")
    fields = {row["field"]: row["value"] for row in render.call_args.args[0]}
    assert fields["Operations"] == "future_operation"
    assert not any("Source" in field for field in fields)


def test_result_operation_keys_stay_separate() -> None:
    job = MagicMock()
    job.results.model_dump.return_value = {
        "benchmarks": [{**BENCHMARK, "future_operation": {"score": 0}}],
        "confidence_interval": INTERVAL,
        "future_operation": {"score": 1},
    }
    assert _post_processing_result_records(job) == [
        {**BENCHMARK, "future_operation": {"score": 0}},
        {"confidence_interval": INTERVAL, "future_operation": {"score": 1}},
    ]


@pytest.mark.parametrize("shape", ["benchmark", "aggregate", "mixed"])
def test_statistic_row_values_and_identity(shape: str) -> None:
    results: dict[str, Any] = {}
    if shape != "aggregate":
        results["benchmarks"] = [BENCHMARK]
    if shape != "benchmark":
        results["confidence_interval"] = {**INTERVAL, "lower": 0.2}
    with patch("evalhub.cli.main.output") as render:
        _output_post_processing_results(make_job(results=results), "table")
    rows = render.call_args.args[0]
    expected_rows = 16 if shape == "mixed" else 8
    assert len(rows) == expected_rows
    assert rows[0]["statistic"] == "lower"
    assert rows[0]["value"] == (0.2 if shape == "aggregate" else 0.0)
    assert rows[5]["statistic"] == "alpha"
    assert rows[5]["value"] == 0.03
    assert rows[7]["statistic"] == "lambda"
    assert rows[7]["value"] == 0
    if shape == "aggregate":
        assert all(set(row) == {"operation", "statistic", "value"} for row in rows)
    else:
        assert rows[0]["benchmark"] == "accuracy"
        assert rows[0]["provider"] == "inspect"
        assert rows[0]["index"] == 0
        if shape == "mixed":
            assert rows[8]["value"] == 0.2
            for row in rows[8:]:
                assert row["benchmark"] == row["provider"] == row["index"] == ""


@pytest.mark.parametrize("command", ["run", "status", "results"])
@pytest.mark.parametrize("fmt", ["table", "json", "yaml"])
@pytest.mark.parametrize("error_code", [404, 500, None])
def test_nonwaiting_api_errors(
    client: MagicMock, config_path: Path, command: str, fmt: str, error_code: int | None
) -> None:
    error = api_error(error_code)
    client.post_processing.submit.side_effect = error
    client.post_processing.get.side_effect = error
    result = invoke(client, config_path, command, "--format", fmt)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_no_implicit_waiting_timeout(client: MagicMock, config_path: Path) -> None:
    client.post_processing.get.side_effect = [make_job("running"), make_job()]
    # No deadline checks are necessary when timeout is omitted.
    with (
        patch("evalhub.cli.main.time.time", side_effect=[0, 1e9]) as clock,
        patch("evalhub.cli.main.time.sleep"),
    ):
        result = invoke(client, config_path, "run", "--wait")
    assert result.exit_code == 0, result.output
    assert clock.call_count == 1


def test_successful_eval_json_configuration(
    client: MagicMock, config_path: Path
) -> None:
    config_path = config_path.with_suffix(".json")
    config_path.write_text(
        json.dumps(
            {
                "name": "evaluation",
                "model": {"url": "http://model:8000/v1", "name": "model"},
                "benchmarks": [{"id": "accuracy", "provider_id": "inspect"}],
            }
        )
    )
    with patch("evalhub.cli.main.get_client", return_value=client):
        result = CliRunner().invoke(main, ["eval", "run", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    client.jobs.submit.assert_called_once()
