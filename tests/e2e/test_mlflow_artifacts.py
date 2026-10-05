"""Artifact round trip through an isolated local MLflow server."""

import importlib
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from evalhub.adapter.callbacks import DefaultCallbacks
from evalhub.adapter.config import MlflowBackend
from evalhub.adapter.mlflow import MlflowArtifact, MlflowClient, MlflowFileArtifact
from evalhub.adapter.models import JobResults, JobSpec
from evalhub.models.api import EvaluationResult, ModelConfig

pytestmark = pytest.mark.e2e


@pytest.fixture
def local_mlflow_environment(
    mlflow_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in tuple(os.environ):
        if key.startswith("MLFLOW_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_server)


def test_file_artifact_round_trip(
    tmp_path: Path, mlflow_server: str, local_mlflow_environment: None
) -> None:
    # Cross multiple 64 KiB boundaries, including a final partial chunk.
    payload = bytes(range(256)) * 1024 + b"final partial chunk"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    empty = tmp_path / "empty.bin"
    empty.touch()
    with MlflowClient(tracking_uri=mlflow_server) as client:
        experiment_id = client.get_or_create_experiment("sdk-file-artifact-e2e")
        with client.start_run(
            experiment_id, run_name="sdk-file-artifact-round-trip"
        ) as run_id:
            for destination, file, expected in (
                ("nested/renamed.bin", source, payload),
                ("nested/empty.bin", empty, b""),
            ):
                client.upload_artifact_file(run_id, destination, file)
                run = client.get_run(run_id)
                path = client._artifact_server_path(
                    run.artifact_uri,
                    destination,
                    experiment_id=run.experiment_id,
                    run_id=run.run_id,
                )
                response = client._client.get(f"{mlflow_server}{path}")
                response.raise_for_status()
                assert response.content == expected
    assert source.read_bytes() == payload
    assert empty.exists()


@pytest.mark.parametrize("backend", [MlflowBackend.ODH, MlflowBackend.UPSTREAM])
def test_callback_save_mixed_artifacts(
    tmp_path: Path,
    mlflow_server: str,
    local_mlflow_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    backend: MlflowBackend,
) -> None:
    """Exercise both save backends against the real server, without mocked APIs."""
    payload = bytes(range(256)) * 1024 + b"final partial chunk"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    empty = tmp_path / "empty.bin"
    empty.touch()
    summary = b'{"status": "complete"}'
    artifacts: list[MlflowArtifact | MlflowFileArtifact] = [
        MlflowArtifact("summary.json", summary, "application/json"),
        MlflowArtifact("metadata/details.json", b"{}", "application/json"),
        MlflowFileArtifact("results/source.bin", source),
        MlflowFileArtifact("results/nested/renamed.bin", source),
        MlflowFileArtifact("empty.bin", empty),
    ]
    spec = JobSpec(
        id=f"job-{backend.value}",
        provider_id="test-provider",
        benchmark_id="test-benchmark",
        benchmark_index=0,
        model=ModelConfig(url="http://localhost/v1", name="test-model"),
        parameters={},
        callback_url="http://localhost:8080",
        experiment_name=f"sdk-callback-save-{backend.value}",
        tags=[{"key": "test_backend", "value": backend.value}],
    )
    results = JobResults(
        id=spec.id,
        benchmark_id=spec.benchmark_id,
        benchmark_index=0,
        model_name=spec.model.name,
        results=[EvaluationResult(metric_name="acc,none", metric_value=0.9)],
        overall_score=0.9,
        num_examples_evaluated=3,
        duration_seconds=1.5,
        completed_at=datetime.now(UTC),
    )
    callbacks = DefaultCallbacks(
        job_id=spec.id,
        benchmark_id=spec.benchmark_id,
        provider_id=spec.provider_id,
        mlflow_backend=backend,
    )

    # Restore the official library's global configuration after the upstream case.
    if backend == MlflowBackend.UPSTREAM:
        mlflow = importlib.import_module("mlflow")
        fluent = importlib.import_module("mlflow.tracking.fluent")
        monkeypatch.setattr(fluent, "_active_experiment_id", None)
        monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
        previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(mlflow_server)
        try:
            run_id = callbacks.mlflow.save(results, spec, artifacts)
            assert mlflow.active_run() is None
        finally:
            mlflow.set_tracking_uri(previous_uri)
    else:
        run_id = callbacks.mlflow.save(results, spec, artifacts)
    assert run_id is not None

    with MlflowClient(tracking_uri=mlflow_server) as client:
        run = client._get("/runs/get", {"run_id": run_id})["run"]
        assert run["info"]["run_id"] == run_id
        assert run["info"]["status"] == "FINISHED"
        assert {item["key"]: item["value"] for item in run["data"]["metrics"]} == {
            "acc_none": 0.9,
            "overall_score": 0.9,
        }
        assert {item["key"]: item["value"] for item in run["data"]["params"]} == {
            "benchmark_id": spec.benchmark_id,
            "provider_id": spec.provider_id,
            "model_name": spec.model.name,
            "num_examples_evaluated": "3",
            "duration_seconds": "1.5",
        }
        tags = {item["key"]: item["value"] for item in run["data"]["tags"]}
        assert tags["test_backend"] == backend.value
        assert tags["mlflow.runName"] == f"{spec.id}_0"
        runs = client._post(
            "/runs/search", {"experiment_ids": [run["info"]["experiment_id"]]}
        )["runs"]
        assert [item["info"]["run_id"] for item in runs] == [run_id]

        for destination, expected in (
            ("summary.json", summary),
            ("metadata/details.json", b"{}"),
            ("results/source.bin", payload),
            ("results/nested/renamed.bin", payload),
            ("empty.bin", b""),
        ):
            path = client._artifact_server_path(
                run["info"]["artifact_uri"], destination
            )
            response = client._client.get(f"{mlflow_server}{path}")
            response.raise_for_status()
            assert response.content == expected
    assert source.read_bytes() == payload
    assert empty.exists() and empty.stat().st_size == 0
