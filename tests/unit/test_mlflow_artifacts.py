"""File artifact streaming and callback destination semantics."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from evalhub.adapter.callbacks import _MlflowOps
from evalhub.adapter.config import MlflowBackend
from evalhub.adapter.mlflow import MlflowArtifact, MlflowClient, MlflowFileArtifact
from evalhub.adapter.models import JobResults, JobSpec
from evalhub.models.api import EvaluationResult, ModelConfig

pytestmark = pytest.mark.unit


def _job_spec(experiment_name: str = "exp") -> JobSpec:
    return JobSpec(
        id="job-1",
        provider_id="provider",
        benchmark_id="benchmark",
        benchmark_index=0,
        model=ModelConfig(url="http://localhost/v1", name="model"),
        parameters={},
        callback_url="http://evalhub:8080",
        experiment_name=experiment_name,
    )


def _results() -> JobResults:
    return JobResults(
        id="job-1",
        benchmark_id="benchmark",
        benchmark_index=0,
        model_name="model",
        results=[EvaluationResult(metric_name="acc", metric_value=0.9)],
        num_examples_evaluated=1,
        duration_seconds=1.0,
        completed_at=datetime.now(UTC),
    )


class ArtifactTransport(httpx.BaseTransport):
    """Inspect the original request stream before HTTPX buffers it."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.uploads: dict[str, tuple[list[int], str, httpx.Headers]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            assert isinstance(request.stream, httpx.SyncByteStream)
            sizes = []
            digest = hashlib.sha256()
            for chunk in request.stream:
                sizes.append(len(chunk))
                digest.update(chunk)
                if self.fail:
                    raise httpx.WriteError("upload failed", request=request)
            self.uploads[request.url.path] = (
                sizes,
                digest.hexdigest(),
                request.headers,
            )
            return httpx.Response(200, json={})
        path = request.url.path
        body = json.loads(request.read()) if request.method == "POST" else {}
        self.calls.append((path, body))
        if path.endswith("/experiments/get-by-name"):
            return httpx.Response(200, json={"experiment": {"experiment_id": "7"}})
        info = {
            "run_id": "run-1",
            "experiment_id": "7",
            "artifact_uri": "mlflow-artifacts:/workspaces/ws/7/run-1/artifacts",
        }
        return httpx.Response(200, json={"run": {"info": info}})


@pytest.mark.parametrize("size", [0, 65536, 2 * 65536 + 13])
@pytest.mark.parametrize(
    ("filename", "override", "expected_type"),
    [
        ("source.json", None, "application/json"),
        ("source.unknown_extension", None, "application/octet-stream"),
        ("source.json", "text/plain", "text/plain"),
    ],
)
def test_file_upload_streaming(
    tmp_path: Path,
    size: int,
    filename: str,
    override: str | None,
    expected_type: str,
) -> None:
    source = tmp_path / filename
    payload = b"x" * size
    source.write_bytes(payload)
    transport = ArtifactTransport()
    with MlflowClient(tracking_uri="http://mlflow") as client:
        client._client.close()
        client._client = httpx.Client(transport=transport)
        with source.open("rb") as handle:
            guarded = MagicMock(wraps=handle)
            guarded.__enter__.return_value = guarded
            guarded.__exit__.side_effect = lambda *args: handle.close()
            with patch.object(Path, "open", return_value=guarded):
                client.upload_artifact_file(
                    "run-1", "nested/renamed.bin", source, override
                )
            assert handle.closed
            assert guarded.read.call_count == (size + 65535) // 65536 + 1
            assert all(call.args == (65536,) for call in guarded.read.call_args_list)
    sizes, digest, headers = next(iter(transport.uploads.values()))
    assert sizes == [65536] * (size // 65536) + ([size % 65536] if size % 65536 else [])
    assert digest == hashlib.sha256(payload).hexdigest()
    assert headers["content-type"] == expected_type
    assert headers["content-length"] == str(size)
    assert "transfer-encoding" not in headers
    assert next(iter(transport.uploads)).endswith("/nested/renamed.bin")


def test_file_closed_on_upload_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * 100000)
    with MlflowClient(tracking_uri="http://mlflow") as client:
        client._client.close()
        client._client = httpx.Client(transport=ArtifactTransport(fail=True))
        handle = source.open("rb")
        with patch.object(Path, "open", return_value=handle):
            with pytest.raises(httpx.WriteError, match="upload failed"):
                client.upload_artifact_file("run-1", "file.bin", source)
        assert handle.closed


def test_odh_mixed_artifacts_share_run(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"result": 1}')
    artifacts = (
        MlflowArtifact("summary.json", b"{}", "application/json"),
        MlflowFileArtifact("nested/renamed.txt", str(source), "text/plain"),
        MlflowFileArtifact("nested/inferred.json", source),
    )
    transport = ArtifactTransport()
    with MlflowClient(tracking_uri="http://mlflow") as client:
        client._client.close()
        client._client = httpx.Client(transport=transport)
        with patch("evalhub.adapter.mlflow.MlflowClient", return_value=client):
            assert _MlflowOps().save(_results(), _job_spec(), artifacts) == "run-1"
    assert len(transport.uploads) == 3
    root = "/api/2.0/mlflow-artifacts/artifacts/workspaces/ws/7/run-1/artifacts/"
    for artifact in artifacts:
        _, digest, headers = transport.uploads[root + artifact.path]
        content = (
            artifact.content
            if isinstance(artifact, MlflowArtifact)
            else source.read_bytes()
        )
        assert digest == hashlib.sha256(content).hexdigest()
        assert headers["content-type"] == (artifact.content_type or "application/json")
        assert headers["content-length"] == str(len(content))
        assert "transfer-encoding" not in headers
    assert sum(path.endswith("/runs/create") for path, _ in transport.calls) == 1
    assert any(
        path.endswith("/runs/log-batch") and body["run_id"] == "run-1"
        for path, body in transport.calls
    )
    assert any(
        path.endswith("/runs/update") and body["status"] == "FINISHED"
        for path, body in transport.calls
    )


def test_existing_bytes_constructor_and_list_callers() -> None:
    artifacts: list[MlflowArtifact] = [MlflowArtifact("file.bin", b"bytes")]
    original = artifacts.copy()
    transport = ArtifactTransport()
    with MlflowClient(tracking_uri="http://mlflow") as client:
        client._client.close()
        client._client = httpx.Client(transport=transport)
        with patch("evalhub.adapter.mlflow.MlflowClient", return_value=client):
            assert _MlflowOps().save(_results(), _job_spec(), artifacts) == "run-1"
    assert artifacts == original
    _, digest, headers = next(iter(transport.uploads.values()))
    assert digest == hashlib.sha256(b"bytes").hexdigest()
    assert headers["content-type"] == "application/octet-stream"


@pytest.mark.parametrize("backend", list(MlflowBackend))
def test_no_experiment_does_not_open_files(backend: MlflowBackend) -> None:
    with patch.object(Path, "open", side_effect=AssertionError("file opened")):
        assert (
            _MlflowOps(backend).save(
                _results(),
                _job_spec(""),
                [MlflowFileArtifact("missing.bin", "missing.bin")],
            )
            is None
        )


@pytest.mark.parametrize("copy_fallback", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("symlink", [False, True])
def test_upstream_mixed_artifacts_and_cleanup(
    tmp_path: Path, copy_fallback: bool, fail: bool, symlink: bool
) -> None:
    source = tmp_path / "source.json"
    payload = b"x" * (2 * 65536 + 13)
    source.write_bytes(payload)
    if symlink:
        link = tmp_path / "link.json"
        link.symlink_to(source.name)
        source = link
    upstream = MagicMock()
    upstream.start_run.return_value.__enter__.return_value.info.run_id = "upstream-run"
    staged: list[Path] = []
    destinations: list[tuple[str, str | None]] = []

    def log_artifact(local_path: str, artifact_path: str | None) -> None:
        path = Path(local_path)
        destinations.append((path.name, artifact_path))
        if path != source:
            staged.append(path)
        if path.name == "summary.json":
            assert path.read_bytes() == b"{}"
        else:
            assert path.read_bytes() == payload
            if path != source and not copy_fallback:
                assert path.stat().st_ino == source.stat().st_ino
        if fail and path.name == "renamed.bin":
            raise OSError("upload failed")

    upstream.log_artifact.side_effect = log_artifact
    artifacts = (
        MlflowArtifact("summary.json", b"{}", "application/json"),
        MlflowFileArtifact(f"nested/{source.name}", source),
        MlflowFileArtifact("nested/deep/renamed.bin", source, "text/plain"),
    )
    with patch.dict("sys.modules", {"mlflow": upstream}):
        with patch(
            "os.link", side_effect=OSError("cross-device")
        ) if copy_fallback else patch("os.link", wraps=os.link):
            with patch("shutil.copyfileobj", wraps=shutil.copyfileobj) as copy:
                ops = _MlflowOps(MlflowBackend.UPSTREAM)
                if fail:
                    with pytest.raises(
                        RuntimeError, match="MLflow save failed: upload failed"
                    ):
                        ops.save(_results(), _job_spec(), artifacts)
                else:
                    assert (
                        ops.save(_results(), _job_spec(), artifacts) == "upstream-run"
                    )
                if copy_fallback:
                    assert copy.call_args.kwargs["length"] == 65536
                else:
                    copy.assert_not_called()
    assert destinations == [
        ("summary.json", None),
        (source.name, "nested"),
        ("renamed.bin", "nested/deep"),
    ]
    assert upstream.log_artifact.call_args_list[1].args == (str(source),)
    assert all(not path.parent.exists() for path in staged)
    assert source.read_bytes() == payload
    upstream.start_run.assert_called_once()
    upstream.log_params.assert_called_once()
    upstream.log_metrics.assert_called_once()
