"""Lightweight MLflow tracking client.

A minimal, zero-dependency (beyond httpx) client for the MLflow REST API.
Covers experiment and run lifecycle plus metric/param/tag logging — the
subset needed by the EvalHub adapter SDK.

Modelled after github.com/opendatahub-io/mlflow-go.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# MLflow REST API base path
_API = "/api/2.0/mlflow"
# MLflow Artifacts server base path (separate from tracking API)
_ARTIFACTS_API = "/api/2.0/mlflow-artifacts/artifacts"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Metric:
    key: str
    value: float
    timestamp: int = 0  # unix ms; 0 = auto-fill
    step: int = 0


# Tracking REST /runs/log-batch rejects keys outside [A-Za-z0-9_\-.\s:/]
_BAD_MLFLOW_METRIC_KEY_CHARS = re.compile(r"[^a-zA-Z0-9_\-.\s:/]+")


def sanitize_metric_key_for_api(name: str) -> str:
    """Map metric names to MLflow-safe keys (e.g. lm-eval ``acc,none`` → ``acc_none``).

    Used only when logging to MLflow; ``JobResults`` metric names are unchanged.
    """
    s = _BAD_MLFLOW_METRIC_KEY_CHARS.sub("_", name).strip().strip("_")
    return s or "metric"


@dataclass
class Param:
    key: str
    value: str


@dataclass
class RunInfo:
    run_id: str
    experiment_id: str
    run_name: str = ""
    status: str = "RUNNING"
    start_time: int = 0
    end_time: int = 0
    artifact_uri: str = ""
    lifecycle_stage: str = ""


@dataclass
class Experiment:
    experiment_id: str
    name: str
    artifact_location: str = ""
    lifecycle_stage: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ArtifactInfo:
    path: str
    is_dir: bool = False
    file_size: int = 0


@dataclass
class MlflowArtifact:
    """An artifact to upload to MLflow alongside metrics."""

    path: str
    content: bytes
    content_type: str = "application/octet-stream"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class MLflowAPIError(Exception):
    """Error returned by the MLflow REST API."""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message} (HTTP {status_code})")


def _is_resource_already_exists(err: Exception) -> bool:
    return (
        isinstance(err, MLflowAPIError) and err.error_code == "RESOURCE_ALREADY_EXISTS"
    )


def _detect_ca_bundle() -> Path | None:
    """Auto-detect CA bundle from standard OpenShift / Kubernetes locations."""
    candidates = [
        Path("/etc/pki/ca-trust/source/anchors/service-ca.crt"),  # OpenShift
        Path(
            "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
        ),  # OpenShift SA
        Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"),  # Kubernetes
    ]
    for path in candidates:
        if path.exists():
            logger.debug("Auto-detected CA bundle at: %s", path)
            return path
    return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MlflowClient:
    """Minimal MLflow tracking client using the REST API.

    Configuration priority (mirrors the official SDK and mlflow-go):
    1. Explicit constructor arguments
    2. Environment variables (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_TOKEN, etc.)

    Example::

        client = MlflowClient(tracking_uri="http://localhost:5000")

        exp_id = client.get_or_create_experiment("my-experiment")
        with client.start_run(exp_id, run_name="eval-42") as run_id:
            client.log_batch(
                run_id,
                metrics=[Metric("accuracy", 0.95)],
                params=[Param("model", "llama-3")],
                tags={"env": "ci"},
            )
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        token: str | None = None,
        token_path: str | Path | None = None,
        headers: dict[str, str] | None = None,
        insecure: bool | None = None,
        timeout: float = 30.0,
        ca_bundle: str | Path | None = None,
    ) -> None:
        # --- Tracking URI ---
        self._tracking_uri = (
            tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or ""
        ).rstrip("/")
        if not self._tracking_uri:
            raise ValueError(
                "MLflow tracking URI is required "
                "(pass tracking_uri or set MLFLOW_TRACKING_URI)"
            )

        # --- Auth token ---
        self._token = self._resolve_token(token, token_path)

        # --- HTTP headers ---
        req_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._token:
            req_headers["Authorization"] = f"Bearer {self._token}"

        # Workspace header (Red Hat midstream multi-tenant MLflow fork)
        workspace = os.environ.get("MLFLOW_WORKSPACE")
        if workspace:
            req_headers["X-MLFLOW-WORKSPACE"] = workspace

        # Caller-supplied headers win (can override workspace / auth)
        if headers:
            req_headers.update(headers)

        # --- TLS verification ---
        # Priority:
        #   1. explicit insecure param
        #   2. MLFLOW_TRACKING_INSECURE_TLS / MLFLOW_INSECURE_SKIP_TLS_VERIFY env
        #   3. explicit ca_bundle param
        #   4. MLFLOW_TRACKING_SERVER_CERT_PATH env (what mlflow-skinny used)
        #   5. auto-detect from standard OpenShift / K8s locations
        #   6. system defaults
        if insecure is None:
            insecure = False
            for env_name in (
                "MLFLOW_TRACKING_INSECURE_TLS",
                "MLFLOW_INSECURE_SKIP_TLS_VERIFY",
            ):
                if os.environ.get(env_name, "").lower() in ("true", "1"):
                    insecure = True
                    break

        if insecure:
            verify: bool | str = False
        elif ca_bundle:
            verify = str(ca_bundle)
        else:
            # Check the env var that mlflow-skinny / the eval-hub service sets
            env_cert = os.environ.get("MLFLOW_TRACKING_SERVER_CERT_PATH")
            if env_cert and Path(env_cert).exists():
                verify = env_cert
            else:
                detected = _detect_ca_bundle()
                verify = str(detected) if detected else True

        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            verify=verify,
            headers=req_headers,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MlflowClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- token resolution ---------------------------------------------------

    @staticmethod
    def _resolve_token(
        explicit: str | None,
        token_path: str | Path | None,
    ) -> str | None:
        if explicit:
            return explicit

        # MLFLOW_TRACKING_TOKEN takes precedence
        env_token = os.environ.get("MLFLOW_TRACKING_TOKEN")
        if env_token:
            return env_token

        # Then try the token-path env var (ROSA/STS projected token)
        path = token_path or os.environ.get("MLFLOW_TRACKING_TOKEN_PATH")
        if path:
            try:
                return Path(path).read_text().strip() or None
            except OSError as e:
                logger.warning("Could not read MLflow token from %s: %s", path, e)

        return None

    # -- low-level HTTP -----------------------------------------------------

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._tracking_uri}{_API}{path}"
        resp = self._client.post(url, json=body)
        return self._handle(resp)

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self._tracking_uri}{_API}{path}"
        resp = self._client.get(url, params=params)
        return self._handle(resp)

    @staticmethod
    def _handle(resp: httpx.Response) -> dict[str, Any]:
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = {}
            raise MLflowAPIError(
                status_code=resp.status_code,
                error_code=body.get("error_code", "UNKNOWN"),
                message=body.get("message", resp.text),
            )
        if not resp.content:
            return {}
        return resp.json()  # type: ignore[no-any-return]

    # -- Experiment operations ----------------------------------------------

    def create_experiment(self, name: str) -> str:
        data = self._post("/experiments/create", {"name": name})
        return str(data["experiment_id"])

    def get_experiment_by_name(self, name: str) -> Experiment | None:
        try:
            data = self._get("/experiments/get-by-name", {"experiment_name": name})
        except MLflowAPIError as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                return None
            raise
        exp = data.get("experiment", {})
        tags: dict[str, str] = {}
        for t in exp.get("tags", []):
            tags[t["key"]] = t["value"]
        return Experiment(
            experiment_id=exp.get("experiment_id", ""),
            name=exp.get("name", ""),
            artifact_location=exp.get("artifact_location", ""),
            lifecycle_stage=exp.get("lifecycle_stage", ""),
            tags=tags,
        )

    def get_or_create_experiment(self, name: str) -> str:
        exp = self.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        try:
            return self.create_experiment(name)
        except MLflowAPIError as e:
            # Handle race condition: another process created it between
            # our check and our create call.
            if _is_resource_already_exists(e):
                exp = self.get_experiment_by_name(name)
                if exp is not None:
                    return exp.experiment_id
            raise

    # -- Run operations -----------------------------------------------------

    def create_run(
        self,
        experiment_id: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> RunInfo:
        body: dict[str, Any] = {
            "experiment_id": experiment_id,
            "start_time": _now_ms(),
        }
        if run_name:
            body["run_name"] = run_name
        if tags:
            body["tags"] = [{"key": k, "value": v} for k, v in tags.items()]

        data = self._post("/runs/create", body)
        run = data.get("run", {}).get("info", {})
        return RunInfo(
            run_id=run.get("run_id", ""),
            experiment_id=run.get("experiment_id", ""),
            run_name=run.get("run_name", ""),
            status=run.get("status", "RUNNING"),
            start_time=run.get("start_time", 0),
            artifact_uri=run.get("artifact_uri", ""),
            lifecycle_stage=run.get("lifecycle_stage", ""),
        )

    def update_run(
        self,
        run_id: str,
        status: str | None = None,
        end_time: int | None = None,
        run_name: str | None = None,
    ) -> None:
        body: dict[str, Any] = {"run_id": run_id}
        if status:
            body["status"] = status
        if end_time is not None:
            body["end_time"] = end_time
        if run_name:
            body["run_name"] = run_name
        self._post("/runs/update", body)

    @contextmanager
    def start_run(
        self,
        experiment_id: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Context manager that creates a run and finalises it on exit.

        Yields the run_id.  On normal exit the run is marked FINISHED;
        on exception it is marked FAILED.
        """
        run_info = self.create_run(experiment_id, run_name=run_name, tags=tags)
        run_id = run_info.run_id
        # Keep the FINISHED and FAILED paths separate so that a transient
        # failure in the FINISHED update doesn't incorrectly mark the run
        # as FAILED (the work completed successfully in that case).
        failed = False
        try:
            yield run_id
        except BaseException:
            failed = True
            try:
                self.update_run(run_id, status="FAILED", end_time=_now_ms())
            except Exception:
                logger.warning("Could not mark run %s as FAILED", run_id)
            raise
        if not failed:
            self.update_run(run_id, status="FINISHED", end_time=_now_ms())

    # -- Logging operations -------------------------------------------------

    def log_batch(
        self,
        run_id: str,
        metrics: list[Metric] | None = None,
        params: list[Param] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        now = _now_ms()
        body: dict[str, Any] = {"run_id": run_id}

        if metrics:
            body["metrics"] = [
                {
                    "key": m.key,
                    "value": m.value,
                    "timestamp": m.timestamp or now,
                    "step": m.step,
                }
                for m in metrics
            ]

        if params:
            body["params"] = [{"key": p.key, "value": p.value} for p in params]

        if tags:
            body["tags"] = [{"key": k, "value": v} for k, v in tags.items()]

        self._post("/runs/log-batch", body)

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        self._post(
            "/runs/log-metric",
            {
                "run_id": run_id,
                "key": key,
                "value": value,
                "timestamp": _now_ms(),
                "step": step,
            },
        )

    def log_param(self, run_id: str, key: str, value: str) -> None:
        self._post(
            "/runs/log-parameter",
            {"run_id": run_id, "key": key, "value": value},
        )

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self._post(
            "/runs/set-tag",
            {"run_id": run_id, "key": key, "value": value},
        )

    # -- Run operations -----------------------------------------------------

    def get_run(self, run_id: str) -> RunInfo:
        """Fetch run metadata by run_id."""
        data = self._get("/runs/get", {"run_id": run_id})
        run = data.get("run", {}).get("info", {})
        return RunInfo(
            run_id=run.get("run_id", ""),
            experiment_id=run.get("experiment_id", ""),
            run_name=run.get("run_name", ""),
            status=run.get("status", ""),
            start_time=run.get("start_time", 0),
            end_time=run.get("end_time", 0),
            artifact_uri=run.get("artifact_uri", ""),
            lifecycle_stage=run.get("lifecycle_stage", ""),
        )

    # -- Artifact operations ------------------------------------------------

    @staticmethod
    def _artifact_server_path(artifact_uri: str, artifact_path: str) -> str:
        """Compute the PUT path for the MLflow Artifacts server from a run's artifact_uri.

        The standard MLflow artifact server derives the storage path from a
        ``?run_id=`` query parameter on artifact upload requests. The ODH fork
        ignores that parameter and instead resolves paths from the ``artifact_uri``
        embedded in the run record, which has the form::

            mlflow-artifacts:/workspaces/{workspace}/{experiment_id}/{run_id}/artifacts

        Stripping ``mlflow-artifacts:/workspaces/{workspace}/`` gives the path
        within the workspace's artifact storage, which is what the ODH server
        expects after ``/api/2.0/mlflow-artifacts/artifacts/`` — no ``?run_id=``
        query string. The upstream library would send ``?run_id=``, which the ODH
        server ignores for path resolution, potentially causing files to land in
        the wrong location.
        """
        # Strip scheme
        path = artifact_uri.removeprefix("mlflow-artifacts:/")
        # path = "workspaces/{workspace}/{experiment_id}/{run_id}/artifacts"
        # Strip "workspaces/{workspace}/" so the server doesn't double-prefix it
        parts = path.split("/", 2)
        if len(parts) == 3 and parts[0] == "workspaces":
            run_root = parts[2]  # "{experiment_id}/{run_id}/artifacts"
        else:
            run_root = path  # fallback: use as-is
        artifact_path = artifact_path.lstrip("/")
        return f"{_ARTIFACTS_API}/{run_root}/{artifact_path}"

    def _put_artifact(
        self, path: str, content: bytes | Iterable[bytes], content_type: str
    ) -> None:
        """Raw PUT to the MLflow Artifacts server."""
        url = f"{self._tracking_uri}{path}"
        headers = {"Content-Type": content_type}
        resp = self._client.put(url, content=content, headers=headers)
        self._handle(resp)

    def upload_artifact(
        self,
        run_id: str,
        artifact_path: str,
        content: bytes | Iterable[bytes],
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload content to the MLflow Artifacts server.

        ``artifact_path`` is the destination path relative to the run's
        artifact root, e.g. ``"results/output.json"``.

        The run's ``artifact_uri`` is resolved first so the file lands at the
        correct path in the artifact store (the ODH fork ignores ``?run_id=``
        for storage path resolution).
        """
        run_info = self.get_run(run_id)
        path = self._artifact_server_path(run_info.artifact_uri, artifact_path)
        self._put_artifact(path, content, content_type)
        logger.debug("Uploaded artifact %s for run %s", artifact_path, run_id)

    def upload_artifact_file(
        self,
        run_id: str,
        artifact_path: str,
        local_path: str | Path,
    ) -> None:
        """Upload a local file to the MLflow Artifacts server.

        The Content-Type is guessed from the file extension; unknown types
        fall back to ``application/octet-stream``.
        """
        local_path = Path(local_path)
        content_type, _ = mimetypes.guess_type(str(local_path))
        if not content_type:
            content_type = "application/octet-stream"
        with local_path.open("rb") as f:
            self.upload_artifact(
                run_id,
                artifact_path,
                iter(lambda: f.read(65536), b""),
                content_type,
            )

    def list_artifacts(self, run_id: str, path: str = "") -> list[ArtifactInfo]:
        """List artifacts for a run, optionally scoped to a sub-path."""
        params: dict[str, str] = {"run_id": run_id}
        if path:
            params["path"] = path
        data = self._get("/artifacts/list", params)
        return [
            ArtifactInfo(
                path=f.get("path", ""),
                is_dir=f.get("is_dir", False),
                file_size=f.get("file_size", 0),
            )
            for f in data.get("files", [])
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)
