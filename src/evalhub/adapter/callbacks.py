"""Default callback implementation for adapters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evalhub.adapter.models.adapter import FrameworkAdapter

from ..models.api import JobStatus
from .config import EvalHubMode, MlflowBackend
from .mlflow import MlflowArtifact
from .models import (
    EnvironmentCardMetadata,
    JobCallbacks,
    JobResults,
    JobSpec,
    JobStatusUpdate,
    OCIArtifactResult,
    OCIArtifactSpec,
)
from .oci import DEFAULT_OCI_PROXY_HOST, OCIArtifactPersister
from .oci.persister import OCIArtifactContext

logger = logging.getLogger(__name__)


class _MlflowOps:
    """Single-method MLflow integration.

    Usage from an adapter::

        from evalhub.adapter.mlflow import MlflowArtifact

        rid = callbacks.mlflow.save(
            results,
            job_spec,
            artifacts=[
                MlflowArtifact("results.json", json_bytes, "application/json"),
                MlflowArtifact("report.html", html_bytes, "text/html"),
            ],
        )
        if rid:
            results.mlflow_run_id = rid

    Metrics, params, and all artifacts are saved in a single MLflow run.
    Does nothing if ``job_spec.experiment_name`` is not set (returns ``None``).

    Returns the MLflow run id when a run is created. Assign it to
    ``results.mlflow_run_id`` before ``callbacks.report_results(results)`` so
    Eval Hub stores the link.

    The backend is controlled by the ``backend`` constructor argument or the
    ``EVALHUB_MLFLOW_BACKEND`` environment variable:

    - ``odh`` (default): lightweight built-in client, no extra dependencies.
    - ``upstream``: official ``mlflow`` library; requires ``mlflow`` or
      ``mlflow-skinny`` to be installed.
    """

    def __init__(self, backend: MlflowBackend = MlflowBackend.ODH) -> None:
        self._backend = backend

    @staticmethod
    def _build_run_name(job_spec: JobSpec) -> str:
        return f"{job_spec.id}_{job_spec.benchmark_index}"

    def save(
        self,
        results: JobResults,
        job_spec: JobSpec,
        artifacts: list[MlflowArtifact] | None = None,
    ) -> str | None:
        if not job_spec.experiment_name:
            logger.debug("No MLflow experiment configured, skipping")
            return None

        try:
            if self._backend == MlflowBackend.UPSTREAM:
                return self._save_upstream(results, job_spec, artifacts)
            return self._save_odh(results, job_spec, artifacts)
        except Exception as e:
            logger.error("Failed to save to MLflow: %s", e)
            raise RuntimeError(f"MLflow save failed: {e}") from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_params_metrics(
        results: JobResults,
        job_spec: JobSpec,
    ) -> tuple[list, list]:
        from .mlflow import Metric, Param, sanitize_metric_key_for_api

        params = [
            Param("benchmark_id", results.benchmark_id),
            Param("provider_id", job_spec.provider_id),
            Param("model_name", results.model_name),
            Param("num_examples_evaluated", str(results.num_examples_evaluated)),
            Param("duration_seconds", str(results.duration_seconds)),
        ]
        # MLflow rejects commas etc. in metric keys; Eval Hub keeps r.metric_name as-is.
        metrics: list[Metric] = [
            Metric(sanitize_metric_key_for_api(r.metric_name), float(r.metric_value))
            for r in results.results
            if isinstance(r.metric_value, int | float)
        ]
        if results.overall_score is not None:
            metrics.append(Metric("overall_score", results.overall_score))

        # EvalCard → MLflow params
        if results.eval_card:
            ec = results.eval_card
            if ec.languages_count is not None:
                params.append(
                    Param("eval_card.languages_count", str(ec.languages_count))
                )
            if ec.languages:
                params.append(Param("eval_card.languages", ",".join(ec.languages)))
            if ec.modalities_input:
                params.append(
                    Param("eval_card.modalities_input", ",".join(ec.modalities_input))
                )
            if ec.capability_evaluations:
                first = ec.capability_evaluations[0]
                params.append(Param("eval_card.primary_ability", first.ability))
                params.append(Param("eval_card.primary_benchmark", first.benchmark))

        # Environment Card → MLflow params
        if results.env_card:
            env = results.env_card
            for field in (
                "python_version",
                "framework_name",
                "framework_version",
                "cuda_version",
                "gpu_model",
                "gpu_driver_version",
                "os_info",
                "container_image",
                "model_id",
                "model_version",
                "model_provider",
                "dataset_hash",
            ):
                val = getattr(env, field, None)
                if val is not None:
                    params.append(Param(f"env_card.{field}", str(val)))
            if env.gpu_count is not None:
                params.append(Param("env_card.gpu_count", str(env.gpu_count)))
            if env.key_packages:
                import json

                params.append(
                    Param("env_card.key_packages", json.dumps(env.key_packages))
                )

        return params, metrics

    def _save_odh(
        self,
        results: JobResults,
        job_spec: JobSpec,
        artifacts: list[MlflowArtifact] | None,
    ) -> str:
        from .mlflow import MlflowClient

        params, metrics = self._build_params_metrics(results, job_spec)
        run_tags: dict[str, str] = {
            tag["key"]: tag["value"] for tag in (job_spec.tags or [])
        }

        run_id: str = ""
        with MlflowClient() as client:
            experiment_id = client.get_or_create_experiment(
                job_spec.experiment_name or ""
            )
            with client.start_run(
                experiment_id,
                run_name=self._build_run_name(job_spec),
                tags=run_tags,
            ) as rid:
                run_id = rid
                client.log_batch(run_id, metrics=metrics, params=params)
                for artifact in artifacts or []:
                    client.upload_artifact(
                        run_id,
                        artifact.path,
                        artifact.content,
                        artifact.content_type,
                    )

        logger.info(
            "Saved to MLflow (odh) experiment '%s' (run_id: %s) — "
            "%d metric(s), %d artifact(s)",
            job_spec.experiment_name,
            run_id,
            len(metrics),
            len(artifacts or []),
        )
        return run_id

    def _save_upstream(
        self,
        results: JobResults,
        job_spec: JobSpec,
        artifacts: list[MlflowArtifact] | None,
    ) -> str:
        import tempfile
        from pathlib import Path as _Path

        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "EVALHUB_MLFLOW_BACKEND=upstream requires the 'mlflow' package. "
                "Install it with: pip install mlflow-skinny"
            ) from exc

        params, metrics = self._build_params_metrics(results, job_spec)
        run_tags: dict[str, str] = {
            tag["key"]: tag["value"] for tag in (job_spec.tags or [])
        }

        mlflow.set_experiment(job_spec.experiment_name)
        run_id = ""
        with mlflow.start_run(
            run_name=self._build_run_name(job_spec), tags=run_tags
        ) as active_run:
            run_id = str(active_run.info.run_id)
            mlflow.log_params({p.key: p.value for p in params})
            mlflow.log_metrics({m.key: m.value for m in metrics})

            for artifact in artifacts or []:
                artifact_file = _Path(artifact.path)
                artifact_dir = (
                    str(artifact_file.parent)
                    if str(artifact_file.parent) != "."
                    else None
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_file = _Path(tmpdir) / artifact_file.name
                    tmp_file.write_bytes(artifact.content)
                    mlflow.log_artifact(str(tmp_file), artifact_path=artifact_dir)

        logger.info(
            "Saved to MLflow (upstream) experiment '%s' (run_id: %s) — "
            "%d metric(s), %d artifact(s)",
            job_spec.experiment_name,
            run_id,
            len(metrics),
            len(artifacts or []),
        )
        return run_id


class DefaultCallbacks(JobCallbacks):
    """Default callback implementation.

    This implementation:
    - Reports status updates to sidecar (if available) or logs them
    - Pushes OCI artifacts directly using OCIArtifactPersister
    - ``report_results(results)``: POSTs final results to Eval Hub; if
      ``results.mlflow_run_id`` is set (for example from ``save()``), that id
      is included (if unset, the field is left out).

    Example::

        rid = callbacks.mlflow.save(results, job_spec)
        if rid:
            results.mlflow_run_id = rid
        callbacks.report_results(results)

    This is the recommended callback implementation for both production and development.

    Example:
        ```python
        # Production (with evalhub for status updates)
        callbacks = DefaultCallbacks(
            job_id="my-job-123",
            benchmark_id="mmlu",
            benchmark_index=0,
            provider_id="lm_evaluation_harness",
            sidecar_url="http://localhost:8080",
            oci_auth_config_path=Path("~/.docker/config.json"),
        )

        # Local development (no evalhub, just logging)
        callbacks = DefaultCallbacks(
            job_id="my-job-123",
            benchmark_id="mmlu",
            benchmark_index=0,
            oci_insecure=True,
        )

        adapter = MyAdapter()
        results = adapter.run_benchmark_job(spec, callbacks)
        ```
    """

    def __init__(
        self,
        job_id: str,
        benchmark_id: str,
        provider_id: str | None = None,
        benchmark_index: int = 0,
        sidecar_url: str | None = None,
        insecure: bool = False,
        auth_token: str | None = None,
        auth_token_path: Path | str | None = None,
        ca_bundle_path: Path | str | None = None,
        events_path_template: str | None = None,
        oci_auth_config_path: Path | None = None,
        oci_insecure: bool = False,
        oci_proxy_host: str | None = None,
        mlflow_backend: MlflowBackend = MlflowBackend.ODH,
    ):
        """Initialize default callbacks.

        Args:
            job_id: Job identifier for API endpoint construction.
            benchmark_id: Benchmark identifier for status event validation.
            provider_id: Provider identifier (optional). If not provided, status updates
                        will not include provider_id field.
            benchmark_index: Index of this benchmark within the job (default 0).
            sidecar_url: URL of evalhub service for status updates (optional).
                        If None, status updates are logged locally.
            insecure: Allow insecure HTTP connections (evalhub)
            auth_token: Explicit authentication token (overrides auto-detection)
            auth_token_path: Path to authentication token file (e.g., ServiceAccount token)
                           If not provided, auto-detects Kubernetes ServiceAccount token
            ca_bundle_path: Path to CA bundle for TLS verification
                          If not provided, auto-detects OpenShift/Kubernetes CA bundles
            oci_proxy_host: OCI proxy host for k8s sidecar mode (e.g. "localhost:8080").
                          When set, the OCI persister pushes to this host instead of the
                          real registry and skips Python-side auth (the sidecar handles
                          authentication). The returned artifact references still use the
                          original registry host. Automatically set via from_adapter()
                          when mode is K8S.
            mlflow_backend: MLflow client backend to use for artifact saving.
                           Use MlflowBackend.ODH (default) for the built-in client or
                           MlflowBackend.UPSTREAM for the official mlflow library.
                           Can also be set via EVALHUB_MLFLOW_BACKEND env var when
                           constructing via from_adapter().
        """
        self.job_id = job_id
        self.benchmark_id = benchmark_id
        self.provider_id = provider_id
        self.benchmark_index = benchmark_index
        self.sidecar_url = sidecar_url.rstrip("/") if sidecar_url else None
        self._events_path_template = (
            events_path_template
            if events_path_template is not None
            else "/api/v1/evaluations/jobs/{job_id}/events"
        )

        # Initialize OCI persister
        self.persister = OCIArtifactPersister(
            context=OCIArtifactContext(
                job_id=job_id,
                benchmark_id=benchmark_id,
                provider_id=provider_id,
                benchmark_index=benchmark_index,
            ),
            oci_auth_config_path=oci_auth_config_path,
            oci_insecure=oci_insecure,
            oci_proxy_host=oci_proxy_host,
        )

        # Store insecure flag for evalhub communication
        self._insecure = insecure

        # Store auth token source for per-request reading
        self._explicit_auth_token = auth_token
        self._auth_token_path = self._resolve_auth_token_path(auth_token_path)

        # Auto-detect or load CA bundle (only if TLS verification is enabled)
        if insecure:
            self._ca_bundle = None
            logger.warning("TLS verification disabled - skipping CA bundle detection")
        else:
            self._ca_bundle = self._resolve_ca_bundle(ca_bundle_path)

        # MLflow integration (single-method API)
        self.mlflow = _MlflowOps(backend=mlflow_backend)

        # Try to import httpx for sidecar communication
        self._httpx_available = False
        self._http_client: Any | None = None
        if self.sidecar_url:
            try:
                import httpx

                self.httpx = httpx
                self._httpx_available = True
                self._http_client = self._create_http_client()
            except ImportError:
                logger.warning(
                    "httpx not installed. Status updates will be logged locally. "
                    "Install with: pip install httpx"
                )

    @staticmethod
    def _resolve_auth_token_path(token_path: Path | str | None) -> Path | None:
        """Resolve the path to the authentication token file.

        Priority:
        1. Specified token path (if it exists)
        2. Auto-detected Kubernetes ServiceAccount token
        3. None (local mode, no authentication)

        Args:
            token_path: Path to token file

        Returns:
            Path to token file or None
        """
        if token_path:
            path = Path(token_path)
            if path.exists():
                return path
            logger.warning(f"Specified token path does not exist: {token_path}")

        default_token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        if default_token_path.exists():
            logger.debug("Auto-detected Kubernetes ServiceAccount token")
            return default_token_path

        logger.debug("No authentication token found - running in local mode")
        return None

    def _read_auth_token(self) -> str | None:
        """Read the authentication token fresh from disk (or return explicit token).

        Returns:
            Token string or None
        """
        if self._explicit_auth_token:
            return self._explicit_auth_token

        if self._auth_token_path:
            try:
                return self._auth_token_path.read_text().strip() or None
            except OSError:
                logger.warning(f"Failed to read token from {self._auth_token_path}")
                return None

        return None

    def _resolve_ca_bundle(self, ca_bundle_path: Path | str | None) -> Path | None:
        """Resolve CA bundle path with auto-detection.

        Priority:
        1. Explicitly specified CA bundle path
        2. Auto-detected OpenShift service-ca
        3. Auto-detected Kubernetes ServiceAccount CA
        4. None (use system defaults or insecure mode)

        Args:
            ca_bundle_path: Path to CA bundle file

        Returns:
            Path to CA bundle or None
        """
        # Use explicit CA bundle if provided
        if ca_bundle_path:
            path = Path(ca_bundle_path)
            if path.exists():
                return path
            logger.warning(f"Specified CA bundle does not exist: {ca_bundle_path}")

        # Try common CA bundle locations
        ca_paths = [
            Path("/etc/pki/ca-trust/source/anchors/service-ca.crt"),  # OpenShift
            Path(
                "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
            ),  # OpenShift SA mount
            Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"),  # Kubernetes
        ]

        for path in ca_paths:
            if path.exists():
                logger.debug(f"Auto-detected CA bundle at: {path}")
                return path

        # No CA bundle found (use system defaults)
        logger.debug("No CA bundle found - using system defaults")
        return None

    @staticmethod
    def _resolve_namespace() -> str | None:
        """Read the pod namespace from the ServiceAccount mount."""
        ns_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
        if ns_path.exists():
            try:
                return ns_path.read_text().strip() or None
            except OSError:
                return None
        return None

    def _request_headers(self) -> dict[str, str]:
        """Build per-request headers with fresh auth token and tenant namespace."""
        headers: dict[str, str] = {}

        token = self._read_auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        namespace = self._resolve_namespace()
        if namespace:
            headers["X-Tenant"] = namespace

        return headers

    def _create_http_client(self) -> Any:
        """Create httpx client with TLS configuration.

        Auth headers are added per-request via _request_headers() so that
        rotated ServiceAccount tokens are picked up automatically.

        Returns:
            httpx.Client: Configured HTTP client
        """
        # Determine TLS verification settings
        verify: bool | str
        if self._insecure:
            verify = False
            logger.warning("TLS verification disabled (insecure mode)")
        elif self._ca_bundle:
            verify = str(self._ca_bundle)
            logger.debug(f"TLS verification using CA bundle: {self._ca_bundle}")
        else:
            verify = True  # Use system CA certificates
            logger.debug("TLS verification using system CA certificates")

        return self.httpx.Client(
            verify=verify,
            timeout=30.0,
        )

    def report_status(self, update: JobStatusUpdate) -> None:
        """Report status update to evalhub or log it.

        Args:
            update: Status update to report
        """
        # If evalhub available, send status update
        if self.sidecar_url and self._httpx_available and self._http_client:
            try:
                url = f"{self.sidecar_url}{self._events_path_template.format(job_id=self.job_id)}"

                # Transform to eval-hub API format
                status_event = {
                    "id": self.benchmark_id,
                    "benchmark_index": self.benchmark_index,
                    "state": update.status.value,
                    "status": update.status.value,
                    "message": update.message.model_dump(mode="json"),
                }

                # Include error details for failed updates
                if update.error:
                    status_event["error_message"] = update.error.model_dump(mode="json")

                # Include provider_id if available
                if self.provider_id:
                    status_event["provider_id"] = self.provider_id

                data = {"benchmark_status_event": status_event}
                logger.debug("Events report_status body: %s", data)

                response = self._http_client.post(
                    url,
                    json=data,
                    headers=self._request_headers(),
                    timeout=10.0,
                )
                response.raise_for_status()

                logger.debug(f"Status update sent to evalhub: {update.status}")
                return

            except self.httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    logger.error(
                        "Authentication failed (401). Ensure the job has a valid "
                        "ServiceAccount token at /var/run/secrets/kubernetes.io/serviceaccount/token"
                    )
                elif e.response.status_code == 403:
                    logger.error(
                        "Authorization failed (403). Ensure the ServiceAccount has RBAC "
                        "permissions for evaluations resource in the trustyai.opendatahub.io API group"
                    )
                else:
                    logger.warning(f"Failed to send status to evalhub: {e}")
                # Fall through to local logging
            except Exception as e:
                logger.warning(f"Failed to send status to evalhub: {e}")
                # Fall through to local logging

        # Local logging
        logger.info(
            f"Status: {update.status.value} | "
            f"Phase: {update.phase.value if update.phase else 'N/A'} | "
            f"Progress: {update.progress or 'N/A'} | "
            f"Message: {update.message or ''}"
        )

    def create_oci_artifact(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
        """Create OCI artifact using the SDK persister.

        Args:
            spec: Artifact specification

        Returns:
            OCIArtifactResult: Result with digest and reference

        Raises:
            RuntimeError: If artifact creation fails
        """
        logger.info(f"Creating OCI artifact for job {self.job_id}")
        result = self.persister.persist(spec)
        logger.info(f"Created OCI artifact for job {self.job_id} as: {result}")
        return result

    def report_results(self, results: JobResults) -> None:
        """Report final evaluation results to evalhub or log them.

        This sends the complete results including metrics to the evalhub service.
        If the provider did not supply an Environment Card, a best-effort card
        is auto-captured from the current runtime.

        Args:
            results: Final job results to report
        """
        # Resolve the Environment Card without mutating the caller's results object.
        # If the provider did not supply one, capture a best-effort card locally.
        env_card = results.env_card
        if env_card is None:
            try:
                env_card = EnvironmentCardMetadata.capture()
                logger.info(
                    "Environment Card auto-captured (completeness: %.0f%%)",
                    (env_card.capture_completeness or 0) * 100,
                )
            except Exception:
                logger.debug("Environment Card auto-capture failed", exc_info=True)

        # If evalhub available, send results with completed status event
        if self.sidecar_url and self._httpx_available and self._http_client:
            try:
                url = f"{self.sidecar_url}{self._events_path_template.format(job_id=self.job_id)}"

                # Convert evaluation results to metrics map
                metrics = {}
                for result in results.results:
                    metrics[result.metric_name] = result.metric_value

                # Build status event with results
                status_event = {
                    "id": self.benchmark_id,
                    "benchmark_index": self.benchmark_index,
                    "state": JobStatus.COMPLETED.value,
                    "status": JobStatus.COMPLETED.value,
                    "message": {
                        "message": "Evaluation completed successfully",
                        "message_code": "evaluation_completed",
                    },
                    "metrics": metrics,
                    "completed_at": results.completed_at.isoformat(),
                    "duration_seconds": int(results.duration_seconds),
                }

                # Include provider_id if available
                if self.provider_id:
                    status_event["provider_id"] = self.provider_id

                if results.mlflow_run_id:
                    status_event["mlflow_run_id"] = results.mlflow_run_id

                # Build artifacts dict: OCI first, then card metadata
                artifacts: dict[str, Any] = {}
                if results.oci_artifact:
                    artifacts["oci_reference"] = results.oci_artifact.reference
                    artifacts["oci_digest"] = results.oci_artifact.digest
                if results.eval_card:
                    artifacts["evalhub.eval_card"] = results.eval_card.model_dump(
                        exclude_none=True
                    )
                if env_card:
                    artifacts["evalhub.env_card"] = env_card.model_dump(
                        exclude_none=True
                    )

                eval_meta_artifacts = (results.evaluation_metadata or {}).get(
                    "artifacts", {}
                )
                if eval_meta_artifacts:
                    artifacts.update(eval_meta_artifacts)

                if artifacts:
                    status_event["artifacts"] = artifacts

                data = {"benchmark_status_event": status_event}
                logger.debug("Events report_results body: %s", data)

                response = self._http_client.post(
                    url,
                    json=data,
                    headers=self._request_headers(),
                    timeout=10.0,
                )
                response.raise_for_status()

                logger.info(
                    f"Results reported to evalhub | "
                    f"Metrics: {len(metrics)} | "
                    f"Score: {results.overall_score}"
                )

            except self.httpx.HTTPStatusError as e:
                logger.error(
                    "Failed to send results to evalhub (HTTP %s): %s",
                    e.response.status_code,
                    e,
                )
                # Fall through to local logging
            except Exception as e:
                logger.error("Failed to send results to evalhub: %s", e)
                # Fall through to local logging

        # Local logging
        logger.info(
            f"Job {results.id} completed | "
            f"Benchmark: {results.benchmark_id} | "
            f"Model: {results.model_name} | "
            f"Score: {results.overall_score} | "
            f"Examples: {results.num_examples_evaluated} | "
            f"Duration: {results.duration_seconds:.2f}s"
        )

    @staticmethod
    def from_adapter(adapter: FrameworkAdapter) -> DefaultCallbacks:
        """convenience method, and do not store adapter instance"""
        return DefaultCallbacks(
            job_id=adapter.job_spec.id,
            provider_id=adapter.job_spec.provider_id,
            benchmark_id=adapter.job_spec.benchmark_id,
            benchmark_index=adapter.job_spec.benchmark_index,
            sidecar_url=adapter.job_spec.callback_url,
            insecure=adapter.settings.evalhub_insecure,
            oci_auth_config_path=adapter.settings.oci_auth_config_path,
            oci_insecure=adapter.settings.oci_insecure,
            oci_proxy_host=(
                DEFAULT_OCI_PROXY_HOST
                if adapter.settings.mode == EvalHubMode.K8S
                else None
            ),
            mlflow_backend=adapter.settings.mlflow_backend,
        )
