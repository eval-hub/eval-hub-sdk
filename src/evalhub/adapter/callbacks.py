"""Default callback implementation for adapters."""

import logging
from pathlib import Path
from typing import Any

from ..models.api import JobStatus
from .models import (
    JobCallbacks,
    JobResults,
    JobStatusUpdate,
    OCIArtifactResult,
    OCIArtifactSpec,
)
from .oci import OCIArtifactPersister

logger = logging.getLogger(__name__)


class DefaultCallbacks(JobCallbacks):
    """Default callback implementation.

    This implementation:
    - Reports status updates to sidecar (if available) or logs them
    - Pushes OCI artifacts directly using OCIArtifactPersister

    This is the recommended callback implementation for both production and development.

    Example:
        ```python
        # Production (with evalhub for status updates)
        callbacks = DefaultCallbacks(
            job_id="my-job-123",
            benchmark_id="mmlu",
            provider_id="lm_evaluation_harness",
            sidecar_url="http://localhost:8080",
            registry_url="ghcr.io",
            registry_username=os.getenv("REGISTRY_USER"),
            registry_password=os.getenv("REGISTRY_TOKEN")
        )

        # Local development (no evalhub, just logging)
        callbacks = DefaultCallbacks(
            job_id="my-job-123",
            benchmark_id="mmlu",
            registry_url="localhost:5000",
            insecure=True
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
        sidecar_url: str | None = None,
        registry_url: str | None = None,
        registry_username: str | None = None,
        registry_password: str | None = None,
        insecure: bool = False,
        auth_token: str | None = None,
        auth_token_path: Path | str | None = None,
        ca_bundle_path: Path | str | None = None,
        events_path_template: str | None = None,
    ):
        """Initialize default callbacks.

        Args:
            job_id: Job identifier for API endpoint construction.
            benchmark_id: Benchmark identifier for status event validation.
            provider_id: Provider identifier (optional). If not provided, status updates
                        will not include provider_id field.
            sidecar_url: URL of evalhub service for status updates (optional).
                        If None, status updates are logged locally.
            registry_url: OCI registry URL (e.g., "ghcr.io")
            registry_username: Registry username
            registry_password: Registry password/token
            insecure: Allow insecure HTTP connections (both registry and evalhub)
            auth_token: Explicit authentication token (overrides auto-detection)
            auth_token_path: Path to authentication token file (e.g., ServiceAccount token)
                           If not provided, auto-detects Kubernetes ServiceAccount token
            ca_bundle_path: Path to CA bundle for TLS verification
                          If not provided, auto-detects OpenShift/Kubernetes CA bundles
        """
        self.job_id = job_id
        self.benchmark_id = benchmark_id
        self.provider_id = provider_id
        self.sidecar_url = sidecar_url.rstrip("/") if sidecar_url else None
        self._events_path_template = (
            events_path_template
            if events_path_template is not None
            else "/api/v1/evaluations/jobs/{job_id}/events"
        )

        # Initialize OCI persister
        self.persister = OCIArtifactPersister(
            registry_url=registry_url,
            username=registry_username,
            password=registry_password,
            insecure=insecure,
        )

        # Store insecure flag for evalhub communication
        self._insecure = insecure

        # Auto-detect or load authentication token
        self._auth_token = self._resolve_auth_token(auth_token, auth_token_path)

        # Auto-detect or load CA bundle (only if TLS verification is enabled)
        if insecure:
            self._ca_bundle = None
            logger.warning("TLS verification disabled - skipping CA bundle detection")
        else:
            self._ca_bundle = self._resolve_ca_bundle(ca_bundle_path)

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

    def _resolve_auth_token(
        self, explicit_token: str | None, token_path: Path | str | None
    ) -> str | None:
        """Resolve authentication token with auto-detection.

        Priority:
        1. Explicit token parameter
        2. Token from specified file path
        3. Auto-detected Kubernetes ServiceAccount token
        4. None (local mode, no authentication)

        Args:
            explicit_token: Explicit token string
            token_path: Path to token file

        Returns:
            Token string or None
        """
        # Use explicit token if provided
        if explicit_token:
            return explicit_token

        # Try specified token path
        if token_path:
            path = Path(token_path)
            if path.exists():
                return path.read_text().strip()
            logger.warning(f"Specified token path does not exist: {token_path}")

        # Auto-detect Kubernetes ServiceAccount token
        default_token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        if default_token_path.exists():
            logger.debug("Auto-detected Kubernetes ServiceAccount token")
            return default_token_path.read_text().strip()

        # No token available (local mode)
        logger.debug("No authentication token found - running in local mode")
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

    def _create_http_client(self) -> Any:
        """Create httpx client with authentication and TLS configuration.

        Returns:
            httpx.Client: Configured HTTP client
        """
        # Build headers
        headers: dict[str, str] = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
            logger.debug("HTTP client configured with Bearer token authentication")

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
            headers=headers,
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
                    "benchmark_id": self.benchmark_id,
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

                data = {"status_event": status_event}

                response = self._http_client.post(url, json=data, timeout=10.0)
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
                        "permissions for services/proxy resource with create/update verbs"
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

        The SDK always handles OCI pushing directly, regardless of whether
        a sidecar is present.

        Args:
            spec: Artifact specification

        Returns:
            OCIArtifactResult: Result with digest and reference

        Raises:
            RuntimeError: If artifact creation fails
        """
        logger.info(f"Creating OCI artifact for job {spec.job_id}")
        return self.persister.persist(spec)

    def report_results(self, results: JobResults) -> None:
        """Report final evaluation results to evalhub or log them.

        This sends the complete results including metrics to the evalhub service.

        Args:
            results: Final job results to report
        """
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
                    "benchmark_id": self.benchmark_id,
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

                # Include OCI artifact reference if available
                if results.oci_artifact:
                    status_event["artifacts"] = {
                        "oci_reference": results.oci_artifact.reference,
                        "oci_digest": results.oci_artifact.digest,
                        "size_bytes": results.oci_artifact.size_bytes,
                    }

                data = {"status_event": status_event}

                response = self._http_client.post(url, json=data, timeout=10.0)
                response.raise_for_status()

                logger.info(
                    f"Results reported to evalhub | "
                    f"Metrics: {len(metrics)} | "
                    f"Score: {results.overall_score}"
                )

            except self.httpx.HTTPStatusError as e:
                logger.error(
                    f"Failed to send results to evalhub (HTTP {e.response.status_code}): {e}"
                )
                # Fall through to local logging
            except Exception as e:
                logger.error(f"Failed to send results to evalhub: {e}")
                # Fall through to local logging

        # Local logging
        logger.info(
            f"Job {results.job_id} completed | "
            f"Benchmark: {results.benchmark_id} | "
            f"Model: {results.model_name} | "
            f"Score: {results.overall_score} | "
            f"Examples: {results.num_examples_evaluated} | "
            f"Duration: {results.duration_seconds:.2f}s"
        )
