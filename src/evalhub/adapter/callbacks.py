"""Default callback implementation for adapters."""

import logging

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
        # Production (with sidecar for status updates)
        callbacks = DefaultCallbacks(
            sidecar_url="http://localhost:8080",
            registry_url="ghcr.io",
            registry_username=os.getenv("REGISTRY_USER"),
            registry_password=os.getenv("REGISTRY_TOKEN")
        )

        # Local development (no sidecar, just logging)
        callbacks = DefaultCallbacks(
            registry_url="localhost:5000",
            insecure=True
        )

        adapter = MyAdapter()
        results = adapter.run_benchmark_job(spec, callbacks)
        ```
    """

    def __init__(
        self,
        sidecar_url: str | None = None,
        registry_url: str | None = None,
        registry_username: str | None = None,
        registry_password: str | None = None,
        insecure: bool = False,
    ):
        """Initialize default callbacks.

        Args:
            sidecar_url: URL of sidecar service for status updates (optional).
                        If None, status updates are logged locally.
            registry_url: OCI registry URL (e.g., "ghcr.io")
            registry_username: Registry username
            registry_password: Registry password/token
            insecure: Allow insecure HTTP connections to registry
        """
        self.sidecar_url = sidecar_url.rstrip("/") if sidecar_url else None

        # Initialize OCI persister
        self.persister = OCIArtifactPersister(
            registry_url=registry_url,
            username=registry_username,
            password=registry_password,
            insecure=insecure,
        )

        # Try to import httpx for sidecar communication
        self._httpx_available = False
        if self.sidecar_url:
            try:
                import httpx

                self.httpx = httpx
                self._httpx_available = True
            except ImportError:
                logger.warning(
                    "httpx not installed. Status updates will be logged locally. "
                    "Install with: pip install httpx"
                )

    def report_status(self, update: JobStatusUpdate) -> None:
        """Report status update to sidecar or log it.

        Args:
            update: Status update to report
        """
        # If sidecar available, send status update
        if self.sidecar_url and self._httpx_available:
            try:
                url = f"{self.sidecar_url}/status"
                data = update.model_dump(mode="json", exclude_none=True)

                response = self.httpx.post(url, json=data, timeout=10.0)
                response.raise_for_status()

                logger.debug(f"Status update sent to sidecar: {update.status}")
                return

            except Exception as e:
                logger.warning(f"Failed to send status to sidecar: {e}")
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
        """Report final evaluation results to sidecar or log them.

        Args:
            results: Final job results to report
        """
        # If sidecar available, send results
        if self.sidecar_url and self._httpx_available:
            try:
                url = f"{self.sidecar_url}/results"
                data = results.model_dump(mode="json", exclude_none=True)

                response = self.httpx.post(url, json=data, timeout=30.0)
                response.raise_for_status()

                logger.info(f"Results for job {results.job_id} sent to sidecar")
                return

            except Exception as e:
                logger.warning(f"Failed to send results to sidecar: {e}")
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
