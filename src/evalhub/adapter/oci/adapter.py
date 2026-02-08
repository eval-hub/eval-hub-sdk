"""Bridge between new adapter and original OCI persister."""

import asyncio
import logging
from datetime import UTC, datetime

from ...models.api import (
    BenchmarkConfig,
    EvaluationJob,
    EvaluationJobFilesLocation,
    EvaluationJobResource,
    EvaluationJobStatus,
    JobStatus,
    ModelConfig,
    OCICoordinate,
)
from ..models import OCIArtifactResult, OCIArtifactSpec
from .persister import OCIArtifactPersister as OriginalPersister

logger = logging.getLogger(__name__)


class OCIArtifactPersister:
    """Adapter bridging new adapter models to original OCI persister."""

    def __init__(
        self,
        registry_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        insecure: bool = False,
    ):
        """Initialize OCI persister.

        Args:
            registry_url: OCI registry URL
            username: Registry username
            password: Registry password
            insecure: Allow insecure connections
        """
        self.registry_url = registry_url or "localhost:5000"
        self._persister = OriginalPersister()

    def persist(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
        """Persist artifact specification.

        Args:
            spec: Artifact specification

        Returns:
            OCIArtifactResult: Persistence result
        """
        return asyncio.run(self._persist_async(spec))

    async def _persist_async(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
        """Persist artifact asynchronously.

        Args:
            spec: Artifact specification

        Returns:
            OCIArtifactResult: Persistence result
        """
        files_location = EvaluationJobFilesLocation(
            id=spec.id,
            path=str(spec.base_path or spec.files[0].parent),
        )

        coordinate = OCICoordinate(
            oci_ref=f"{self.registry_url}/eval-results/{spec.benchmark_id}:{spec.id}",
            oci_subject=None,
        )

        # Create job structure for OCI persistence (adapter SDK use)
        job_resource = EvaluationJobResource(
            id=spec.id,
            tenant="default",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        job_status = EvaluationJobStatus(
            state=JobStatus.RUNNING,
        )

        benchmark_config = BenchmarkConfig(
            id=spec.benchmark_id, provider_id="unknown", parameters={}
        )

        job = EvaluationJob(
            resource=job_resource,
            status=job_status,
            model=ModelConfig(url="http://localhost", name=spec.model_name),
            benchmarks=[benchmark_config],
        )

        response = await self._persister.persist(files_location, coordinate, job)

        return OCIArtifactResult(
            digest=response.digest,
            reference=response.oci_ref,
            size_bytes=response.files_count * 1024,
        )
