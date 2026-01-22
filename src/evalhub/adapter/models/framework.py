"""Framework adapter models and base classes."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from evalhub.adapter.oci.persister import Persister

from ...models.api import (
    BenchmarkInfo,
    EvaluationJob,
    EvaluationJobFilesLocation,
    EvaluationRequest,
    EvaluationResponse,
    FrameworkInfo,
    HealthResponse,
    JobStatus,
    OCICoordinate,
    PersistResponse,
)

logger = logging.getLogger(__name__)


class AdapterConfig(BaseModel):
    """Base configuration for framework adapters."""

    # Adapter identification
    framework_id: str = Field(..., description="Unique framework identifier")
    adapter_name: str = Field(..., description="Adapter display name")
    version: str = Field(default="1.0.0", description="Adapter version")

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host to bind to")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")

    # Framework-specific settings
    framework_config: dict[str, Any] = Field(
        default_factory=dict, description="Framework-specific configuration"
    )

    # Resource limits
    max_concurrent_jobs: int = Field(
        default=10, description="Maximum concurrent evaluation jobs"
    )
    job_timeout_seconds: int = Field(
        default=3600, description="Maximum job execution time"
    )
    memory_limit_gb: float | None = Field(
        default=None, description="Memory limit in GB"
    )

    # Logging and monitoring
    log_level: str = Field(default="INFO", description="Logging level")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    health_check_interval: int = Field(
        default=30, description="Health check interval in seconds"
    )

    class Config:
        """Pydantic configuration."""

        extra = "allow"


class AdapterMetadata(BaseModel):
    """Metadata for framework adapters."""

    # Basic information
    name: str = Field(..., description="Adapter name")
    description: str | None = Field(None, description="Adapter description")
    version: str = Field(..., description="Adapter version")
    author: str | None = Field(None, description="Adapter author")

    # Framework information
    framework_name: str = Field(..., description="Name of the wrapped framework")
    framework_version: str = Field(..., description="Version of the wrapped framework")
    framework_url: str | None = Field(
        None, description="Framework documentation/repository URL"
    )

    # Capabilities
    supported_model_types: list[str] = Field(
        default_factory=list, description="Supported model types"
    )
    supported_metrics: list[str] = Field(
        default_factory=list, description="Supported evaluation metrics"
    )
    supports_batch_evaluation: bool = Field(
        True, description="Supports batch evaluation"
    )
    supports_few_shot: bool = Field(True, description="Supports few-shot evaluation")
    supports_custom_datasets: bool = Field(
        False, description="Supports custom datasets"
    )

    # Resource requirements
    min_memory_gb: float | None = Field(None, description="Minimum memory requirement")
    requires_gpu: bool = Field(False, description="Requires GPU")
    max_batch_size: int | None = Field(None, description="Maximum batch size")

    # Contact and documentation
    contact_email: str | None = Field(None, description="Contact email")
    documentation_url: str | None = Field(None, description="Documentation URL")
    repository_url: str | None = Field(None, description="Source repository URL")
    license: str | None = Field(None, description="License information")


class FrameworkAdapter(ABC):
    """Abstract base class for framework adapters.

    This class defines the interface that all framework adapters must implement
    to integrate with EvalHub via the SDK.
    """

    def __init__(self, config: AdapterConfig):
        """Initialize the adapter with configuration."""
        self.config = config
        self._jobs: dict[str, EvaluationJob] = {}
        self._shutdown_event = asyncio.Event()

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the framework adapter.

        This method should:
        - Load the underlying evaluation framework
        - Verify dependencies are available
        - Set up any required resources
        - Prepare for evaluation requests

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    async def get_framework_info(self) -> FrameworkInfo:
        """Get information about this framework adapter.

        Returns:
            FrameworkInfo: Metadata about the framework and its capabilities
        """
        pass

    @abstractmethod
    async def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List all available benchmarks.

        Returns:
            List[BenchmarkInfo]: Available benchmarks with their metadata
        """
        pass

    @abstractmethod
    async def get_benchmark_info(self, benchmark_id: str) -> BenchmarkInfo | None:
        """Get detailed information about a specific benchmark.

        Args:
            benchmark_id: The benchmark identifier

        Returns:
            BenchmarkInfo: Benchmark information, or None if not found
        """
        pass

    @abstractmethod
    async def submit_evaluation(self, request: EvaluationRequest) -> EvaluationJob:
        """Submit an evaluation job.

        Args:
            request: The evaluation request

        Returns:
            EvaluationJob: The created job with initial status

        Raises:
            ValueError: If request is invalid
            RuntimeError: If unable to submit job
        """
        pass

    @abstractmethod
    async def get_job_status(self, job_id: str) -> EvaluationJob | None:
        """Get the current status of an evaluation job.

        Args:
            job_id: The job identifier

        Returns:
            EvaluationJob: Current job status, or None if not found
        """
        pass

    @abstractmethod
    async def get_evaluation_results(self, job_id: str) -> EvaluationResponse | None:
        """Get the results of a completed evaluation.

        Args:
            job_id: The job identifier

        Returns:
            EvaluationResponse: Evaluation results, or None if not available
        """
        pass

    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running evaluation job.

        Args:
            job_id: The job identifier

        Returns:
            bool: True if job was cancelled, False if not found or already completed
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthResponse:
        """Perform a health check of the framework adapter.

        Returns:
            HealthResponse: Current health status and resource information
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the framework adapter.

        This method should:
        - Cancel any running jobs
        - Clean up resources
        - Save any necessary state
        """
        pass

    # Optional methods with default implementations

    async def stream_job_updates(
        self, job_id: str
    ) -> AsyncGenerator[EvaluationJob, None]:
        """Stream real-time updates for a job.

        Default implementation polls get_job_status. Framework adapters
        can override this to provide true streaming updates.

        Args:
            job_id: The job identifier

        Yields:
            EvaluationJob: Updated job status
        """
        while not self._shutdown_event.is_set():
            job = await self.get_job_status(job_id)
            if not job:
                break

            yield job

            if job.status in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                break

            await asyncio.sleep(1.0)  # Poll every second

    async def list_active_jobs(self) -> list[EvaluationJob]:
        """List all active evaluation jobs.

        Returns:
            List[EvaluationJob]: List of active jobs
        """
        active_jobs = []
        for job in self._jobs.values():
            if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                active_jobs.append(job)
        return active_jobs

    async def cleanup_completed_jobs(self, max_age_seconds: int = 3600) -> int:
        """Clean up old completed jobs.

        Args:
            max_age_seconds: Maximum age for completed jobs

        Returns:
            int: Number of jobs cleaned up
        """
        from datetime import datetime, timezone

        current_time = datetime.now(timezone.utc)
        cleaned_count = 0

        jobs_to_remove = []
        for job_id, job in self._jobs.items():
            if job.status in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                if job.completed_at:
                    age_seconds = (current_time - job.completed_at).total_seconds()
                    if age_seconds > max_age_seconds:
                        jobs_to_remove.append(job_id)
                        cleaned_count += 1

        for job_id in jobs_to_remove:
            del self._jobs[job_id]

        return cleaned_count

    def get_adapter_metadata(self) -> AdapterMetadata:
        """Get metadata about this adapter.

        Subclasses should override this to provide specific metadata.

        Returns:
            AdapterMetadata: Adapter metadata
        """
        return AdapterMetadata(
            name=self.config.adapter_name,
            description=f"Framework adapter for {self.config.framework_id}",
            version=self.config.version,
            framework_name=self.config.framework_id,
            framework_version="unknown",
            author=None,
            framework_url=None,
            supports_batch_evaluation=True,
            supports_few_shot=True,
            supports_custom_datasets=False,
            min_memory_gb=None,
            requires_gpu=False,
            max_batch_size=None,
            contact_email=None,
            documentation_url=None,
            repository_url=None,
            license=None,
        )

    async def job_files(self, job_id: str) -> EvaluationJobFilesLocation:
        """Default implementation returns empty files location.

        User should override to specify files location when opting-in for OCI persistence.

        Args:
            job_id: The job identifier

        Returns:
            EvaluationJobFilesLocation: Files location (path=None by default)
        """
        return EvaluationJobFilesLocation(job_id=job_id, path=None)

    async def persist_job_files_oci(
        self, job_id: str, coordinate: OCICoordinate
    ) -> PersistResponse | None:
        """Persist evaluation job files as OCI artifact (manual trigger).

        Args:
            job_id: The job identifier
            coordinate: OCI coordinates (reference and optional subject)

        Returns:
            PersistResponse: Persistence status and artifact digest
            None: If no files to persist

        Raises:
            ValueError: If job not found, OCI reference invalid, or job not in completed state
            RuntimeError: If persistence fails
        """
        # Validate job exists
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Validate job is completed
        if job.status != JobStatus.COMPLETED:
            raise ValueError(
                f"Job {job_id} is not completed (status: {job.status}). "
                "Only completed jobs can be persisted."
            )

        # Get files location
        files_location = await self.job_files(job_id)

        if not files_location.path:
            logger.info(f"No files to persist for job {job_id}")
            return None

        # Create OCI artifact
        persister = self._get_persister()

        return await persister.persist(
            files_location=files_location,
            coordinate=coordinate,
            job=job,
        )

    def _get_persister(self) -> Persister:
        """Get OCI artifact persister instance.

        This method can be overridden in tests or subclasses to provide
        a custom persister implementation.

        Returns:
            OCIArtifactPersister: The persister instance
        """
        from evalhub.adapter.oci.persister import OCIArtifactPersister

        return OCIArtifactPersister()
