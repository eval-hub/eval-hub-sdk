"""Simplified adapter models for benchmark job execution."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...models.api import EvaluationResult, JobStatus, ModelConfig


class JobPhase(str, Enum):
    """Job execution phases."""

    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    RUNNING_EVALUATION = "running_evaluation"
    POST_PROCESSING = "post_processing"
    PERSISTING_ARTIFACTS = "persisting_artifacts"
    COMPLETED = "completed"


class JobSpec(BaseModel):
    """Job specification loaded from ConfigMap at pod startup.

    This contains all the information needed to run a benchmark evaluation job.
    The service creates this and mounts it via ConfigMap when launching the job pod.
    """

    # Job identification
    job_id: str = Field(..., description="Unique job identifier from service")
    benchmark_id: str = Field(..., description="Benchmark to evaluate")

    # Model configuration
    model: ModelConfig = Field(..., description="Model configuration")

    # Evaluation parameters
    num_examples: int | None = Field(
        default=None, description="Number of examples to evaluate (None = all)"
    )
    num_few_shot: int | None = Field(
        default=None, description="Number of few-shot examples"
    )
    random_seed: int | None = Field(
        default=42, description="Random seed for reproducibility"
    )

    # Benchmark-specific configuration
    benchmark_config: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark-specific parameters"
    )

    # Job metadata
    experiment_name: str | None = Field(
        default=None, description="Name for this evaluation experiment"
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="Custom tags for the job"
    )

    # Resource hints
    timeout_seconds: int | None = Field(
        default=3600, description="Maximum job execution time"
    )
    memory_limit_gb: float | None = Field(default=None, description="Memory limit hint")


class JobStatusUpdate(BaseModel):
    """Status update sent to service via callback."""

    status: JobStatus = Field(..., description="Current job status")
    phase: JobPhase | None = Field(default=None, description="Current execution phase")
    progress: float | None = Field(
        default=None, description="Progress percentage (0.0 to 1.0)"
    )
    message: str | None = Field(default=None, description="Status message")
    current_step: str | None = Field(
        default=None, description="Current step description"
    )
    total_steps: int | None = Field(default=None, description="Total number of steps")
    completed_steps: int | None = Field(
        default=None, description="Number of completed steps"
    )
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    error_details: dict[str, Any] | None = Field(
        default=None, description="Detailed error information"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Update timestamp"
    )


class OCIArtifactSpec(BaseModel):
    """Specification for OCI artifact creation."""

    # Source files
    files: list[Path] = Field(..., description="Paths to files to include in artifact")
    base_path: Path | None = Field(
        default=None, description="Base path for relative file paths"
    )

    # Artifact metadata
    title: str | None = Field(default=None, description="Artifact title")
    description: str | None = Field(default=None, description="Artifact description")
    annotations: dict[str, str] = Field(
        default_factory=dict, description="Custom annotations"
    )

    # Job context
    job_id: str = Field(..., description="Job ID for tracking")
    benchmark_id: str = Field(..., description="Benchmark ID")
    model_name: str = Field(..., description="Model name")


class OCIArtifactResult(BaseModel):
    """Result of OCI artifact creation."""

    digest: str = Field(..., description="Artifact digest (SHA256)")
    reference: str = Field(..., description="Full OCI reference with digest")
    size_bytes: int = Field(..., description="Artifact size in bytes")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )


class JobResults(BaseModel):
    """Results returned by run_benchmark_job.

    This is returned synchronously when the job completes.
    """

    # Core results
    job_id: str = Field(..., description="Job identifier")
    benchmark_id: str = Field(..., description="Benchmark that was evaluated")
    model_name: str = Field(..., description="Model that was evaluated")
    results: list[EvaluationResult] = Field(..., description="Evaluation results")

    # Summary statistics
    overall_score: float | None = Field(
        default=None, description="Overall score if applicable"
    )
    num_examples_evaluated: int = Field(
        ..., description="Number of examples actually evaluated"
    )

    # Execution metadata
    duration_seconds: float = Field(..., description="Total evaluation time")
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Completion timestamp"
    )
    evaluation_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Framework-specific metadata"
    )

    # Artifact information (if persisted)
    oci_artifact: OCIArtifactResult | None = Field(
        default=None, description="OCI artifact info if persisted"
    )


class JobCallbacks(ABC):
    """Abstract interface for job callbacks.

    Implementations of this interface communicate with the localhost sidecar
    to report status and persist artifacts back to the service.
    """

    @abstractmethod
    def report_status(self, update: JobStatusUpdate) -> None:
        """Report job status update to the service.

        This sends a status update to the localhost sidecar, which forwards
        it to the eval-hub service to update the job record.

        Args:
            update: Status update to report

        Raises:
            RuntimeError: If status update fails
        """
        pass

    @abstractmethod
    def create_oci_artifact(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
        """Create and push OCI artifact via sidecar.

        This requests the localhost sidecar to create an OCI artifact from
        the specified files and push it to the configured registry.

        Args:
            spec: Specification for the artifact to create

        Returns:
            OCIArtifactResult: Information about the created artifact

        Raises:
            RuntimeError: If artifact creation or push fails
        """
        pass

    @abstractmethod
    def report_results(self, results: JobResults) -> None:
        """Report final evaluation results to the service.

        This sends the complete evaluation results to the localhost sidecar,
        which forwards them to the eval-hub service to update the job record
        with final outcomes.

        Args:
            results: Final job results to report

        Raises:
            RuntimeError: If results reporting fails
        """
        pass
