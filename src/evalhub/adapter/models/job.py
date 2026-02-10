"""Simplified adapter models for benchmark job execution."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from ...models.api import EvaluationResult, JobStatus, ModelConfig


class MessageInfo(BaseModel):
    """Message information with message and code.

    Matches the MessageInfo structure from eval-hub API.
    """

    message: str = Field(..., description="Message text")
    message_code: str = Field(..., description="Message code identifier")


class ErrorInfo(BaseModel):
    """Error information with message and code.

    Matches the MessageInfo structure from eval-hub API.
    """

    message: str = Field(..., description="Error message")
    message_code: str = Field(..., description="Error code identifier")


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

    Matches the Go service's EvaluationJobConfig structure.

    Mandatory fields:
        - job_id: Unique job identifier
        - benchmark_id: Benchmark to evaluate
        - model: Model configuration (url and name)
        - benchmark_config: Benchmark-specific parameters
        - callback_url: URL for status and result callbacks

    Optional fields:
        - num_examples: Number of examples to evaluate (None = all)
        - experiment_name: Name for this evaluation experiment
        - tags: Custom tags for the job
        - timeout_seconds: Maximum job execution time (default: 3600)
        - retry_attempts: Number of retry attempts on failure
    """

    # ============================================================================
    # MANDATORY FIELDS
    # ============================================================================

    # Job identification (mandatory)
    id: str = Field(..., description="Unique job identifier from service")
    benchmark_id: str = Field(..., description="Benchmark to evaluate")

    # Model configuration (mandatory)
    model: ModelConfig = Field(..., description="Model configuration")

    # Benchmark-specific configuration (mandatory)
    # adapter-specific params go here
    benchmark_config: dict[str, Any] = Field(
        ..., description="Benchmark-specific parameters"
    )

    # Callback configuration (mandatory)
    callback_url: str = Field(
        ...,
        description="Base URL for callbacks",
    )

    # ============================================================================
    # OPTIONAL FIELDS
    # ============================================================================

    # Evaluation parameters (optional)
    num_examples: int | None = Field(
        default=None, description="Number of examples to evaluate (None = all)"
    )

    # Job metadata (optional)
    experiment_name: str | None = Field(
        default=None, description="Name for this evaluation experiment"
    )
    tags: list[dict[str, str]] = Field(
        default_factory=list, description="Custom tags for the job"
    )

    # Resource hints (optional)
    timeout_seconds: int | None = Field(
        default=3600, description="Maximum job execution time"
    )
    retry_attempts: int | None = Field(
        default=None, description="Number of retry attempts on failure"
    )

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Load a JobSpec from a JSON file.

        Args:
            path: Path to the JSON file containing the job specification.

        Returns:
            JobSpec: Parsed job specification.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON is invalid or doesn't match the schema.

        Example:
            ```python
            # Load from explicit path
            spec = JobSpec.from_file("/meta/job.json")

            # Or use settings to get the path
            spec = JobSpec.from_file(settings.resolved_job_spec_path)
            ```
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Job spec file not found: {file_path}")

        try:
            with open(file_path) as f:
                data = json.load(f)
            return cls(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in job spec file: {e}") from e


class JobStatusUpdate(BaseModel):
    """Status update sent to service via callback."""

    status: JobStatus = Field(..., description="Current job status")
    phase: JobPhase | None = Field(default=None, description="Current execution phase")
    progress: float | None = Field(
        default=None, description="Progress percentage (0.0 to 1.0)"
    )
    message: MessageInfo = Field(
        default_factory=lambda: MessageInfo(
            message="Status update",
            message_code="status_update",
        ),
        description="Status message",
    )
    current_step: str | None = Field(
        default=None, description="Current step description"
    )
    total_steps: int | None = Field(default=None, description="Total number of steps")
    completed_steps: int | None = Field(
        default=None, description="Number of completed steps"
    )
    error: ErrorInfo | None = Field(
        default=None, description="Error information if failed"
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
    id: str = Field(..., description="Job ID for tracking")
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
    id: str = Field(..., description="Job identifier")
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
