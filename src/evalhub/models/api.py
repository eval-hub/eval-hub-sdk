"""Core API models for the EvalHub SDK common interface."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobStatus(str, Enum):
    """Standard job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationStatus(str, Enum):
    """Evaluation-specific status values."""

    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorInfo(BaseModel):
    """Error information with message and code.

    Matches the MessageInfo structure from eval-hub API.
    """

    message: str = Field(..., description="Error message")
    message_code: str = Field(..., description="Error code identifier")


class ModelConfig(BaseModel):
    """Configuration for the model being evaluated.

    This matches the eval-hub API's ModelRef schema:
    - url: The model endpoint URL (e.g., vLLM, OpenAI-compatible endpoint)
    - name: Model name/identifier
    """

    url: str = Field(..., description="Model endpoint URL")
    name: str = Field(..., description="Model name or identifier")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model URL cannot be empty")
        return v


class BenchmarkInfo(BaseModel):
    """Information about an available benchmark."""

    benchmark_id: str = Field(..., description="Unique benchmark identifier")
    name: str = Field(..., description="Human-readable benchmark name")
    description: str | None = Field(default=None, description="Benchmark description")
    category: str | None = Field(default=None, description="Benchmark category")
    tags: list[str] = Field(default_factory=list, description="Benchmark tags")
    metrics: list[str] = Field(default_factory=list, description="Available metrics")
    dataset_size: int | None = Field(
        default=None, description="Number of examples in dataset"
    )
    supports_few_shot: bool = Field(
        default=True, description="Whether benchmark supports few-shot evaluation"
    )
    num_few_shot: int | None = Field(
        default=None,
        description="Default number of few-shot examples",
    )
    custom_config_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for custom benchmark configuration"
    )

    @field_validator("benchmark_id", "name")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("String fields cannot be empty")
        return v


class EvaluationRequest(BaseModel):
    """Request to run an evaluation."""

    benchmark_id: str = Field(..., description="Benchmark to evaluate on")
    model: ModelConfig = Field(..., description="Model configuration")

    # Evaluation parameters
    num_examples: int | None = Field(
        default=None, description="Number of examples to evaluate (None = all)"
    )

    # Custom benchmark configuration
    benchmark_config: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark-specific configuration"
    )

    # Job metadata
    experiment_name: str | None = Field(
        default=None, description="Name for this evaluation experiment"
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="Custom tags for the job"
    )
    priority: int = Field(
        default=0, description="Job priority (higher = more priority)"
    )


class EvaluationResult(BaseModel):
    """Individual evaluation result."""

    metric_name: str = Field(..., description="Name of the metric")
    metric_value: float | int | str | bool = Field(..., description="Metric value")
    metric_type: str = Field(
        default="float", description="Type of metric (float, int, accuracy, etc.)"
    )
    confidence_interval: tuple[float, float] | None = Field(
        default=None, description="95% confidence interval if available"
    )

    # Additional metadata
    num_samples: int | None = Field(
        default=None, description="Number of samples used for this metric"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metric-specific metadata"
    )


class MessageInfo(BaseModel):
    """Message information with code."""

    message: str = Field(..., description="Message text")
    message_code: str = Field(..., description="Message code")


class EvaluationJobResource(BaseModel):
    """Resource information for an evaluation job."""

    id: str = Field(..., description="Unique job identifier")
    tenant: str = Field(..., description="Tenant identifier")
    created_at: datetime = Field(..., description="When the job was created")
    updated_at: datetime = Field(..., description="When the job was last updated")
    mlflow_experiment_id: str | None = Field(
        default=None, description="MLFlow experiment ID"
    )
    message: MessageInfo | None = Field(default=None, description="Status message")


class EvaluationJobState(BaseModel):
    """State information for an evaluation job."""

    state: str = Field(..., description="Job state")
    message: MessageInfo | None = Field(default=None, description="State message")


class BenchmarkStatus(BaseModel):
    """Status of a benchmark in an evaluation job."""

    id: str = Field(..., description="Benchmark identifier")
    provider_id: str = Field(..., description="Provider identifier")
    status: JobStatus = Field(..., description="Benchmark status")
    message: MessageInfo | None = Field(default=None, description="Status message")

    # Convenience property for consistency with job state
    @property
    def state(self) -> JobStatus:
        """Get benchmark state (alias for status)."""
        return self.status


class EvaluationJobStatus(BaseModel):
    """Status information for an evaluation job."""

    state: JobStatus = Field(..., description="Job state")
    message: MessageInfo | None = Field(default=None, description="Status message")
    benchmarks: list[BenchmarkStatus] = Field(
        default_factory=list, description="Benchmark statuses"
    )


class BenchmarkResult(BaseModel):
    """Results from a single benchmark evaluation."""

    id: str = Field(..., description="Benchmark identifier")
    provider_id: str = Field(..., description="Provider identifier")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark metrics"
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark artifacts"
    )
    mlflow_run_id: str | None = Field(
        default=None, description="MLFlow run ID if tracking enabled"
    )
    logs_path: str | None = Field(default=None, description="Path to evaluation logs")


class EvaluationJobResults(BaseModel):
    """Results from an evaluation job."""

    total_evaluations: int = Field(..., description="Total number of evaluations")
    completed_evaluations: int = Field(
        default=0, description="Number of completed evaluations"
    )
    failed_evaluations: int = Field(
        default=0, description="Number of failed evaluations"
    )
    benchmarks: list[BenchmarkResult] = Field(
        default_factory=list, description="Benchmark results"
    )
    mlflow_experiment_url: str | None = Field(
        default=None, description="MLFlow experiment URL if tracking enabled"
    )


class BenchmarkConfig(BaseModel):
    """Benchmark configuration for job submission."""

    id: str = Field(..., description="Benchmark identifier")
    provider_id: str = Field(..., description="Provider identifier")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark-specific parameters"
    )


class JobSubmissionRequest(BaseModel):
    """Request to submit an evaluation job."""

    model: ModelConfig = Field(..., description="Model configuration")
    benchmarks: list[BenchmarkConfig] = Field(
        ..., description="List of benchmarks to evaluate", min_length=1
    )
    timeout_minutes: int | None = Field(
        default=None, description="Job timeout in minutes"
    )
    retry_attempts: int | None = Field(
        default=None, description="Number of retry attempts on failure"
    )


class EvaluationJob(BaseModel):
    """Evaluation job information from the API.

    Matches EvaluationJobResource from the Go API.
    """

    resource: EvaluationJobResource = Field(..., description="Resource metadata")
    status: EvaluationJobStatus | None = Field(
        default=None, description="Job status information"
    )
    results: EvaluationJobResults | None = Field(
        default=None, description="Job results"
    )

    # Embedded EvaluationJobConfig fields
    model: ModelConfig = Field(..., description="Model configuration")
    benchmarks: list[BenchmarkConfig] = Field(
        ..., description="Benchmark configurations"
    )
    timeout_minutes: int | None = Field(
        default=None, description="Job timeout in minutes"
    )
    retry_attempts: int | None = Field(
        default=None, description="Number of retry attempts"
    )

    # Convenience properties to access nested fields
    @property
    def id(self) -> str:
        """Get job ID from resource."""
        return self.resource.id

    @property
    def state(self) -> JobStatus:
        """Get job state."""
        return self.status.state if self.status else JobStatus.PENDING


class JobsList(BaseModel):
    """List of evaluation jobs response."""

    total_count: int = Field(..., description="Total number of jobs")
    items: list[EvaluationJob] = Field(
        default_factory=list, description="List of evaluation jobs"
    )

    @field_validator("items", mode="before")
    @classmethod
    def handle_none_items(cls, v: list[EvaluationJob] | None) -> list[EvaluationJob]:
        """Convert None to empty list for compatibility with server responses."""
        return v if v is not None else []


class EvaluationResponse(BaseModel):
    """Response containing evaluation results."""

    job_id: str = Field(..., description="Job identifier")
    benchmark_id: str = Field(..., description="Benchmark that was evaluated")
    model_name: str = Field(..., description="Model that was evaluated")

    # Results
    results: list[EvaluationResult] = Field(..., description="Evaluation results")

    # Summary statistics
    overall_score: float | None = Field(
        default=None, description="Overall score if applicable"
    )
    num_examples_evaluated: int = Field(
        ..., description="Number of examples actually evaluated"
    )

    # Metadata
    evaluation_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Framework-specific evaluation metadata"
    )
    completed_at: datetime = Field(..., description="When evaluation was completed")
    duration_seconds: float = Field(..., description="Total evaluation time")


class OCICoordinate(BaseModel):
    """OCI artifact coordinates for persistence."""

    oci_ref: str = Field(
        ..., description="OCI reference (e.g., 'ghcr.io/org/repo:tag')"
    )
    oci_subject: str | None = Field(
        default=None, description="Optional OCI subject identifier"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "oci_ref": "ghcr.io/my-org/eval-results:latest",
                "oci_subject": "not used atm",
            }
        }
    )


class EvaluationJobFilesLocation(BaseModel):
    """Files location for persisting as OCI artifacts for an evaluation job."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "job_123",
                "path": "/tmp/lighteval_output/job_123",
                "metadata": {
                    "framework": "lighteval",
                    "benchmark_id": "benchmark_id_value",
                },
            }
        },
    )

    id: str = Field(..., description="Job identifier")
    path: str | None = Field(
        default=None,
        description="Directory path containing files to persist. None if no files to persist.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Framework-specific metadata (e.g., OCI annotations)",
    )


class PersistResponse(BaseModel):
    """Response from OCI artifact persistence operation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "job_123",
                "oci_ref": "ghcr.io/org/repo:latest@sha256:abc123...",
                "digest": "sha256:abc123...",
                "files_count": 42,
                "metadata": {"placeholder": True},
            }
        },
    )

    id: str = Field(..., description="Job identifier")
    oci_ref: str = Field(..., description="Full OCI reference including digest")
    digest: str = Field(..., description="SHA256 digest of artifact")
    files_count: int = Field(..., description="Number of files persisted")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional persistence metadata"
    )


class Provider(BaseModel):
    """Provider information from EvalHub API.

    Matches the Go ProviderResource structure from pkg/api/providers.go
    """

    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Provider display name")
    description: str = Field(..., description="Provider description")
    type: str = Field(..., description="Provider type")
    benchmarks: list["Benchmark"] = Field(
        default_factory=list, description="Benchmarks supported by this provider"
    )


class ProviderList(BaseModel):
    """List of providers response."""

    total_count: int = Field(..., description="Total number of providers")
    items: list[Provider] = Field(default_factory=list, description="List of providers")

    @field_validator("items", mode="before")
    @classmethod
    def handle_none_items(cls, v: list[Provider] | None) -> list[Provider]:
        """Convert None to empty list for compatibility with server responses."""
        return v if v is not None else []


class Benchmark(BaseModel):
    """Benchmark information from EvalHub API."""

    id: str = Field(..., description="Unique benchmark identifier")
    provider_id: str | None = Field(
        None, description="Provider that owns this benchmark"
    )
    name: str = Field(..., description="Human-readable benchmark name")
    description: str = Field(..., description="Benchmark description")
    category: str = Field(..., description="Benchmark category")
    metrics: list[str] = Field(default_factory=list, description="List of metrics")
    num_few_shot: int | None = Field(None, description="Number of few-shot examples")
    dataset_size: int | None = Field(None, description="Size of the evaluation dataset")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class BenchmarksList(BaseModel):
    """List of benchmarks response."""

    total_count: int = Field(..., description="Total number of benchmarks")
    items: list[Benchmark] = Field(..., description="List of benchmarks")


class Resource(BaseModel):
    """Resource metadata."""

    id: str = Field(..., description="Resource identifier")
    tenant: str = Field(..., description="Tenant identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class BenchmarkReference(BaseModel):
    """Reference to a benchmark within a collection."""

    provider_id: str = Field(..., description="Provider identifier")
    benchmark_id: str = Field(..., description="Benchmark identifier")
    weight: float = Field(default=1.0, description="Benchmark weight in collection")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark configuration"
    )


class Collection(BaseModel):
    """Collection of benchmarks from EvalHub API."""

    resource: Resource = Field(..., description="Resource metadata")
    name: str = Field(..., description="Collection name")
    description: str = Field(..., description="Collection description")
    tags: list[str] = Field(default_factory=list, description="Collection tags")
    custom: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    benchmarks: list[BenchmarkReference] = Field(
        default_factory=list, description="Collection benchmarks"
    )


class CollectionList(BaseModel):
    """List of collections response."""

    total_count: int = Field(..., description="Total number of collections")
    items: list[Collection] = Field(..., description="Collection resources")
    # Pagination fields
    first: dict[str, str] | None = Field(None, description="Link to first page")
    next: dict[str, str] | None = Field(None, description="Link to next page")
    limit: int | None = Field(None, description="Page size limit")


class FrameworkInfo(BaseModel):
    """Information about a framework adapter."""

    framework_id: str = Field(..., description="Unique framework identifier")
    name: str = Field(..., description="Framework display name")
    version: str = Field(..., description="Framework version")
    description: str | None = Field(default=None, description="Framework description")

    # Capabilities
    supported_benchmarks: list[BenchmarkInfo] = Field(
        default_factory=list, description="Benchmarks supported by this framework"
    )
    supported_model_types: list[str] = Field(
        default_factory=list,
        description="Model types supported (e.g., 'transformers', 'vllm')",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Framework capabilities (e.g., 'text-generation', 'classification')",
    )

    # Configuration schema
    default_model_config: dict[str, Any] = Field(
        default_factory=dict, description="Default model configuration"
    )
    config_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for framework configuration"
    )

    # Metadata
    author: str | None = Field(default=None, description="Framework adapter author")
    contact: str | None = Field(default=None, description="Contact information")
    documentation_url: str | None = Field(default=None, description="Documentation URL")
    repository_url: str | None = Field(
        default=None, description="Source repository URL"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Type of error")
    message: str = Field(..., description="Human-readable error message")
    error_code: str | None = Field(
        default=None, description="Framework-specific error code"
    )
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When error occurred"
    )
    request_id: str | None = Field(default=None, description="Request ID for debugging")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(
        ..., description="Health status ('healthy', 'unhealthy', 'degraded')"
    )
    framework_id: str = Field(..., description="Framework identifier")
    version: str = Field(..., description="Framework adapter version")

    # Dependency status
    dependencies: dict[str, dict[str, Any]] | None = Field(
        default=None, description="Status of framework dependencies"
    )

    # Resource information
    memory_usage: dict[str, Any] | None = Field(
        default=None, description="Memory usage information"
    )
    gpu_usage: dict[str, Any] | None = Field(
        default=None, description="GPU usage information"
    )

    # Timing
    uptime_seconds: float | None = Field(
        default=None, description="Adapter uptime in seconds"
    )
    last_evaluation_time: datetime | None = Field(
        default=None, description="Time of last evaluation"
    )

    # Error information for unhealthy status
    error: ErrorInfo | None = Field(
        default=None, description="Error information when status is unhealthy"
    )

    # Additional info
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional health metadata"
    )
