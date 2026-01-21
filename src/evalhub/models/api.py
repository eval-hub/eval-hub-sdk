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


class ModelConfig(BaseModel):
    """Configuration for the model being evaluated."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Model name or identifier")
    provider: str | None = Field(
        default=None, description="Model provider (e.g., 'vllm', 'transformers')"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific parameters (temperature, max_tokens, etc.)",
    )
    device: str | None = Field(default=None, description="Device specification")
    batch_size: int | None = Field(
        default=None, description="Batch size for evaluation"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v

    def merge_with_defaults(self, defaults: dict[str, Any]) -> "ModelConfig":
        """Merge configuration with default values."""
        merged_params = {**defaults, **self.parameters}
        return self.model_copy(update={"parameters": merged_params})


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
    default_few_shot: int | None = Field(
        default=None, description="Default number of few-shot examples"
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
    num_few_shot: int | None = Field(
        default=None, description="Number of few-shot examples"
    )
    random_seed: int | None = Field(
        default=42, description="Random seed for reproducibility"
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


class EvaluationJob(BaseModel):
    """Evaluation job information."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    evaluation_status: EvaluationStatus | None = Field(
        default=None, description="Detailed evaluation status"
    )

    # Request information
    request: EvaluationRequest = Field(..., description="Original evaluation request")

    # Timing information
    submitted_at: datetime = Field(..., description="When the job was submitted")
    started_at: datetime | None = Field(
        default=None, description="When evaluation started"
    )
    completed_at: datetime | None = Field(
        default=None, description="When evaluation completed"
    )

    # Progress information
    progress: float | None = Field(
        default=None, description="Progress percentage (0.0 to 1.0)"
    )
    current_step: str | None = Field(
        default=None, description="Current step description"
    )
    total_steps: int | None = Field(default=None, description="Total number of steps")
    completed_steps: int | None = Field(
        default=None, description="Number of completed steps"
    )

    # Error information
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    error_details: dict[str, Any] | None = Field(
        default=None, description="Detailed error information"
    )

    # Resource usage
    estimated_duration: int | None = Field(
        default=None, description="Estimated duration in seconds"
    )
    actual_duration: int | None = Field(
        default=None, description="Actual duration in seconds"
    )


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

    job_id: str = Field(..., description="Job identifier")
    path: str | None = Field(
        default=None,
        description="Directory path containing files to persist. None if no files to persist.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Framework-specific metadata (e.g., OCI annotations)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "job_123",
                "path": "/tmp/lighteval_output/job_123",
                "metadata": {
                    "framework": "lighteval",
                    "benchmark_id": "benchmark_id_value",
                },
            }
        }
    )


class PersistResponse(BaseModel):
    """Response from OCI artifact persistence operation."""

    job_id: str = Field(..., description="Job identifier")
    oci_ref: str = Field(..., description="Full OCI reference including digest")
    digest: str = Field(..., description="SHA256 digest of artifact")
    files_count: int = Field(..., description="Number of files persisted")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional persistence metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "job_123",
                "oci_ref": "ghcr.io/org/repo:latest@sha256:abc123...",
                "digest": "sha256:abc123...",
                "files_count": 42,
                "metadata": {"placeholder": True},
            }
        }
    )


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
    error_message: str | None = Field(
        default=None, description="Error message when status is unhealthy"
    )

    # Additional info
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional health metadata"
    )
