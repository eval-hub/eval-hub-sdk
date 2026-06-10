"""Core API models for the EvalHub SDK common interface."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OCI_ARTIFACT_TYPE = "application/vnd.eval-hub.github.io"

OCI_ANNOTATION_JOB_ID = "io.github.eval-hub.job_id"
OCI_ANNOTATION_BENCHMARK_ID = "io.github.eval-hub.benchmark_id"
OCI_ANNOTATION_PROVIDER_ID = "io.github.eval-hub.provider_id"


class JobStatus(str, Enum):
    """Standard job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_FAILED = "partially_failed"


class EvaluationStatus(str, Enum):
    """Evaluation-specific status values."""

    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPhase(str, Enum):
    """Job execution phases.

    Represents the granular execution phase within a running benchmark job.
    Sent by adapters via status events and surfaced in server responses.
    """

    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    RUNNING_EVALUATION = "running_evaluation"
    POST_PROCESSING = "post_processing"
    PERSISTING_ARTIFACTS = "persisting_artifacts"
    COMPLETED = "completed"


class ErrorInfo(BaseModel):
    """Error information with message and code.

    Matches the MessageInfo structure from eval-hub API.
    """

    message: str = Field(..., description="Error message")
    message_code: str = Field(..., description="Error code identifier")


class ModelAuth(BaseModel):
    """Authentication configuration for the model endpoint."""

    secret_ref: str = Field(
        ..., description="Kubernetes Secret name containing model credentials"
    )

    @field_validator("secret_ref")
    @classmethod
    def validate_secret_ref(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("secret_ref cannot be empty")
        return cleaned


class ModelConfig(BaseModel):
    """Configuration for the model being evaluated.

    This matches the eval-hub API's ModelRef schema:
    - url: The model endpoint URL (e.g., vLLM, OpenAI-compatible endpoint)
    - name: Model name/identifier
    - auth: Optional model authentication (secret_ref)
    """

    url: str = Field(..., description="Model endpoint URL")
    name: str = Field(..., description="Model name or identifier")
    auth: ModelAuth | None = Field(
        default=None, description="Authentication configuration for the model endpoint"
    )

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
    tenant: str | None = Field(default=None, description="Tenant identifier")
    created_at: datetime = Field(..., description="When the job was created")
    updated_at: datetime | None = Field(
        default=None, description="When the job was last updated"
    )
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
    benchmark_index: int | None = Field(default=None, description="Benchmark index")
    status: JobStatus = Field(..., description="Benchmark status")
    phase: JobPhase | None = Field(
        default=None, description="Current execution phase of the benchmark job"
    )
    error_message: MessageInfo | None = Field(default=None, description="Error message")
    started_at: datetime | None = Field(
        default=None, description="When benchmark started"
    )
    completed_at: datetime | None = Field(
        default=None, description="When benchmark completed"
    )

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
    benchmark_index: int | None = Field(default=None, description="Benchmark index")
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

    benchmarks: list[BenchmarkResult] = Field(
        default_factory=list, description="Benchmark results"
    )
    mlflow_experiment_url: str | None = Field(
        default=None, description="MLFlow experiment URL if tracking enabled"
    )


class S3TestDataRef(BaseModel):
    """S3 source for custom test data."""

    bucket: str = Field(..., description="S3 bucket name")
    key: str = Field(..., description="S3 object key or prefix")
    secret_ref: str = Field(
        ..., description="Kubernetes Secret name containing S3 credentials"
    )


class TestDataRef(BaseModel):
    """Reference to an external test data source."""

    s3: S3TestDataRef | None = Field(default=None, description="S3 data source")


class BenchmarkConfig(BaseModel):
    """Benchmark configuration for job submission."""

    id: str = Field(..., description="Benchmark identifier")
    provider_id: str = Field(..., description="Provider identifier")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark-specific parameters"
    )
    test_data_ref: TestDataRef | None = Field(
        default=None, description="Reference to custom external test data"
    )


class CollectionRef(BaseModel):
    """Reference to a collection for job submission."""

    id: str = Field(..., description="The unique identifier of the collection")
    benchmarks: list[BenchmarkConfig] | None = Field(
        default=None,
        description="Optional subset of benchmarks from the collection",
    )


class ExperimentTag(BaseModel):
    """Tag on an experiment (e.g. MLFlow).

    Matches the ExperimentTag schema from the eval-hub API.
    """

    key: str = Field(..., description="Tag key", max_length=250)
    value: str = Field(..., description="Tag value", max_length=5000)


class ExperimentConfig(BaseModel):
    """Configuration for MLFlow experiment tracking.

    When provided on a job submission, the evaluation job will be
    tracked in MLFlow.

    Matches the ExperimentConfig schema from the eval-hub API.
    """

    name: str | None = Field(default=None, description="Experiment name")
    tags: list[ExperimentTag] = Field(
        default_factory=list, description="Experiment tags", max_length=20
    )
    artifact_location: str | None = Field(
        default=None, description="Artifact storage location"
    )


class OCICoordinates(BaseModel):
    """OCI artifact coordinates for persistence."""

    oci_host: str = Field(..., description="OCI registry host (e.g., 'quay.io')")
    oci_repository: str = Field(
        ..., description="OCI repository (e.g., 'my-org/my-repo')"
    )
    oci_tag: str | None = Field(default=None, description="OCI tag (e.g., 'eval-123')")
    oci_subject: str | None = Field(
        default=None,
        description="Optional OCI subject identifier (in same registry and repo)",
    )
    annotations: dict[str, str] = Field(
        default_factory=dict, description="Custom annotations"
    )


class QueueConfig(BaseModel):
    """Scheduling queue configuration for Kubernetes runtimes (e.g. Kueue LocalQueue)."""

    kind: str | None = Field(
        default=None,
        description="Queue integration type. Only 'kueue' is supported. Defaults to 'kueue' server-side when omitted.",
    )
    name: str = Field(
        ..., description="Queue resource name (for Kueue, the LocalQueue name)."
    )


class OCIConnectionConfig(BaseModel):
    """K8s connection configuration for OCI registry authentication."""

    connection: str = Field(
        ...,
        description="Name of a K8s Secret (type kubernetes.io/dockerconfigjson) for OCI registry auth",
    )


class EvaluationExportsOCI(BaseModel):
    """OCI export configuration for an evaluation job."""

    coordinates: OCICoordinates = Field(..., description="OCI artifact coordinates")
    k8s: OCIConnectionConfig | None = Field(
        default=None, description="K8s connection for OCI registry auth"
    )


class EvaluationExports(BaseModel):
    """Optional exports configuration for an evaluation job."""

    oci: EvaluationExportsOCI | None = Field(
        default=None, description="OCI export configuration"
    )


class JobSubmissionRequest(BaseModel):
    """Request to submit an evaluation job.

    Either ``benchmarks`` or ``collection`` must be provided, but not both.
    """

    name: str = Field(..., description="Name for the evaluation job")
    description: str | None = Field(
        default=None, description="The evaluation job description"
    )
    tags: list[str] = Field(default_factory=list, description="The evaluation job tags")
    model: ModelConfig = Field(..., description="Model configuration")
    benchmarks: list[BenchmarkConfig] | None = Field(
        default=None, description="List of benchmarks to evaluate"
    )
    collection: CollectionRef | None = Field(
        default=None, description="Collection reference for the evaluation job"
    )
    experiment: ExperimentConfig | None = Field(
        default=None,
        description="MLFlow experiment configuration. When provided, the evaluation job will be tracked in MLFlow.",
    )
    exports: EvaluationExports | None = Field(
        default=None,
        description="Optional exports configuration (e.g., OCI artifact persistence)",
    )
    queue: QueueConfig | None = Field(
        default=None,
        description="Optional scheduling queue for Kubernetes-backed evaluation jobs (e.g. Kueue).",
    )

    @model_validator(mode="after")
    def check_benchmarks_or_collection(self) -> "JobSubmissionRequest":
        if self.benchmarks and self.collection:
            raise ValueError("Cannot specify both 'benchmarks' and 'collection'")
        if not self.benchmarks and not self.collection:
            raise ValueError("Must specify either 'benchmarks' or 'collection'")
        return self


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
    name: str = Field(..., description="The evaluation job name")
    description: str | None = Field(
        default=None, description="The evaluation job description"
    )
    tags: list[str] = Field(default_factory=list, description="The evaluation job tags")
    model: ModelConfig = Field(..., description="Model configuration")
    benchmarks: list[BenchmarkConfig] | None = Field(
        default=None, description="Benchmark configurations"
    )
    collection: CollectionRef | None = Field(
        default=None, description="Collection reference for the evaluation job"
    )
    experiment: ExperimentConfig | None = Field(
        default=None,
        description="MLFlow experiment configuration",
    )
    exports: EvaluationExports | None = Field(
        default=None,
        description="Optional exports configuration",
    )
    queue: QueueConfig | None = Field(
        default=None,
        description="Optional scheduling queue for Kubernetes-backed evaluation jobs (e.g. Kueue).",
    )

    # Convenience properties to access nested fields
    @property
    def id(self) -> str:
        """Get job ID from resource."""
        return self.resource.id

    _TERMINAL_STATES = {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.PARTIALLY_FAILED,
    }

    _BENCHMARK_TERMINAL_STATES = {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }

    @property
    def state(self) -> JobStatus:
        """Get job state."""
        return self.status.state if self.status else JobStatus.PENDING

    @property
    def effective_state(self) -> JobStatus:
        """Derive effective job state, considering benchmark-level status.

        When the server has not yet promoted individual benchmark completions
        to the top-level job state, this property infers the overall state
        from the benchmark statuses.  If every benchmark has reached a
        terminal state the job is considered completed, partially failed
        (mix of completed and failed/cancelled), or fully failed.
        """
        top = self.state
        if top in self._TERMINAL_STATES:
            return top

        if not self.status or not self.status.benchmarks:
            return top

        bench_states = [b.state for b in self.status.benchmarks]
        if all(s in self._BENCHMARK_TERMINAL_STATES for s in bench_states):
            has_failed = any(s == JobStatus.FAILED for s in bench_states)
            has_cancelled = any(s == JobStatus.CANCELLED for s in bench_states)
            has_completed = any(s == JobStatus.COMPLETED for s in bench_states)
            all_failed = all(s == JobStatus.FAILED for s in bench_states)
            all_cancelled = all(s == JobStatus.CANCELLED for s in bench_states)

            if all_failed:
                return JobStatus.FAILED
            if all_cancelled:
                return JobStatus.CANCELLED
            if (has_failed or has_cancelled) and has_completed:
                return JobStatus.PARTIALLY_FAILED
            if has_failed:
                return JobStatus.FAILED
            if has_cancelled:
                return JobStatus.CANCELLED
            return JobStatus.COMPLETED

        return top


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


class Resource(BaseModel):
    """Resource metadata."""

    id: str = Field(..., description="Resource identifier")
    tenant: str | None = Field(default=None, description="Tenant identifier")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    updated_at: datetime | None = Field(
        default=None, description="Last update timestamp"
    )
    owner: str | None = Field(default=None, description="Resource owner")


class PrimaryScore(BaseModel):
    """Primary score configuration for a benchmark."""

    metric: str = Field(..., description="Metric name for the primary score")
    lower_is_better: bool = Field(
        default=False, description="Whether lower scores are better"
    )


class PassCriteria(BaseModel):
    """Pass/fail criteria for a benchmark."""

    threshold: float = Field(..., description="Threshold value for passing")


class ScoreRange(BaseModel):
    """A score band with semantic meaning."""

    range: str = Field(..., description="Score range (e.g. '0-50')")
    meaning: str = Field(..., description="Human-readable meaning of this band")


class BenchmarkAgentMetadata(BaseModel):
    """Agent-consumption metadata at the benchmark level."""

    result_interpretation: str | None = Field(default=None)
    score_ranges: list[ScoreRange] = Field(default_factory=list)


class AgentMetadata(BaseModel):
    """Agent-consumption metadata at the provider level."""

    evaluates: list[str] = Field(default_factory=list)
    recommended_when: list[str] = Field(default_factory=list)
    target_type: str | None = Field(default=None)
    summary: str | None = Field(default=None)
    complements: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    result_interpretation: list[str] = Field(default_factory=list)


class Benchmark(BaseModel):
    """Benchmark information from EvalHub API."""

    id: str = Field(..., description="Unique benchmark identifier")
    url: str | None = Field(default=None, description="Benchmark URL")
    name: str = Field(..., description="Human-readable benchmark name")
    description: str | None = Field(default=None, description="Benchmark description")
    category: str = Field(..., description="Benchmark category")
    metrics: list[str] = Field(default_factory=list, description="List of metrics")
    num_few_shot: int = Field(default=0, description="Number of few-shot examples")
    dataset_size: int = Field(default=0, description="Size of the evaluation dataset")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    primary_score: PrimaryScore | None = Field(
        None, description="Primary score configuration"
    )
    pass_criteria: PassCriteria | None = Field(None, description="Pass/fail criteria")
    agent: BenchmarkAgentMetadata | None = Field(default=None)


class BenchmarksList(BaseModel):
    """List of benchmarks response."""

    total_count: int = Field(..., description="Total number of benchmarks")
    items: list[Benchmark] = Field(..., description="List of benchmarks")


class Provider(BaseModel):
    """Provider information from EvalHub API.

    Matches the Go ProviderResource structure from pkg/api/providers.go
    """

    resource: Resource = Field(..., description="Resource metadata")
    name: str = Field(..., description="Provider name")
    title: str = Field(default="", description="Provider display title")
    description: str | None = Field(default=None, description="Provider description")
    tags: list[str] = Field(default_factory=list, description="Provider tags")
    runtime: dict[str, Any] | None = Field(
        default=None, description="Runtime configuration"
    )
    benchmarks: list[Benchmark] = Field(
        default_factory=list, description="Benchmarks supported by this provider"
    )
    agent: AgentMetadata | None = Field(default=None)


class ProviderList(BaseModel):
    """List of providers response."""

    total_count: int = Field(..., description="Total number of providers")
    items: list[Provider] = Field(default_factory=list, description="List of providers")

    @field_validator("items", mode="before")
    @classmethod
    def handle_none_items(cls, v: list[Provider] | None) -> list[Provider]:
        """Convert None to empty list for compatibility with server responses."""
        return v if v is not None else []


class ProviderCreateRequest(BaseModel):
    """Request body for creating a new evaluation provider."""

    name: str = Field(..., description="Provider name")
    title: str = Field(default="", description="Provider display title")
    description: str | None = Field(default=None, description="Provider description")
    tags: list[str] = Field(default_factory=list, description="Provider tags")
    runtime: dict[str, Any] | None = Field(
        default=None, description="Runtime configuration"
    )
    benchmarks: list[Benchmark] = Field(
        default_factory=list, description="Benchmarks provided by this provider"
    )


class BenchmarkReference(BaseModel):
    """Reference to a benchmark within a collection."""

    id: str = Field(..., description="Benchmark identifier")
    provider_id: str = Field(..., description="Provider identifier")
    weight: float = Field(default=1.0, description="Benchmark weight in collection")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark-specific parameters"
    )
    primary_score: PrimaryScore | None = Field(
        default=None, description="Primary score configuration"
    )
    pass_criteria: PassCriteria | None = Field(
        default=None, description="Pass/fail criteria"
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_id(cls, data: Any) -> Any:
        """Accept 'benchmark_id' as an alias for 'id' for backwards compatibility."""
        if isinstance(data, dict) and "benchmark_id" in data and "id" not in data:
            data = dict(data)
            data["id"] = data.pop("benchmark_id")
        return data


class Collection(BaseModel):
    """Collection of benchmarks from EvalHub API."""

    resource: Resource = Field(..., description="Resource metadata")
    name: str = Field(..., description="Collection name")
    description: str = Field(..., description="Collection description")
    category: str = Field(..., description="Collection category")
    tags: list[str] = Field(default_factory=list, description="Collection tags")
    custom: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    benchmarks: list[BenchmarkReference] = Field(
        default_factory=list, description="Collection benchmarks"
    )
    pass_criteria: PassCriteria | None = Field(
        default=None, description="Pass/fail criteria"
    )


class CollectionList(BaseModel):
    """List of collections response."""

    total_count: int = Field(..., description="Total number of collections")
    items: list[Collection] = Field(
        default_factory=list, description="Collection resources"
    )

    @field_validator("items", mode="before")
    @classmethod
    def handle_none_items(cls, v: list[Collection] | None) -> list[Collection]:
        """Convert None to empty list for compatibility with server responses."""
        return v if v is not None else []

    # Pagination fields
    first: dict[str, str] | None = Field(None, description="Link to first page")
    next: dict[str, str] | None = Field(None, description="Link to next page")
    limit: int | None = Field(None, description="Page size limit")


class CollectionCreateRequest(BaseModel):
    """Request body for creating a new benchmark collection."""

    name: str = Field(..., description="Collection name")
    description: str = Field(default="", description="Collection description")
    category: str = Field(..., description="Collection category")
    tags: list[str] = Field(default_factory=list, description="Collection tags")
    benchmarks: list[BenchmarkReference] = Field(
        default_factory=list, description="Benchmarks to include in the collection"
    )
    pass_criteria: PassCriteria | None = Field(
        default=None, description="Pass/fail criteria for the collection"
    )
    custom: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


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
