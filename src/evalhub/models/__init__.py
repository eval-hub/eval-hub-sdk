"""EvalHub SDK Models - Standard request/response models for framework adapters."""

from .api import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkInfo,
    BenchmarkReference,
    BenchmarkResult,
    BenchmarksList,
    BenchmarkStatus,
    Collection,
    CollectionList,
    ErrorInfo,
    ErrorResponse,
    EvaluationJob,
    EvaluationJobResource,
    EvaluationJobResults,
    EvaluationJobStatus,
    # Core API models
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    EvaluationStatus,
    FrameworkInfo,
    HealthResponse,
    # Status and metadata
    JobsList,
    JobStatus,
    JobSubmissionRequest,
    ModelConfig,
    Provider,
    ProviderList,
    Resource,
)

__all__ = [
    # Job & Evaluation models
    "JobStatus",
    "EvaluationStatus",
    "ModelConfig",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluationJob",
    "EvaluationJobResource",
    "EvaluationJobResults",
    "EvaluationJobStatus",
    "JobsList",
    "JobSubmissionRequest",
    "EvaluationResponse",
    # Provider & Benchmark models
    "Provider",
    "ProviderList",
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkInfo",
    "BenchmarkResult",
    "BenchmarksList",
    "BenchmarkReference",
    "BenchmarkStatus",
    # Collection models
    "Resource",
    "Collection",
    "CollectionList",
    # Framework models
    "FrameworkInfo",
    # Response models
    "ErrorInfo",
    "ErrorResponse",
    "HealthResponse",
]
