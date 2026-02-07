"""EvalHub SDK Models - Standard request/response models for framework adapters."""

from .api import (
    Benchmark,
    BenchmarkInfo,
    BenchmarkReference,
    BenchmarksList,
    Collection,
    CollectionList,
    ErrorInfo,
    ErrorResponse,
    EvaluationJob,
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
    "JobsList",
    "EvaluationResponse",
    # Provider & Benchmark models
    "Provider",
    "ProviderList",
    "Benchmark",
    "BenchmarkInfo",
    "BenchmarksList",
    "BenchmarkReference",
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
