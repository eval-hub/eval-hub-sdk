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
    CollectionRef,
    ErrorInfo,
    ErrorResponse,
    EvaluationJob,
    EvaluationJobResource,
    EvaluationJobResults,
    EvaluationJobStatus,
    # Core API models
    EvaluationResponse,
    EvaluationResult,
    EvaluationStatus,
    ExperimentConfig,
    ExperimentTag,
    FrameworkInfo,
    HealthResponse,
    # Status and metadata
    JobsList,
    JobStatus,
    JobSubmissionRequest,
    ModelConfig,
    PassCriteria,
    PrimaryScore,
    Provider,
    ProviderList,
    Resource,
)

__all__ = [
    # Job & Evaluation models
    "JobStatus",
    "EvaluationStatus",
    "ModelConfig",
    "EvaluationResult",
    "EvaluationJob",
    "EvaluationJobResource",
    "EvaluationJobResults",
    "EvaluationJobStatus",
    "JobsList",
    "JobSubmissionRequest",
    "EvaluationResponse",
    # Experiment models
    "ExperimentTag",
    "ExperimentConfig",
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
    "PrimaryScore",
    "PassCriteria",
    # Collection models
    "Resource",
    "Collection",
    "CollectionList",
    "CollectionRef",
    # Framework models
    "FrameworkInfo",
    # Response models
    "ErrorInfo",
    "ErrorResponse",
    "HealthResponse",
]
