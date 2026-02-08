"""EvalHub SDK - Python SDK for interacting with EvalHub.

This SDK provides three main components:

1. **Common Models**: Core data models for EvalHub (always available)
   - Benchmarks, evaluations, jobs, providers, and collections
   - Request/response schemas and status enums

2. **Python Client**: High-level API for interacting with EvalHub REST endpoints
   - Synchronous and asynchronous clients
   - Evaluation job submission and management
   - Provider and benchmark discovery

3. **Framework Adapters**: SDK for building custom evaluation framework adapters
   - Standardised interfaces for "Bring Your Own Framework" (BYOF) approach
   - Integration patterns for custom evaluation frameworks

Installation extras:
  - core: Basic functionality for HTTP client operations
  - adapter: Components for building custom evaluation framework adapters
  - client: High-level Python API for end users
  - cli: Command-line interface
  - all: All functionality except examples
"""

# Always available - core models
from .models import (
    BenchmarkConfig,
    BenchmarkInfo,
    ErrorResponse,
    EvaluationJob,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    EvaluationStatus,
    FrameworkInfo,
    HealthResponse,
    JobStatus,
    JobSubmissionRequest,
    ModelConfig,
)

__version__ = "0.1.0a5"

# Base exports - always available
__all__ = [
    "__version__",
    # Core data models
    "BenchmarkConfig",
    "BenchmarkInfo",
    "ErrorResponse",
    "EvaluationJob",
    "EvaluationRequest",
    "EvaluationResponse",
    "EvaluationResult",
    "EvaluationStatus",
    "FrameworkInfo",
    "HealthResponse",
    "JobStatus",
    "JobSubmissionRequest",
    "ModelConfig",
]

# Conditional imports based on available extras

# Client extra - EvalHub client library
try:
    from .client import (
        AsyncEvalHubClient,
        AsyncEvaluationsClient,
        AsyncProvidersClient,
        EvalHubClient,
        SyncEvalHubClient,
        SyncEvaluationsClient,
        SyncProvidersClient,
    )

    __all__.extend(
        [
            "AsyncEvalHubClient",
            "AsyncProvidersClient",
            "AsyncEvaluationsClient",
            "SyncEvalHubClient",
            "SyncProvidersClient",
            "SyncEvaluationsClient",
            "EvalHubClient",  # Alias for AsyncEvalHubClient
        ]
    )
except ImportError:
    pass

# Package metadata
__title__ = "eval-hub"
__description__ = (
    "Python SDK for EvalHub: common models, REST API client, and framework adapter SDK"
)
__author__ = "AI Evaluations Team"
__author_email__ = "rui@redhat.com"
__license__ = "Apache 2.0"
