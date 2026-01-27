"""EvalHub SDK - Framework adapter SDK for integrating with TrustyAI EvalHub.

This SDK provides a standardized way to create framework adapters that can
be consumed by EvalHub, enabling a "Bring Your Own Framework" (BYOF) approach.

Installation extras:
  - core: Basic functionality for HTTP client operations
  - adapter: Components for building custom evaluation framework adapters
  - client: High-level Python API for end users
  - cli: Command-line interface
  - all: All functionality except examples
"""

# Always available - core models
from .models import (
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
    ModelConfig,
)

__version__ = "0.1.0"

# Base exports - always available
__all__ = [
    "__version__",
    # Core data models
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
    "SDK for building framework adapters that integrate with TrustyAI EvalHub"
)
__author__ = "TrustyAI Team"
__author_email__ = "trustyai@redhat.com"
__license__ = "Apache 2.0"
