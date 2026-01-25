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

# Core extra - HTTP client functionality (adapter client)
try:
    from .adapter.client import AdapterClient, AdapterDiscovery

    __all__.extend(["AdapterClient", "AdapterDiscovery"])
except ImportError:
    pass

# Adapter extra - server and API components
try:
    from .adapter.api import AdapterAPIRouter, create_adapter_api
    from .adapter.models import AdapterConfig, AdapterMetadata, FrameworkAdapter
    from .adapter.server import AdapterServer

    __all__.extend(
        [
            "AdapterAPIRouter",
            "create_adapter_api",
            "AdapterConfig",
            "AdapterMetadata",
            "FrameworkAdapter",
            "AdapterServer",
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
