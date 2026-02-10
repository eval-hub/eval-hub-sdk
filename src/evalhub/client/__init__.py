"""EvalHub client SDK for interacting with EvalHub REST API.

The SDK provides separate client classes for async and sync operations,
with a nested resource structure.

Example (async):
    >>> from evalhub import AsyncEvalHubClient
    >>> async with AsyncEvalHubClient() as client:
    ...     # Provider operations
    ...     providers = await client.providers.list()
    ...     provider = await client.providers.get("provider_id")
    ...
    ...     # Benchmark operations
    ...     benchmarks = await client.benchmarks.list()
    ...     benchmarks_math = await client.benchmarks.list(category="math")
    ...
    ...     # Collection operations
    ...     collections = await client.collections.list()
    ...     collection = await client.collections.get("collection_id")
    ...
    ...     # Job operations
    ...     job = await client.jobs.submit(request)
    ...     status = await client.jobs.get(job.id)
    ...     jobs = await client.jobs.list()

Example (synchronous):
    >>> from evalhub import SyncEvalHubClient
    >>> with SyncEvalHubClient() as client:
    ...     # Provider operations (no await needed)
    ...     providers = client.providers.list()
    ...     provider = client.providers.get("provider_id")
    ...
    ...     # Benchmark operations
    ...     benchmarks = client.benchmarks.list()
    ...     benchmarks_math = client.benchmarks.list(category="math")
    ...
    ...     # Collection operations
    ...     collections = client.collections.list()
    ...     collection = client.collections.get("collection_id")
    ...
    ...     # Job operations
    ...     job = client.jobs.submit(request)
    ...     status = client.jobs.get(job.id)
    ...     jobs = client.jobs.list()

Note: EvalHubClient is an alias for AsyncEvalHubClient (async by default).
"""

from .base import (
    BaseAsyncClient,
    BaseSyncClient,
    ClientError,
    JobCanNotBeCancelledError,
    JobNotFoundError,
)
from .evalhub import AsyncEvalHubClient, EvalHubClient, SyncEvalHubClient
from .evaluations import AsyncEvaluationsClient, SyncEvaluationsClient
from .providers import AsyncProvidersClient, SyncProvidersClient

__all__ = [
    # Base classes
    "BaseAsyncClient",
    "BaseSyncClient",
    "ClientError",
    "JobNotFoundError",
    "JobCanNotBeCancelledError",
    # Main clients (recommended)
    "AsyncEvalHubClient",
    "SyncEvalHubClient",
    "EvalHubClient",  # Alias for AsyncEvalHubClient
    # Legacy clients (for backward compatibility)
    "AsyncProvidersClient",
    "AsyncEvaluationsClient",
    "SyncProvidersClient",
    "SyncEvaluationsClient",
]
