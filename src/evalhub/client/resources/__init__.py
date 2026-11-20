"""Resource modules for EvalHub client."""

from .benchmarks import AsyncBenchmarksResource, SyncBenchmarksResource
from .collections import AsyncCollectionsResource, SyncCollectionsResource
from .jobs import AsyncJobsResource, SyncJobsResource
from .post_processing import (
    AsyncPostProcessingResource,
    SyncPostProcessingResource,
)
from .providers import AsyncProvidersResource, SyncProvidersResource

__all__ = [
    "AsyncBenchmarksResource",
    "AsyncCollectionsResource",
    "AsyncJobsResource",
    "AsyncPostProcessingResource",
    "AsyncProvidersResource",
    "SyncBenchmarksResource",
    "SyncCollectionsResource",
    "SyncJobsResource",
    "SyncPostProcessingResource",
    "SyncProvidersResource",
]
