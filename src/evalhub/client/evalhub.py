"""Main EvalHub clients that combine all client capabilities."""

from __future__ import annotations

from functools import cached_property

from .base import BaseAsyncClient, BaseSyncClient
from .resources import (
    AsyncBenchmarksResource,
    AsyncCollectionsResource,
    AsyncJobsResource,
    AsyncProvidersResource,
    SyncBenchmarksResource,
    SyncCollectionsResource,
    SyncJobsResource,
    SyncProvidersResource,
)


class AsyncEvalHubClient(BaseAsyncClient):
    """Complete asynchronous EvalHub client with all capabilities.

    This client provides access to all EvalHub API endpoints through a nested
    resource structure.

    All methods are async and must be awaited.

    Example:
        >>> async with AsyncEvalHubClient() as client:
        ...     providers = await client.providers.list()
        ...     benchmarks = await client.benchmarks.list()
        ...     job = await client.jobs.submit(request)
        ...     status = await client.jobs.get(job.job_id)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        auth_token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        """Initialize the async EvalHub client.

        Args:
            base_url: Base URL of the EvalHub service (default: http://localhost:8080)
            auth_token: Optional authentication token for API access
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retry attempts (default: 3)
            verify_ssl: Whether to verify SSL certificates (default: True)
        """
        super().__init__(
            base_url=base_url,
            auth_token=auth_token,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
        )

    @cached_property
    def providers(self) -> AsyncProvidersResource:
        """Access provider operations."""
        return AsyncProvidersResource(self)

    @cached_property
    def benchmarks(self) -> AsyncBenchmarksResource:
        """Access benchmark operations."""
        return AsyncBenchmarksResource(self)

    @cached_property
    def collections(self) -> AsyncCollectionsResource:
        """Access collection operations."""
        return AsyncCollectionsResource(self)

    @cached_property
    def jobs(self) -> AsyncJobsResource:
        """Access evaluation job operations."""
        return AsyncJobsResource(self)


class SyncEvalHubClient(BaseSyncClient):
    """Complete synchronous EvalHub client with all capabilities.

    This client provides access to all EvalHub API endpoints through a nested
    resource structure.

    All methods are synchronous and do not require await.

    Example:
        >>> with SyncEvalHubClient() as client:
        ...     providers = client.providers.list()
        ...     benchmarks = client.benchmarks.list()
        ...     job = client.jobs.submit(request)
        ...     status = client.jobs.get(job.job_id)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        auth_token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        """Initialize the sync EvalHub client.

        Args:
            base_url: Base URL of the EvalHub service (default: http://localhost:8080)
            auth_token: Optional authentication token for API access
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retry attempts (default: 3)
            verify_ssl: Whether to verify SSL certificates (default: True)
        """
        super().__init__(
            base_url=base_url,
            auth_token=auth_token,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
        )

    @cached_property
    def providers(self) -> SyncProvidersResource:
        """Access provider operations."""
        return SyncProvidersResource(self)

    @cached_property
    def benchmarks(self) -> SyncBenchmarksResource:
        """Access benchmark operations."""
        return SyncBenchmarksResource(self)

    @cached_property
    def collections(self) -> SyncCollectionsResource:
        """Access collection operations."""
        return SyncCollectionsResource(self)

    @cached_property
    def jobs(self) -> SyncJobsResource:
        """Access evaluation job operations."""
        return SyncJobsResource(self)


# Aliases for backward compatibility and convenience
EvalHubClient = AsyncEvalHubClient  # Default to async
