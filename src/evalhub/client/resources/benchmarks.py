"""Benchmark resource for EvalHub client."""

from __future__ import annotations

import logging

from ...models import Benchmark, BenchmarksList
from ..base import BaseAsyncClient, BaseSyncClient

logger = logging.getLogger(__name__)


class AsyncBenchmarksResource:
    """Asynchronous resource for benchmark operations."""

    def __init__(self, client: BaseAsyncClient):
        self._client = client

    async def list(
        self,
        provider_id: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Benchmark]:
        """List available benchmarks.

        Args:
            provider_id: Filter by provider (optional)
            category: Filter by category (optional)
            limit: Maximum number of benchmarks to return (optional)

        Returns:
            list[Benchmark]: List of benchmarks

        Raises:
            httpx.HTTPError: If request fails
        """
        params = {}
        if provider_id:
            params["provider_id"] = provider_id
        if category:
            params["category"] = category
        if limit:
            params["limit"] = str(limit)

        response = await self._client._request_get(
            "/evaluations/benchmarks", params=params
        )
        data = response.json()
        benchmarks_list = BenchmarksList(**data)
        return benchmarks_list.items


class SyncBenchmarksResource:
    """Synchronous resource for benchmark operations."""

    def __init__(self, client: BaseSyncClient):
        self._client = client

    def list(
        self,
        provider_id: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Benchmark]:
        """List available benchmarks.

        Args:
            provider_id: Filter by provider (optional)
            category: Filter by category (optional)
            limit: Maximum number of benchmarks to return (optional)

        Returns:
            list[Benchmark]: List of benchmarks

        Raises:
            httpx.HTTPError: If request fails
        """
        params = {}
        if provider_id:
            params["provider_id"] = provider_id
        if category:
            params["category"] = category
        if limit:
            params["limit"] = str(limit)

        response = self._client._request_get("/evaluations/benchmarks", params=params)
        data = response.json()
        benchmarks_list = BenchmarksList(**data)
        return benchmarks_list.items
