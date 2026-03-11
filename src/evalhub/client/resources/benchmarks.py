"""Benchmark resource for EvalHub client.

Benchmarks are nested inside providers on the server. This resource fetches
providers and extracts/filters benchmarks client-side.
"""

from __future__ import annotations

import logging

from ...models import Benchmark, ProviderList
from ..base import BaseAsyncClient, BaseSyncClient

logger = logging.getLogger(__name__)


def _extract_benchmarks(
    data: dict,
    provider_id: str | None,
    category: str | None,
    limit: int | None,
) -> list[Benchmark]:
    """Extract and filter benchmarks from a providers response."""
    provider_list = ProviderList(**data)
    benchmarks: list[Benchmark] = []
    for provider in provider_list.items:
        if provider_id and provider.resource.id != provider_id:
            continue
        for b in provider.benchmarks:
            if category and b.category != category:
                continue
            benchmarks.append(b)
    if limit:
        benchmarks = benchmarks[:limit]
    return benchmarks


class AsyncBenchmarksResource:
    """Asynchronous resource for benchmark operations."""

    def __init__(self, client: BaseAsyncClient):
        self._client = client

    async def list(
        self,
        provider_id: str | None = None,
        category: str | None = None,
        limit: int | None = None,
        *,
        tenant: str | None = None,
    ) -> list[Benchmark]:
        """List available benchmarks.

        Fetches providers and extracts their benchmarks, applying optional
        filters client-side.

        Args:
            provider_id: Filter by provider (optional)
            category: Filter by category (optional)
            limit: Maximum number of benchmarks to return (optional)
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            list[Benchmark]: List of benchmarks

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._client._request_get(
            "/evaluations/providers", tenant=tenant
        )
        return _extract_benchmarks(response.json(), provider_id, category, limit)


class SyncBenchmarksResource:
    """Synchronous resource for benchmark operations."""

    def __init__(self, client: BaseSyncClient):
        self._client = client

    def list(
        self,
        provider_id: str | None = None,
        category: str | None = None,
        limit: int | None = None,
        *,
        tenant: str | None = None,
    ) -> list[Benchmark]:
        """List available benchmarks.

        Fetches providers and extracts their benchmarks, applying optional
        filters client-side.

        Args:
            provider_id: Filter by provider (optional)
            category: Filter by category (optional)
            limit: Maximum number of benchmarks to return (optional)
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            list[Benchmark]: List of benchmarks

        Raises:
            httpx.HTTPError: If request fails
        """
        response = self._client._request_get("/evaluations/providers", tenant=tenant)
        return _extract_benchmarks(response.json(), provider_id, category, limit)
