"""Client for EvalHub provider and benchmark operations."""

from __future__ import annotations

import builtins
import logging

from ..models import (
    BenchmarkInfo,
    Collection,
    CollectionList,
    Provider,
    ProviderList,
)
from .base import BaseAsyncClient, BaseSyncClient

logger = logging.getLogger(__name__)


class AsyncProvidersClient(BaseAsyncClient):
    """Asynchronous client for provider and benchmark operations.

    Inherits from BaseAsyncClient and provides methods for discovering and
    querying evaluation providers and benchmarks.
    """

    async def list(self) -> builtins.list[Provider]:
        """List all registered providers.

        Returns:
            list[Provider]: List of provider information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._request_get("/evaluations/providers")
        data = response.json()
        provider_list = ProviderList(**data)
        return provider_list.items

    async def get_provider(self, provider_id: str) -> Provider:
        """Get information about a specific provider.

        Args:
            provider_id: The provider identifier

        Returns:
            Provider: Provider information

        Raises:
            httpx.HTTPError: If provider not found or request fails
        """
        response = await self._request_get(f"/evaluations/providers/{provider_id}")
        return Provider(**response.json())

    async def list_benchmarks(
        self,
        provider_id: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> builtins.list[BenchmarkInfo]:
        """List available benchmarks.

        Args:
            provider_id: Filter by provider (optional)
            category: Filter by category (optional)
            limit: Maximum number of benchmarks to return (optional)

        Returns:
            list[BenchmarkInfo]: List of benchmarks

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

        response = await self._request_get("/evaluations/benchmarks", params=params)
        data = response.json()
        # Convert API Benchmark format to BenchmarkInfo
        benchmarks: builtins.list[BenchmarkInfo] = []
        for item in data.get("items", []):
            # Map API fields to BenchmarkInfo fields
            benchmark_info = BenchmarkInfo(
                benchmark_id=item.get("id", item.get("benchmark_id")),
                name=item.get("label", item.get("name")),
                description=item.get("description", ""),
                category=item.get("category"),
                metrics=item.get("metrics", []),
                tags=item.get("tags", []),
                dataset_size=item.get("dataset_size"),
                supports_few_shot=True,  # Assume true by default
                default_few_shot=item.get("num_few_shot"),
            )
            benchmarks.append(benchmark_info)
        return benchmarks

    async def list_collections(self) -> builtins.list[Collection]:
        """List all available benchmark collections.

        Returns:
            list[Collection]: List of collection information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._request_get("/evaluations/collections")
        data = response.json()
        collection_list = CollectionList(**data)
        return collection_list.items

    async def get_collection(self, collection_id: str) -> Collection:
        """Get information about a specific collection.

        Args:
            collection_id: The collection identifier

        Returns:
            Collection: Collection information including benchmarks

        Raises:
            httpx.HTTPError: If collection not found or request fails
        """
        response = await self._request(
            "GET", f"/evaluations/collections/{collection_id}"
        )
        return Collection(**response.json())


class SyncProvidersClient(BaseSyncClient):
    """Synchronous client for provider and benchmark operations.

    Inherits from BaseSyncClient and provides methods for discovering and
    querying evaluation providers and benchmarks.
    """

    def list(self) -> builtins.list[Provider]:
        """List all registered providers.

        Returns:
            list[Provider]: List of provider information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = self._request_get("/evaluations/providers")
        data = response.json()
        provider_list = ProviderList(**data)
        return provider_list.items

    def get_provider(self, provider_id: str) -> Provider:
        """Get information about a specific provider.

        Args:
            provider_id: The provider identifier

        Returns:
            Provider: Provider information

        Raises:
            httpx.HTTPError: If provider not found or request fails
        """
        response = self._request_get(f"/evaluations/providers/{provider_id}")
        return Provider(**response.json())

    def list_benchmarks(
        self,
        provider_id: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> builtins.list[BenchmarkInfo]:
        """List available benchmarks.

        Args:
            provider_id: Filter by provider (optional)
            category: Filter by category (optional)
            limit: Maximum number of benchmarks to return (optional)

        Returns:
            list[BenchmarkInfo]: List of benchmarks

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

        response = self._request_get("/evaluations/benchmarks", params=params)
        data = response.json()
        # Convert API Benchmark format to BenchmarkInfo
        benchmarks: builtins.list[BenchmarkInfo] = []
        for item in data.get("items", []):
            # Map API fields to BenchmarkInfo fields
            benchmark_info = BenchmarkInfo(
                benchmark_id=item.get("id", item.get("benchmark_id")),
                name=item.get("label", item.get("name")),
                description=item.get("description", ""),
                category=item.get("category"),
                metrics=item.get("metrics", []),
                tags=item.get("tags", []),
                dataset_size=item.get("dataset_size"),
                supports_few_shot=True,  # Assume true by default
                default_few_shot=item.get("num_few_shot"),
            )
            benchmarks.append(benchmark_info)
        return benchmarks

    def list_collections(self) -> builtins.list[Collection]:
        """List all available benchmark collections.

        Returns:
            list[Collection]: List of collection information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = self._request_get("/evaluations/collections")
        data = response.json()
        collection_list = CollectionList(**data)
        return collection_list.items

    def get_collection(self, collection_id: str) -> Collection:
        """Get information about a specific collection.

        Args:
            collection_id: The collection identifier

        Returns:
            Collection: Collection information including benchmarks

        Raises:
            httpx.HTTPError: If collection not found or request fails
        """
        response = self._request_get(f"/evaluations/collections/{collection_id}")
        return Collection(**response.json())
