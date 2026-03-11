"""Collection resource for EvalHub client."""

from __future__ import annotations

import logging

from ...models import Collection, CollectionList
from ..base import BaseAsyncClient, BaseSyncClient

logger = logging.getLogger(__name__)


class AsyncCollectionsResource:
    """Asynchronous resource for collection operations."""

    def __init__(self, client: BaseAsyncClient):
        self._client = client

    async def list(self, *, tenant: str | None = None) -> list[Collection]:
        """List all available benchmark collections.

        Args:
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            list[Collection]: List of collection information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._client._request_get(
            "/evaluations/collections", tenant=tenant
        )
        data = response.json()
        collection_list = CollectionList(**data)
        return collection_list.items

    async def get(self, collection_id: str, *, tenant: str | None = None) -> Collection:
        """Get information about a specific collection.

        Args:
            collection_id: The collection identifier
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            Collection: Collection information including benchmarks

        Raises:
            httpx.HTTPError: If collection not found or request fails
        """
        response = await self._client._request_get(
            f"/evaluations/collections/{collection_id}", tenant=tenant
        )
        return Collection(**response.json())


class SyncCollectionsResource:
    """Synchronous resource for collection operations."""

    def __init__(self, client: BaseSyncClient):
        self._client = client

    def list(self, *, tenant: str | None = None) -> list[Collection]:
        """List all available benchmark collections.

        Args:
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            list[Collection]: List of collection information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = self._client._request_get("/evaluations/collections", tenant=tenant)
        data = response.json()
        collection_list = CollectionList(**data)
        return collection_list.items

    def get(self, collection_id: str, *, tenant: str | None = None) -> Collection:
        """Get information about a specific collection.

        Args:
            collection_id: The collection identifier
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            Collection: Collection information including benchmarks

        Raises:
            httpx.HTTPError: If collection not found or request fails
        """
        response = self._client._request_get(
            f"/evaluations/collections/{collection_id}", tenant=tenant
        )
        return Collection(**response.json())
