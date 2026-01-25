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

    async def list(self) -> list[Collection]:
        """List all available benchmark collections.

        Returns:
            list[Collection]: List of collection information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._client._request_get("/evaluations/collections")
        data = response.json()
        collection_list = CollectionList(**data)
        return collection_list.items

    async def get(self, collection_id: str) -> Collection:
        """Get information about a specific collection.

        Args:
            collection_id: The collection identifier

        Returns:
            Collection: Collection information including benchmarks

        Raises:
            httpx.HTTPError: If collection not found or request fails
        """
        response = await self._client._request_get(
            f"/evaluations/collections/{collection_id}"
        )
        return Collection(**response.json())


class SyncCollectionsResource:
    """Synchronous resource for collection operations."""

    def __init__(self, client: BaseSyncClient):
        self._client = client

    def list(self) -> list[Collection]:
        """List all available benchmark collections.

        Returns:
            list[Collection]: List of collection information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = self._client._request_get("/evaluations/collections")
        data = response.json()
        collection_list = CollectionList(**data)
        return collection_list.items

    def get(self, collection_id: str) -> Collection:
        """Get information about a specific collection.

        Args:
            collection_id: The collection identifier

        Returns:
            Collection: Collection information including benchmarks

        Raises:
            httpx.HTTPError: If collection not found or request fails
        """
        response = self._client._request_get(
            f"/evaluations/collections/{collection_id}"
        )
        return Collection(**response.json())
