"""Provider resource for EvalHub client."""

from __future__ import annotations

import logging

from ...models import Provider, ProviderList
from ..base import BaseAsyncClient, BaseSyncClient

logger = logging.getLogger(__name__)


class AsyncProvidersResource:
    """Asynchronous resource for provider operations."""

    def __init__(self, client: BaseAsyncClient):
        self._client = client

    async def list(self) -> list[Provider]:
        """List all registered providers.

        Returns:
            list[Provider]: List of provider information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._client._request_get("/evaluations/providers")
        data = response.json()
        provider_list = ProviderList(**data)
        return provider_list.items

    async def get(self, provider_id: str) -> Provider:
        """Get information about a specific provider.

        Args:
            provider_id: The provider identifier

        Returns:
            Provider: Provider information

        Raises:
            httpx.HTTPError: If provider not found or request fails
        """
        response = await self._client._request_get(
            "/evaluations/providers", params={"id": provider_id}
        )
        provider_list = ProviderList(**response.json())
        if provider_list.total_count != 1:
            raise ValueError(
                f"Expected exactly 1 provider for id '{provider_id}', got {provider_list.total_count}"
            )
        return provider_list.items[0]


class SyncProvidersResource:
    """Synchronous resource for provider operations."""

    def __init__(self, client: BaseSyncClient):
        self._client = client

    def list(self) -> list[Provider]:
        """List all registered providers.

        Returns:
            list[Provider]: List of provider information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = self._client._request_get("/evaluations/providers")
        data = response.json()
        provider_list = ProviderList(**data)
        return provider_list.items

    def get(self, provider_id: str) -> Provider:
        """Get information about a specific provider.

        Args:
            provider_id: The provider identifier

        Returns:
            Provider: Provider information

        Raises:
            httpx.HTTPError: If provider not found or request fails
        """
        response = self._client._request_get(
            "/evaluations/providers", params={"id": provider_id}
        )
        provider_list = ProviderList(**response.json())
        if provider_list.total_count != 1:
            raise ValueError(
                f"Expected exactly 1 provider for id '{provider_id}', got {provider_list.total_count}"
            )
        return provider_list.items[0]
