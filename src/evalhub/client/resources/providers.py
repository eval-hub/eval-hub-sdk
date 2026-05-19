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

    async def list(self, *, tenant: str | None = None) -> list[Provider]:
        """List all registered providers.

        Args:
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            list[Provider]: List of provider information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._client._request_get(
            "/evaluations/providers", tenant=tenant
        )
        data = response.json()
        provider_list = ProviderList(**data)
        return provider_list.items

    async def get(self, provider_id: str, *, tenant: str | None = None) -> Provider:
        """Get information about a specific provider.

        Args:
            provider_id: The provider identifier
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            Provider: Provider information

        Raises:
            httpx.HTTPError: If provider not found or request fails
        """
        response = await self._client._request_get(
            f"/evaluations/providers/{provider_id}", tenant=tenant
        )
        return Provider(**response.json())

    async def create(self, data: dict, *, tenant: str | None = None) -> Provider:
        """Create a new evaluation provider.

        Args:
            data: Provider specification as a dict (name, title, description, runtime, benchmarks, ...)
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            Provider: The newly created provider

        Raises:
            httpx.HTTPError: If the request fails
        """
        response = await self._client._request_post(
            "/evaluations/providers", json=data, tenant=tenant
        )
        return Provider(**response.json())

    async def delete(self, provider_id: str, *, tenant: str | None = None) -> None:
        """Delete an evaluation provider.

        Args:
            provider_id: The provider identifier
            tenant: Tenant override for this request (default: client-level tenant)

        Raises:
            httpx.HTTPError: If the provider is not found or request fails
        """
        await self._client._request_delete(
            f"/evaluations/providers/{provider_id}", tenant=tenant
        )


class SyncProvidersResource:
    """Synchronous resource for provider operations."""

    def __init__(self, client: BaseSyncClient):
        self._client = client

    def list(self, *, tenant: str | None = None) -> list[Provider]:
        """List all registered providers.

        Args:
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            list[Provider]: List of provider information

        Raises:
            httpx.HTTPError: If request fails
        """
        response = self._client._request_get("/evaluations/providers", tenant=tenant)
        data = response.json()
        provider_list = ProviderList(**data)
        return provider_list.items

    def get(self, provider_id: str, *, tenant: str | None = None) -> Provider:
        """Get information about a specific provider.

        Args:
            provider_id: The provider identifier
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            Provider: Provider information

        Raises:
            httpx.HTTPError: If provider not found or request fails
        """
        response = self._client._request_get(
            f"/evaluations/providers/{provider_id}", tenant=tenant
        )
        return Provider(**response.json())

    def create(self, data: dict, *, tenant: str | None = None) -> Provider:
        """Create a new evaluation provider.

        Args:
            data: Provider specification as a dict (name, title, description, runtime, benchmarks, ...)
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            Provider: The newly created provider

        Raises:
            httpx.HTTPError: If the request fails
        """
        response = self._client._request_post(
            "/evaluations/providers", json=data, tenant=tenant
        )
        return Provider(**response.json())

    def delete(self, provider_id: str, *, tenant: str | None = None) -> None:
        """Delete an evaluation provider.

        Args:
            provider_id: The provider identifier
            tenant: Tenant override for this request (default: client-level tenant)

        Raises:
            httpx.HTTPError: If the provider is not found or request fails
        """
        self._client._request_delete(
            f"/evaluations/providers/{provider_id}", tenant=tenant
        )
