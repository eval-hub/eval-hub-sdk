"""Standalone post-processing resource for the EvalHub client."""

from __future__ import annotations

from ...models import PostProcessingResource, StandalonePostProcessingRequest
from ..base import BaseAsyncClient, BaseSyncClient


class AsyncPostProcessingResource:
    """Asynchronous resource for standalone post-processing operations."""

    def __init__(self, client: BaseAsyncClient):
        self._client = client

    async def submit(
        self,
        request: StandalonePostProcessingRequest,
        *,
        tenant: str | None = None,
    ) -> PostProcessingResource:
        """Submit a standalone post-processing job.

        Args:
            request: The standalone post-processing request.
            tenant: Tenant override for this request (default: client-level tenant).

        Returns:
            PostProcessingResource: The submitted post-processing job.

        Raises:
            httpx.HTTPError: If the request fails or is invalid.
        """
        response = await self._client._request_post(
            "/evaluations/post-processing",
            json=request.model_dump(mode="json", exclude_none=True, by_alias=True),
            tenant=tenant,
        )
        return PostProcessingResource(**response.json())

    async def get(
        self,
        post_processing_job_id: str,
        *,
        tenant: str | None = None,
    ) -> PostProcessingResource:
        """Get a standalone post-processing job.

        Args:
            post_processing_job_id: The post-processing job identifier.
            tenant: Tenant override for this request (default: client-level tenant).

        Returns:
            PostProcessingResource: The current post-processing job.

        Raises:
            httpx.HTTPError: If the job is not found or the request fails.
        """
        response = await self._client._request_get(
            f"/evaluations/post-processing/{post_processing_job_id}",
            tenant=tenant,
        )
        return PostProcessingResource(**response.json())

    async def delete(
        self,
        post_processing_job_id: str,
        *,
        tenant: str | None = None,
    ) -> None:
        """Delete a standalone post-processing job.

        Args:
            post_processing_job_id: The post-processing job identifier.
            tenant: Tenant override for this request (default: client-level tenant).

        Raises:
            httpx.HTTPError: If the job is not found or the request fails.
        """
        await self._client._request_delete(
            f"/evaluations/post-processing/{post_processing_job_id}",
            tenant=tenant,
        )


class SyncPostProcessingResource:
    """Synchronous resource for standalone post-processing operations."""

    def __init__(self, client: BaseSyncClient):
        self._client = client

    def submit(
        self,
        request: StandalonePostProcessingRequest,
        *,
        tenant: str | None = None,
    ) -> PostProcessingResource:
        """Submit a standalone post-processing job.

        Args:
            request: The standalone post-processing request.
            tenant: Tenant override for this request (default: client-level tenant).

        Returns:
            PostProcessingResource: The submitted post-processing job.

        Raises:
            httpx.HTTPError: If the request fails or is invalid.
        """
        response = self._client._request_post(
            "/evaluations/post-processing",
            json=request.model_dump(mode="json", exclude_none=True, by_alias=True),
            tenant=tenant,
        )
        return PostProcessingResource(**response.json())

    def get(
        self,
        post_processing_job_id: str,
        *,
        tenant: str | None = None,
    ) -> PostProcessingResource:
        """Get a standalone post-processing job.

        Args:
            post_processing_job_id: The post-processing job identifier.
            tenant: Tenant override for this request (default: client-level tenant).

        Returns:
            PostProcessingResource: The current post-processing job.

        Raises:
            httpx.HTTPError: If the job is not found or the request fails.
        """
        response = self._client._request_get(
            f"/evaluations/post-processing/{post_processing_job_id}",
            tenant=tenant,
        )
        return PostProcessingResource(**response.json())

    def delete(
        self,
        post_processing_job_id: str,
        *,
        tenant: str | None = None,
    ) -> None:
        """Delete a standalone post-processing job.

        Args:
            post_processing_job_id: The post-processing job identifier.
            tenant: Tenant override for this request (default: client-level tenant).

        Raises:
            httpx.HTTPError: If the job is not found or the request fails.
        """
        self._client._request_delete(
            f"/evaluations/post-processing/{post_processing_job_id}",
            tenant=tenant,
        )
