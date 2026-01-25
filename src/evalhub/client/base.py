"""Base client for EvalHub API communication."""

from __future__ import annotations

import logging
from typing import Any, cast

import httpx
from typing_extensions import Self

logger = logging.getLogger(__name__)


class ClientError(Exception):
    """Base exception for client errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class BaseAsyncClient:
    """Base async client for EvalHub API communication.

    Provides common HTTP client functionality, authentication, and error handling
    for asynchronous operations.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        auth_token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        """Initialize the base async client.

        Args:
            base_url: Base URL of the EvalHub service
            auth_token: Optional authentication token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"
        self.auth_token = auth_token
        self.max_retries = max_retries

        # Build headers
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # Create async HTTP client
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
            verify=verify_ssl,
            headers=headers,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx

        Returns:
            httpx.Response: Response object

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        url = f"{self.api_base}{path}"

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except httpx.TimeoutException:
                if attempt == self.max_retries:
                    logger.error(
                        f"Request to {url} timed out after {self.max_retries} retries"
                    )
                    raise
                logger.warning(
                    f"Request to {url} timed out, retrying ({attempt + 1}/{self.max_retries})"
                )

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx), only server errors (5xx)
                if e.response.status_code < 500 or attempt == self.max_retries:
                    raise
                logger.warning(
                    f"Server error {e.response.status_code} for {url}, retrying ({attempt + 1}/{self.max_retries})"
                )

            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Connection error to {url} after {self.max_retries} retries: {e}"
                    )
                    raise
                logger.warning(
                    f"Connection error to {url}, retrying ({attempt + 1}/{self.max_retries}): {e}"
                )

        # This should never be reached, but mypy needs a return
        raise RuntimeError("Request retry loop completed without returning")

    async def _request_get(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make GET request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx (params, headers, etc.)

        Returns:
            httpx.Response: Response object
        """
        return await self._request("GET", path, **kwargs)

    async def _request_post(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make POST request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx (json, data, etc.)

        Returns:
            httpx.Response: Response object
        """
        return await self._request("POST", path, **kwargs)

    async def _request_put(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make PUT request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx (json, data, etc.)

        Returns:
            httpx.Response: Response object
        """
        return await self._request("PUT", path, **kwargs)

    async def _request_delete(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make DELETE request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx

        Returns:
            httpx.Response: Response object
        """
        return await self._request("DELETE", path, **kwargs)

    async def _request_patch(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make PATCH request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx (json, data, etc.)

        Returns:
            httpx.Response: Response object
        """
        return await self._request("PATCH", path, **kwargs)

    async def health(self) -> dict[str, Any]:
        """Check the health of the EvalHub service.

        Returns:
            dict: Health status response

        Raises:
            httpx.HTTPError: If health check fails
        """
        response = await self._request_get("/health")
        return cast(dict[str, Any], response.json())


class BaseSyncClient:
    """Base synchronous client for EvalHub API communication.

    Provides common HTTP client functionality, authentication, and error handling
    for synchronous operations.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        auth_token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        """Initialize the base sync client.

        Args:
            base_url: Base URL of the EvalHub service
            auth_token: Optional authentication token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"
        self.auth_token = auth_token
        self.max_retries = max_retries

        # Build headers
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # Create sync HTTP client
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
            verify=verify_ssl,
            headers=headers,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx

        Returns:
            httpx.Response: Response object

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        url = f"{self.api_base}{path}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except httpx.TimeoutException:
                if attempt == self.max_retries:
                    logger.error(
                        f"Request to {url} timed out after {self.max_retries} retries"
                    )
                    raise
                logger.warning(
                    f"Request to {url} timed out, retrying ({attempt + 1}/{self.max_retries})"
                )

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx), only server errors (5xx)
                if e.response.status_code < 500 or attempt == self.max_retries:
                    raise
                logger.warning(
                    f"Server error {e.response.status_code} for {url}, retrying ({attempt + 1}/{self.max_retries})"
                )

            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Connection error to {url} after {self.max_retries} retries: {e}"
                    )
                    raise
                logger.warning(
                    f"Connection error to {url}, retrying ({attempt + 1}/{self.max_retries}): {e}"
                )

        # This should never be reached, but mypy needs a return
        raise RuntimeError("Request retry loop completed without returning")

    def _request_get(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make GET request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx (params, headers, etc.)

        Returns:
            httpx.Response: Response object
        """
        return self._request("GET", path, **kwargs)

    def _request_post(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make POST request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx (json, data, etc.)

        Returns:
            httpx.Response: Response object
        """
        return self._request("POST", path, **kwargs)

    def _request_put(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make PUT request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx (json, data, etc.)

        Returns:
            httpx.Response: Response object
        """
        return self._request("PUT", path, **kwargs)

    def _request_delete(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make DELETE request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx

        Returns:
            httpx.Response: Response object
        """
        return self._request("DELETE", path, **kwargs)

    def _request_patch(self, path: str, **kwargs: Any) -> httpx.Response:
        """Make PATCH request.

        Args:
            path: API path (without base URL)
            **kwargs: Additional arguments for httpx (json, data, etc.)

        Returns:
            httpx.Response: Response object
        """
        return self._request("PATCH", path, **kwargs)

    def health(self) -> dict[str, Any]:
        """Check the health of the EvalHub service.

        Returns:
            dict: Health status response

        Raises:
            httpx.HTTPError: If health check fails
        """
        response = self._request_get("/health")
        return cast(dict[str, Any], response.json())
