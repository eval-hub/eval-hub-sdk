"""Base client for EvalHub API communication."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Self, cast

import httpx

logger = logging.getLogger(__name__)


def _calculate_retry_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float,
    randomization: bool,
) -> float:
    """Calculate retry delay with exponential backoff and optional jitter.

    Args:
        attempt: Current retry attempt number (0-indexed)
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        randomization: Whether to add random jitter to prevent thundering herd

    Returns:
        float: Delay in seconds before next retry
    """
    # Calculate exponential backoff: initial_delay * (backoff_factor ^ attempt)
    delay = min(initial_delay * (backoff_factor**attempt), max_delay)

    # Add jitter if enabled (random value between 0 and delay)
    if randomization:
        delay = delay * (0.5 + random.random() * 0.5)

    return delay


class ClientError(Exception):
    """Base exception for client errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class BaseAsyncClient:
    """Base async client for EvalHub API communication.

    Provides common HTTP client functionality, authentication, and error handling
    for asynchronous operations with exponential backoff retry logic.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        auth_token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        retry_backoff_factor: float = 2.0,
        retry_randomization: bool = True,
    ):
        """Initialize the base async client.

        Args:
            base_url: Base URL of the EvalHub service
            auth_token: Optional authentication token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts (default: 3)
            verify_ssl: Whether to verify SSL certificates
            retry_initial_delay: Initial delay between retries in seconds (default: 1.0)
            retry_max_delay: Maximum delay between retries in seconds (default: 60.0)
            retry_backoff_factor: Multiplier for exponential backoff (default: 2.0)
            retry_randomization: Add random jitter to retry delays to prevent thundering herd (default: True)
        """
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"
        self.auth_token = auth_token
        self.max_retries = max_retries
        self.retry_initial_delay = retry_initial_delay
        self.retry_max_delay = retry_max_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_randomization = retry_randomization

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
        """Make HTTP request with exponential backoff retry logic.

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
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except httpx.TimeoutException as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(
                        f"Request to {url} timed out after {self.max_retries} retries"
                    )
                    raise
                delay = _calculate_retry_delay(
                    attempt,
                    self.retry_initial_delay,
                    self.retry_max_delay,
                    self.retry_backoff_factor,
                    self.retry_randomization,
                )
                logger.warning(
                    f"Request to {url} timed out, retrying in {delay:.2f}s "
                    f"({attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                last_exception = e
                # Don't retry client errors (4xx), only server errors (5xx)
                if e.response.status_code < 500 or attempt == self.max_retries:
                    raise
                delay = _calculate_retry_delay(
                    attempt,
                    self.retry_initial_delay,
                    self.retry_max_delay,
                    self.retry_backoff_factor,
                    self.retry_randomization,
                )
                logger.warning(
                    f"Server error {e.response.status_code} for {url}, "
                    f"retrying in {delay:.2f}s ({attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)

            except httpx.RequestError as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(
                        f"Connection error to {url} after {self.max_retries} retries: {e}"
                    )
                    raise
                delay = _calculate_retry_delay(
                    attempt,
                    self.retry_initial_delay,
                    self.retry_max_delay,
                    self.retry_backoff_factor,
                    self.retry_randomization,
                )
                logger.warning(
                    f"Connection error to {url}, retrying in {delay:.2f}s "
                    f"({attempt + 1}/{self.max_retries}): {e}"
                )
                await asyncio.sleep(delay)

        # This should never be reached, but mypy needs a return
        if last_exception:
            raise last_exception
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
    for synchronous operations with exponential backoff retry logic.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        auth_token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        retry_backoff_factor: float = 2.0,
        retry_randomization: bool = True,
    ):
        """Initialize the base sync client.

        Args:
            base_url: Base URL of the EvalHub service
            auth_token: Optional authentication token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts (default: 3)
            verify_ssl: Whether to verify SSL certificates
            retry_initial_delay: Initial delay between retries in seconds (default: 1.0)
            retry_max_delay: Maximum delay between retries in seconds (default: 60.0)
            retry_backoff_factor: Multiplier for exponential backoff (default: 2.0)
            retry_randomization: Add random jitter to retry delays to prevent thundering herd (default: True)
        """
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"
        self.auth_token = auth_token
        self.max_retries = max_retries
        self.retry_initial_delay = retry_initial_delay
        self.retry_max_delay = retry_max_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_randomization = retry_randomization

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
        """Make HTTP request with exponential backoff retry logic.

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
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except httpx.TimeoutException as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(
                        f"Request to {url} timed out after {self.max_retries} retries"
                    )
                    raise
                delay = _calculate_retry_delay(
                    attempt,
                    self.retry_initial_delay,
                    self.retry_max_delay,
                    self.retry_backoff_factor,
                    self.retry_randomization,
                )
                logger.warning(
                    f"Request to {url} timed out, retrying in {delay:.2f}s "
                    f"({attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)

            except httpx.HTTPStatusError as e:
                last_exception = e
                # Don't retry client errors (4xx), only server errors (5xx)
                if e.response.status_code < 500 or attempt == self.max_retries:
                    raise
                delay = _calculate_retry_delay(
                    attempt,
                    self.retry_initial_delay,
                    self.retry_max_delay,
                    self.retry_backoff_factor,
                    self.retry_randomization,
                )
                logger.warning(
                    f"Server error {e.response.status_code} for {url}, "
                    f"retrying in {delay:.2f}s ({attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)

            except httpx.RequestError as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(
                        f"Connection error to {url} after {self.max_retries} retries: {e}"
                    )
                    raise
                delay = _calculate_retry_delay(
                    attempt,
                    self.retry_initial_delay,
                    self.retry_max_delay,
                    self.retry_backoff_factor,
                    self.retry_randomization,
                )
                logger.warning(
                    f"Connection error to {url}, retrying in {delay:.2f}s "
                    f"({attempt + 1}/{self.max_retries}): {e}"
                )
                time.sleep(delay)

        # This should never be reached, but mypy needs a return
        if last_exception:
            raise last_exception
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
