"""Client for communicating with framework adapters via the standard SDK API."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

# typing imports removed - using PEP 604 union syntax
import httpx

from ...models.api import (
    BenchmarkInfo,
    EvaluationJob,
    EvaluationRequest,
    EvaluationResponse,
    FrameworkInfo,
    HealthResponse,
    JobStatus,
    OCICoordinate,
    PersistResponse,
)

logger = logging.getLogger(__name__)


class ClientError(Exception):
    """Base exception for client errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class AdapterClient:
    """Client for communicating with framework adapters.

    This client provides a standardized way for EvalHub to communicate
    with any framework adapter that implements the SDK API.
    """

    def __init__(self, base_url: str, timeout: float = 30.0, max_retries: int = 3):
        """Initialize the adapter client.

        Args:
            base_url: Base URL of the framework adapter (e.g., "http://adapter:8080")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        )

        self.max_retries = max_retries

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "AdapterClient":
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

    # Health and Info endpoints

    async def health_check(self) -> HealthResponse:
        """Check the health of the framework adapter.

        Returns:
            HealthResponse: Current health status

        Raises:
            httpx.HTTPError: If health check fails
        """
        response = await self._request("GET", "/health")
        return HealthResponse(**response.json())

    async def get_framework_info(self) -> FrameworkInfo:
        """Get information about the framework adapter.

        Returns:
            FrameworkInfo: Framework capabilities and metadata

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._request("GET", "/info")
        return FrameworkInfo(**response.json())

    # Benchmark endpoints

    async def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List all available benchmarks.

        Returns:
            List[BenchmarkInfo]: Available benchmarks

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._request("GET", "/benchmarks")
        return [BenchmarkInfo(**benchmark) for benchmark in response.json()]

    async def get_benchmark_info(self, benchmark_id: str) -> BenchmarkInfo:
        """Get detailed information about a specific benchmark.

        Args:
            benchmark_id: The benchmark identifier

        Returns:
            BenchmarkInfo: Benchmark information

        Raises:
            httpx.HTTPError: If benchmark not found or request fails
        """
        response = await self._request("GET", f"/benchmarks/{benchmark_id}")
        return BenchmarkInfo(**response.json())

    # Evaluation endpoints

    async def submit_evaluation(self, request: EvaluationRequest) -> EvaluationJob:
        """Submit an evaluation job.

        Args:
            request: The evaluation request

        Returns:
            EvaluationJob: The submitted job

        Raises:
            httpx.HTTPError: If request fails or is invalid
        """
        response = await self._request(
            "POST", "/evaluations", json=request.model_dump()
        )
        return EvaluationJob(**response.json())

    async def get_job_status(self, job_id: str) -> EvaluationJob:
        """Get the status of an evaluation job.

        Args:
            job_id: The job identifier

        Returns:
            EvaluationJob: Current job status

        Raises:
            httpx.HTTPError: If job not found or request fails
        """
        response = await self._request("GET", f"/evaluations/{job_id}")
        return EvaluationJob(**response.json())

    async def get_evaluation_results(self, job_id: str) -> EvaluationResponse:
        """Get the results of a completed evaluation.

        Args:
            job_id: The job identifier

        Returns:
            EvaluationResponse: Evaluation results

        Raises:
            httpx.HTTPError: If results not available or request fails
        """
        response = await self._request("GET", f"/evaluations/{job_id}/results")
        return EvaluationResponse(**response.json())

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an evaluation job.

        Args:
            job_id: The job identifier

        Returns:
            bool: True if job was cancelled

        Raises:
            httpx.HTTPError: If request fails
        """
        try:
            await self._request("DELETE", f"/evaluations/{job_id}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False  # Job not found
            elif e.response.status_code == 409:
                return False  # Job cannot be cancelled
            else:
                raise

    async def list_jobs(
        self, status: JobStatus | None = None, limit: int | None = None
    ) -> list[EvaluationJob]:
        """List evaluation jobs.

        Args:
            status: Filter by job status
            limit: Maximum number of jobs to return

        Returns:
            List[EvaluationJob]: List of jobs

        Raises:
            httpx.HTTPError: If request fails
        """
        params = {}
        if status:
            params["status"] = status.value
        if limit:
            params["limit"] = str(limit)

        response = await self._request("GET", "/evaluations", params=params)
        return [EvaluationJob(**job) for job in response.json()]

    async def stream_job_updates(
        self, job_id: str
    ) -> AsyncGenerator[EvaluationJob, None]:
        """Stream real-time updates for an evaluation job.

        Args:
            job_id: The job identifier

        Yields:
            EvaluationJob: Updated job status

        Raises:
            httpx.HTTPError: If streaming fails
        """
        url = f"{self.api_base}/evaluations/{job_id}/stream"

        try:
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data.strip():
                            try:
                                import json

                                job_data = json.loads(data)
                                yield EvaluationJob(**job_data)
                            except Exception as e:
                                logger.warning(f"Failed to parse streaming data: {e}")

        except httpx.HTTPError:
            # Fall back to polling if streaming is not supported
            logger.info(
                f"Streaming not available for {job_id}, falling back to polling"
            )
            async for job_update in self._poll_job_updates(job_id):
                yield job_update

    async def _poll_job_updates(
        self, job_id: str, interval: float = 2.0
    ) -> AsyncGenerator[EvaluationJob, None]:
        """Poll for job updates (fallback for streaming).

        Args:
            job_id: The job identifier
            interval: Polling interval in seconds

        Yields:
            EvaluationJob: Updated job status
        """
        import asyncio

        while True:
            try:
                job = await self.get_job_status(job_id)
                yield job

                if job.status in [
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ]:
                    break

                await asyncio.sleep(interval)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    break  # Job not found
                else:
                    raise

    async def wait_for_completion(
        self, job_id: str, timeout: float | None = None, poll_interval: float = 5.0
    ) -> EvaluationJob:
        """Wait for an evaluation job to complete.

        Args:
            job_id: The job identifier
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds

        Returns:
            EvaluationJob: Final job status

        Raises:
            TimeoutError: If job doesn't complete within timeout
            httpx.HTTPError: If request fails
        """
        import asyncio
        import time

        start_time = time.time()

        while True:
            job = await self.get_job_status(job_id)

            if job.status in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                return job

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds"
                )

            await asyncio.sleep(poll_interval)

    async def persist_job_files(
        self, job_id: str, coordinate: OCICoordinate
    ) -> PersistResponse:
        """Persist job files as OCI artifact.

        Args:
            job_id: The job identifier
            coordinate: OCI coordinates (reference and optional subject)

        Returns:
            PersistResponse: Persistence status and artifact information

        Raises:
            httpx.HTTPError: If request fails
            ClientError: If job not found or has no files to persist
        """
        try:
            response = await self._request(
                "POST", f"/evaluations/{job_id}/persist", json=coordinate.model_dump()
            )
            return PersistResponse(**response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ClientError(
                    f"Job {job_id} not found or has no files to persist", cause=e
                )
            raise
