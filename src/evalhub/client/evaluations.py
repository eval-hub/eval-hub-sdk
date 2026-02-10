"""Client for EvalHub evaluation operations."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from ..models import (
    EvaluationJob,
    EvaluationRequest,
    JobsList,
    JobStatus,
)
from .base import (
    BaseAsyncClient,
    BaseSyncClient,
    JobCanNotBeCancelledError,
    JobNotFoundError,
)

logger = logging.getLogger(__name__)


class AsyncEvaluationsClient(BaseAsyncClient):
    """Asynchronous client for evaluation job operations.

    Inherits from BaseAsyncClient and provides methods for submitting,
    monitoring, and managing evaluation jobs.
    """

    async def submit(self, request: EvaluationRequest) -> EvaluationJob:
        """Submit an evaluation job.

        Args:
            request: The evaluation request

        Returns:
            EvaluationJob: The submitted job

        Raises:
            httpx.HTTPError: If request fails or is invalid
        """
        response = await self._request(
            "POST", "/evaluations/jobs", json=request.model_dump()
        )
        return EvaluationJob(**response.json())

    async def get_job(self, job_id: str) -> EvaluationJob:
        """Get the status of an evaluation job.

        Args:
            job_id: The job identifier

        Returns:
            EvaluationJob: Current job status

        Raises:
            httpx.HTTPError: If job not found or request fails
        """
        response = await self._request_get(f"/evaluations/jobs/{job_id}")
        return EvaluationJob(**response.json())

    async def cancel(self, job_id: str) -> bool:
        """Cancel an evaluation job.

        Args:
            job_id: The job identifier

        Returns:
            bool: True if job was successfully cancelled

        Raises:
            JobNotFoundError: If the job does not exist or was already deleted (HTTP 404)
            JobCanNotBeCancelledError: If the job cannot be cancelled, e.g. already
                completed, failed, or cancelled (HTTP 400)
            httpx.HTTPError: If request fails for other reasons
        """
        try:
            await self._request_delete(f"/evaluations/jobs/{job_id}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise JobNotFoundError(job_id, cause=e) from e
            if e.response.status_code in [400, 409]:
                reason = None
                try:
                    body = e.response.json()
                    reason = body.get("message")
                except Exception:
                    pass
                raise JobCanNotBeCancelledError(job_id, reason=reason, cause=e) from e
            raise

    async def list(
        self, status: JobStatus | None = None, limit: int | None = None
    ) -> list[EvaluationJob]:
        """List evaluation jobs.

        Args:
            status: Filter by job status (optional)
            limit: Maximum number of jobs to return (optional)

        Returns:
            list[EvaluationJob]: List of jobs

        Raises:
            httpx.HTTPError: If request fails
        """
        params = {}
        if status:
            params["status_filter"] = status.value
        if limit:
            params["limit"] = str(limit)

        response = await self._request_get("/evaluations/jobs", params=params)
        data = response.json()
        jobs_list = JobsList(**data)
        return jobs_list.items

    async def wait_for_completion(
        self, job_id: str, timeout: float | None = None, poll_interval: float = 5.0
    ) -> EvaluationJob:
        """Wait for an evaluation job to complete.

        Args:
            job_id: The job identifier
            timeout: Maximum time to wait in seconds (optional)
            poll_interval: Polling interval in seconds

        Returns:
            EvaluationJob: Final job status

        Raises:
            TimeoutError: If job doesn't complete within timeout
            httpx.HTTPError: If request fails
        """
        start_time = time.time()

        while True:
            job = await self.get_job(job_id)

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


class SyncEvaluationsClient(BaseSyncClient):
    """Synchronous client for evaluation job operations.

    Inherits from BaseSyncClient and provides methods for submitting,
    monitoring, and managing evaluation jobs.
    """

    def submit(self, request: EvaluationRequest) -> EvaluationJob:
        """Submit an evaluation job.

        Args:
            request: The evaluation request

        Returns:
            EvaluationJob: The submitted job

        Raises:
            httpx.HTTPError: If request fails or is invalid
        """
        response = self._request_post("/evaluations/jobs", json=request.model_dump())
        return EvaluationJob(**response.json())

    def get_job(self, job_id: str) -> EvaluationJob:
        """Get the status of an evaluation job.

        Args:
            job_id: The job identifier

        Returns:
            EvaluationJob: Current job status

        Raises:
            httpx.HTTPError: If job not found or request fails
        """
        response = self._request_get(f"/evaluations/jobs/{job_id}")
        return EvaluationJob(**response.json())

    def cancel(self, job_id: str) -> bool:
        """Cancel an evaluation job.

        Args:
            job_id: The job identifier

        Returns:
            bool: True if job was successfully cancelled

        Raises:
            JobNotFoundError: If the job does not exist or was already deleted (HTTP 404)
            JobCanNotBeCancelledError: If the job cannot be cancelled, e.g. already
                completed, failed, or cancelled (HTTP 400)
            httpx.HTTPError: If request fails for other reasons
        """
        try:
            self._request_delete(f"/evaluations/jobs/{job_id}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise JobNotFoundError(job_id, cause=e) from e
            if e.response.status_code in [400, 409]:
                reason = None
                try:
                    body = e.response.json()
                    reason = body.get("message")
                except Exception:
                    pass
                raise JobCanNotBeCancelledError(job_id, reason=reason, cause=e) from e
            raise

    def list(
        self, status: JobStatus | None = None, limit: int | None = None
    ) -> list[EvaluationJob]:
        """List evaluation jobs.

        Args:
            status: Filter by job status (optional)
            limit: Maximum number of jobs to return (optional)

        Returns:
            list[EvaluationJob]: List of jobs

        Raises:
            httpx.HTTPError: If request fails
        """
        params = {}
        if status:
            params["status_filter"] = status.value
        if limit:
            params["limit"] = str(limit)

        response = self._request_get("/evaluations/jobs", params=params)
        data = response.json()
        jobs_list = JobsList(**data)
        return jobs_list.items

    def wait_for_completion(
        self, job_id: str, timeout: float | None = None, poll_interval: float = 5.0
    ) -> EvaluationJob:
        """Wait for an evaluation job to complete.

        Args:
            job_id: The job identifier
            timeout: Maximum time to wait in seconds (optional)
            poll_interval: Polling interval in seconds

        Returns:
            EvaluationJob: Final job status

        Raises:
            TimeoutError: If job doesn't complete within timeout
            httpx.HTTPError: If request fails
        """
        start_time = time.time()

        while True:
            job = self.get_job(job_id)

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

            time.sleep(poll_interval)
