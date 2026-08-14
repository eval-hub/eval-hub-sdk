"""Job resource for EvalHub client."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator

import httpx

from ...models import (
    EvaluationJob,
    JobsList,
    JobStatus,
    JobSubmissionRequest,
)
from ..base import (
    BaseAsyncClient,
    BaseSyncClient,
    JobCanNotBeCancelledError,
    JobNotFoundError,
)
from ..job_logs import (
    JobLogOptions,
    JobLogUpdate,
    fetch_job_logs,
    fetch_job_logs_sync,
    is_terminal_job,
    log_delta,
)

logger = logging.getLogger(__name__)


class AsyncJobsResource:
    """Asynchronous resource for evaluation job operations."""

    def __init__(self, client: BaseAsyncClient):
        self._client = client

    async def submit(
        self, request: JobSubmissionRequest, *, tenant: str | None = None
    ) -> EvaluationJob:
        """Submit an evaluation job.

        Args:
            request: The job submission request
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            EvaluationJob: The submitted job

        Raises:
            httpx.HTTPError: If request fails or is invalid
        """
        response = await self._client._request_post(
            "/evaluations/jobs",
            json=request.model_dump(exclude_none=True, context={"for_submission": True}),
            tenant=tenant,
        )
        return EvaluationJob(**response.json())

    async def get(self, job_id: str, *, tenant: str | None = None) -> EvaluationJob:
        """Get the status of an evaluation job.

        Args:
            job_id: The job identifier
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            EvaluationJob: Current job status

        Raises:
            httpx.HTTPError: If job not found or request fails
        """
        response = await self._client._request_get(
            f"/evaluations/jobs/{job_id}", tenant=tenant
        )
        return EvaluationJob(**response.json())

    async def cancel(
        self, job_id: str, hard_delete: bool = False, *, tenant: str | None = None
    ) -> bool:
        """Cancel an evaluation job.

        Args:
            job_id: The job identifier
            hard_delete: If True, permanently delete the job instead of just cancelling it
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            bool: True if job was successfully cancelled

        Raises:
            JobNotFoundError: If the job does not exist or was already deleted (HTTP 404)
            JobCanNotBeCancelledError: If the job cannot be cancelled, e.g. already
                completed, failed, or cancelled (HTTP 400)
            httpx.HTTPError: If request fails for other reasons
        """
        try:
            params = {}
            if hard_delete:
                params["hard_delete"] = "true"
            await self._client._request_delete(
                f"/evaluations/jobs/{job_id}", params=params, tenant=tenant
            )
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
        self,
        status: JobStatus | None = None,
        limit: int | None = None,
        *,
        tenant: str | None = None,
    ) -> list[EvaluationJob]:
        """List evaluation jobs.

        Args:
            status: Filter by job status (optional)
            limit: Maximum number of jobs to return (optional)
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            list[EvaluationJob]: List of jobs

        Raises:
            httpx.HTTPError: If request fails
        """
        params = {}
        if status:
            params["status"] = status.value
        if limit:
            params["limit"] = str(limit)

        response = await self._client._request_get(
            "/evaluations/jobs", params=params, tenant=tenant
        )
        data = response.json()
        jobs_list = JobsList(**data)
        return jobs_list.items

    async def wait_for_completion(
        self,
        job_id: str,
        timeout: float | None = None,
        poll_interval: float = 5.0,
        *,
        tenant: str | None = None,
    ) -> EvaluationJob:
        """Wait for an evaluation job to complete.

        Args:
            job_id: The job identifier
            timeout: Maximum time to wait in seconds (optional)
            poll_interval: Polling interval in seconds
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            EvaluationJob: Final job status

        Raises:
            TimeoutError: If job doesn't complete within timeout
            httpx.HTTPError: If request fails
        """
        start_time = time.time()

        while True:
            job = await self.get(job_id, tenant=tenant)

            # Check if job is in a terminal state (also considers
            # benchmark-level completion when the server hasn't promoted it)
            if is_terminal_job(job):
                return job

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds"
                )

            await asyncio.sleep(poll_interval)

    async def get_logs(
        self,
        job_id: str,
        *,
        benchmark_index: int | None = None,
        options: JobLogOptions | None = None,
        tenant: str | None = None,
    ) -> str:
        """Fetch evaluation job logs as plain text.

        Args:
            job_id: The job identifier
            benchmark_index: When set, fetch logs for a single benchmark index
            options: Log query options (tail_lines, timestamps, etc.)
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            str: Plain-text job logs

        Raises:
            httpx.HTTPError: If request fails
        """
        return await fetch_job_logs(
            self._client,
            job_id,
            benchmark_index=benchmark_index,
            options=options,
            tenant=tenant,
        )

    async def watch_logs(
        self,
        job_id: str,
        *,
        benchmark_index: int | None = None,
        options: JobLogOptions | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        tenant: str | None = None,
    ) -> AsyncIterator[JobLogUpdate]:
        """Stream job logs while polling status until the job completes.

        The EvalHub server returns on-demand log snapshots, so this method
        polls ``get_logs`` and ``get`` until the job reaches a terminal state.
        Each yielded :class:`JobLogUpdate` contains only new log content since
        the previous poll (which may be empty) and the current job status.

        Args:
            job_id: The job identifier
            benchmark_index: When set, watch logs for a single benchmark index
            options: Log query options (tail_lines, timestamps, etc.)
            poll_interval: Seconds between status and log polls
            timeout: Maximum time to watch in seconds (optional)
            tenant: Tenant override for this request (default: client-level tenant)

        Yields:
            JobLogUpdate: Incremental log content and current job status

        Raises:
            TimeoutError: If the job does not complete within timeout
            httpx.HTTPError: If request fails
        """
        start_time = time.time()
        seen = ""
        log_options = options or JobLogOptions()

        while True:
            job = await self.get(job_id, tenant=tenant)
            logs = await self.get_logs(
                job_id,
                benchmark_index=benchmark_index,
                options=log_options,
                tenant=tenant,
            )
            delta = log_delta(seen, logs)
            if logs:
                seen = logs
            yield JobLogUpdate(logs=delta, job=job)

            if is_terminal_job(job):
                return

            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds "
                    "while watching logs"
                )

            await asyncio.sleep(poll_interval)


class SyncJobsResource:
    """Synchronous resource for evaluation job operations."""

    def __init__(self, client: BaseSyncClient):
        self._client = client

    def submit(
        self, request: JobSubmissionRequest, *, tenant: str | None = None
    ) -> EvaluationJob:
        """Submit an evaluation job.

        Args:
            request: The job submission request
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            EvaluationJob: The submitted job

        Raises:
            httpx.HTTPError: If request fails or is invalid
        """
        response = self._client._request_post(
            "/evaluations/jobs",
            json=request.model_dump(exclude_none=True, context={"for_submission": True}),
            tenant=tenant,
        )
        return EvaluationJob(**response.json())

    def get(self, job_id: str, *, tenant: str | None = None) -> EvaluationJob:
        """Get the status of an evaluation job.

        Args:
            job_id: The job identifier
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            EvaluationJob: Current job status

        Raises:
            httpx.HTTPError: If job not found or request fails
        """
        response = self._client._request_get(
            f"/evaluations/jobs/{job_id}", tenant=tenant
        )
        return EvaluationJob(**response.json())

    def cancel(
        self, job_id: str, hard_delete: bool = False, *, tenant: str | None = None
    ) -> bool:
        """Cancel an evaluation job.

        Args:
            job_id: The job identifier
            hard_delete: If True, permanently delete the job instead of just cancelling it
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            bool: True if job was successfully cancelled

        Raises:
            JobNotFoundError: If the job does not exist or was already deleted (HTTP 404)
            JobCanNotBeCancelledError: If the job cannot be cancelled, e.g. already
                completed, failed, or cancelled (HTTP 400)
            httpx.HTTPError: If request fails for other reasons
        """
        try:
            params = {}
            if hard_delete:
                params["hard_delete"] = "true"
            self._client._request_delete(
                f"/evaluations/jobs/{job_id}", params=params, tenant=tenant
            )
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
        self,
        status: JobStatus | None = None,
        limit: int | None = None,
        *,
        tenant: str | None = None,
    ) -> list[EvaluationJob]:
        """List evaluation jobs.

        Args:
            status: Filter by job status (optional)
            limit: Maximum number of jobs to return (optional)
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            list[EvaluationJob]: List of jobs

        Raises:
            httpx.HTTPError: If request fails
        """
        params = {}
        if status:
            params["status"] = status.value
        if limit:
            params["limit"] = str(limit)

        response = self._client._request_get(
            "/evaluations/jobs", params=params, tenant=tenant
        )
        data = response.json()
        jobs_list = JobsList(**data)
        return jobs_list.items

    def wait_for_completion(
        self,
        job_id: str,
        timeout: float | None = None,
        poll_interval: float = 5.0,
        *,
        tenant: str | None = None,
    ) -> EvaluationJob:
        """Wait for an evaluation job to complete.

        Args:
            job_id: The job identifier
            timeout: Maximum time to wait in seconds (optional)
            poll_interval: Polling interval in seconds
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            EvaluationJob: Final job status

        Raises:
            TimeoutError: If job doesn't complete within timeout
            httpx.HTTPError: If request fails
        """
        start_time = time.time()

        while True:
            job = self.get(job_id, tenant=tenant)

            # Check if job is in a terminal state (also considers
            # benchmark-level completion when the server hasn't promoted it)
            if is_terminal_job(job):
                return job

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds"
                )

            time.sleep(poll_interval)

    def get_logs(
        self,
        job_id: str,
        *,
        benchmark_index: int | None = None,
        options: JobLogOptions | None = None,
        tenant: str | None = None,
    ) -> str:
        """Fetch evaluation job logs as plain text.

        Args:
            job_id: The job identifier
            benchmark_index: When set, fetch logs for a single benchmark index
            options: Log query options (tail_lines, timestamps, etc.)
            tenant: Tenant override for this request (default: client-level tenant)

        Returns:
            str: Plain-text job logs

        Raises:
            httpx.HTTPError: If request fails
        """
        return fetch_job_logs_sync(
            self._client,
            job_id,
            benchmark_index=benchmark_index,
            options=options,
            tenant=tenant,
        )

    def watch_logs(
        self,
        job_id: str,
        *,
        benchmark_index: int | None = None,
        options: JobLogOptions | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        tenant: str | None = None,
    ) -> Iterator[JobLogUpdate]:
        """Stream job logs while polling status until the job completes.

        The EvalHub server returns on-demand log snapshots, so this method
        polls ``get_logs`` and ``get`` until the job reaches a terminal state.
        Each yielded :class:`JobLogUpdate` contains only new log content since
        the previous poll (which may be empty) and the current job status.

        Args:
            job_id: The job identifier
            benchmark_index: When set, watch logs for a single benchmark index
            options: Log query options (tail_lines, timestamps, etc.)
            poll_interval: Seconds between status and log polls
            timeout: Maximum time to watch in seconds (optional)
            tenant: Tenant override for this request (default: client-level tenant)

        Yields:
            JobLogUpdate: Incremental log content and current job status

        Raises:
            TimeoutError: If the job does not complete within timeout
            httpx.HTTPError: If request fails
        """
        start_time = time.time()
        seen = ""
        log_options = options or JobLogOptions()

        while True:
            job = self.get(job_id, tenant=tenant)
            logs = self.get_logs(
                job_id,
                benchmark_index=benchmark_index,
                options=log_options,
                tenant=tenant,
            )
            delta = log_delta(seen, logs)
            if logs:
                seen = logs
            yield JobLogUpdate(logs=delta, job=job)

            if is_terminal_job(job):
                return

            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds "
                    "while watching logs"
                )

            time.sleep(poll_interval)
