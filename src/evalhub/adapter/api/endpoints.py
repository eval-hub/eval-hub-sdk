"""Standard API endpoints for framework adapters."""

import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

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
from ..models.framework import FrameworkAdapter

logger = logging.getLogger(__name__)


def create_adapter_api(adapter: FrameworkAdapter) -> APIRouter:
    """Create FastAPI router with standard endpoints for a framework adapter.

    Args:
        adapter: The framework adapter instance

    Returns:
        APIRouter: Router with standard endpoints
    """
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, tags=["Health"])
    @router.options("/health", tags=["Health"])
    async def health_check() -> HealthResponse:
        """Check the health of the framework adapter."""
        try:
            return await adapter.health_check()
        except Exception as e:
            logger.exception("Health check failed")
            raise HTTPException(
                status_code=503, detail=f"Health check failed: {str(e)}"
            )

    @router.get("/info", response_model=FrameworkInfo, tags=["Info"])
    async def get_framework_info() -> FrameworkInfo:
        """Get information about the framework adapter."""
        try:
            return await adapter.get_framework_info()
        except Exception as e:
            logger.exception("Failed to get framework info")
            raise HTTPException(
                status_code=500, detail=f"Failed to get framework info: {str(e)}"
            )

    @router.get("/benchmarks", response_model=list[BenchmarkInfo], tags=["Benchmarks"])
    async def list_benchmarks() -> list[BenchmarkInfo]:
        """List all available benchmarks."""
        try:
            return await adapter.list_benchmarks()
        except Exception as e:
            logger.exception("Failed to list benchmarks")
            raise HTTPException(
                status_code=500, detail=f"Failed to list benchmarks: {str(e)}"
            )

    @router.get(
        "/benchmarks/{benchmark_id}", response_model=BenchmarkInfo, tags=["Benchmarks"]
    )
    async def get_benchmark_info(benchmark_id: str) -> BenchmarkInfo:
        """Get detailed information about a specific benchmark."""
        try:
            benchmark_info = await adapter.get_benchmark_info(benchmark_id)
            if not benchmark_info:
                raise HTTPException(
                    status_code=404, detail=f"Benchmark '{benchmark_id}' not found"
                )
            return benchmark_info
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to get benchmark info for {benchmark_id}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get benchmark info: {str(e)}"
            )

    @router.post(
        "/evaluations",
        response_model=EvaluationJob,
        status_code=201,
        tags=["Evaluations"],
    )
    async def submit_evaluation(
        request: EvaluationRequest, background_tasks: BackgroundTasks
    ) -> EvaluationJob:
        """Submit an evaluation job."""
        try:
            # Validate the request
            benchmark_info = await adapter.get_benchmark_info(request.benchmark_id)
            if not benchmark_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Benchmark '{request.benchmark_id}' not found",
                )

            # Submit the evaluation
            job = await adapter.submit_evaluation(request)

            logger.info(
                f"Submitted evaluation job {job.id} for benchmark {request.benchmark_id}"
            )
            return job

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid evaluation request: {str(e)}"
            )
        except Exception as e:
            logger.exception("Failed to submit evaluation")
            raise HTTPException(
                status_code=500, detail=f"Failed to submit evaluation: {str(e)}"
            )

    @router.get(
        "/evaluations/{job_id}", response_model=EvaluationJob, tags=["Evaluations"]
    )
    async def get_job_status(job_id: str) -> EvaluationJob:
        """Get the status of an evaluation job."""
        try:
            job = await adapter.get_job_status(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
            return job
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to get job status for {job_id}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get job status: {str(e)}"
            )

    @router.get(
        "/evaluations/{job_id}/results",
        response_model=EvaluationResponse,
        tags=["Evaluations"],
    )
    async def get_evaluation_results(job_id: str) -> EvaluationResponse:
        """Get the results of a completed evaluation."""
        try:
            # First check if job exists
            job = await adapter.get_job_status(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

            # Check if results are available
            if job.status == JobStatus.PENDING:
                raise HTTPException(
                    status_code=202, detail=f"Job '{job_id}' is still pending"
                )
            elif job.status == JobStatus.RUNNING:
                raise HTTPException(
                    status_code=202, detail=f"Job '{job_id}' is still running"
                )
            elif job.status == JobStatus.FAILED:
                raise HTTPException(
                    status_code=422,
                    detail=f"Job '{job_id}' failed: {job.error_message}",
                )
            elif job.status == JobStatus.CANCELLED:
                raise HTTPException(
                    status_code=410, detail=f"Job '{job_id}' was cancelled"
                )

            # Get results
            results = await adapter.get_evaluation_results(job_id)
            if not results:
                raise HTTPException(
                    status_code=404, detail=f"Results for job '{job_id}' not found"
                )

            return results

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to get evaluation results for {job_id}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get evaluation results: {str(e)}"
            )

    @router.delete("/evaluations/{job_id}", tags=["Evaluations"])
    async def cancel_job(job_id: str) -> dict[str, bool | str]:
        """Cancel an evaluation job."""
        try:
            success = await adapter.cancel_job(job_id)
            if not success:
                # Check if job exists
                job = await adapter.get_job_status(job_id)
                if not job:
                    raise HTTPException(
                        status_code=404, detail=f"Job '{job_id}' not found"
                    )
                else:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Job '{job_id}' cannot be cancelled (status: {job.status})",
                    )

            return {
                "success": True,
                "message": f"Job '{job_id}' cancelled successfully",
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to cancel job {job_id}")
            raise HTTPException(
                status_code=500, detail=f"Failed to cancel job: {str(e)}"
            )

    @router.get(
        "/evaluations", response_model=list[EvaluationJob], tags=["Evaluations"]
    )
    async def list_jobs(
        status: JobStatus | None = None, limit: int | None = None
    ) -> list[EvaluationJob]:
        """List evaluation jobs, optionally filtered by status."""
        try:
            jobs = await adapter.list_active_jobs()

            # Filter by status if specified
            if status:
                jobs = [job for job in jobs if job.status == status]

            # Apply limit if specified
            if limit and limit > 0:
                jobs = jobs[:limit]

            return jobs

        except Exception as e:
            logger.exception("Failed to list jobs")
            raise HTTPException(
                status_code=500, detail=f"Failed to list jobs: {str(e)}"
            )

    @router.get(
        "/evaluations/{job_id}/stream",
        response_class=StreamingResponse,
        tags=["Evaluations"],
    )
    async def stream_job_updates(job_id: str) -> StreamingResponse:
        """Stream real-time updates for an evaluation job."""
        try:
            # Check if job exists
            job = await adapter.get_job_status(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

            async def event_stream() -> AsyncGenerator[str, None]:
                """Generate Server-Sent Events for job updates."""
                async for updated_job in adapter.stream_job_updates(job_id):
                    # Format as Server-Sent Event
                    yield f"data: {updated_job.model_dump_json()}\n\n"

                    # Stop streaming when job is complete
                    if updated_job.status in [
                        JobStatus.COMPLETED,
                        JobStatus.FAILED,
                        JobStatus.CANCELLED,
                    ]:
                        break

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to stream job updates for {job_id}")
            raise HTTPException(
                status_code=500, detail=f"Failed to stream job updates: {str(e)}"
            )

    @router.post(
        "/evaluations/{job_id}/persist",
        response_model=PersistResponse,
        tags=["Evaluations"],
    )
    async def persist_job_files(
        job_id: str, coordinate: OCICoordinate
    ) -> PersistResponse:
        """Persist job files as OCI artifact.

        Manually trigger OCI artifact persistence for completed job files.

        Args:
            job_id: The job identifier
            coordinate: OCI coordinates (reference and optional subject)

        Returns:
            PersistResponse: Persistence status and artifact information

        Raises:
            HTTPException: If job not found, not completed, or has no files to persist
        """
        try:
            result = await adapter.persist_job_files_oci(job_id, coordinate)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Job {job_id} has no files to persist",
                )
            return result
        except ValueError as e:
            error_msg = str(e)
            # Job not found -> 404, job not completed -> 409
            if "not found" in error_msg:
                raise HTTPException(status_code=404, detail=error_msg)
            else:
                raise HTTPException(status_code=409, detail=error_msg)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to persist files for job {job_id}")
            raise HTTPException(
                status_code=500, detail=f"Failed to persist files: {str(e)}"
            )

    return router
