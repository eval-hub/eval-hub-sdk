"""Integration tests for persist endpoint."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from evalhub.adapter.api.endpoints import create_adapter_api
from evalhub.adapter.models.framework import AdapterConfig, FrameworkAdapter
from evalhub.models.api import (
    BenchmarkInfo,
    EvaluationJob,
    EvaluationJobFilesLocation,
    EvaluationRequest,
    EvaluationResponse,
    FrameworkInfo,
    HealthResponse,
    JobStatus,
    ModelConfig,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient


class MockAdapter(FrameworkAdapter):
    """Minimal mock adapter for testing persist endpoint."""

    def __init__(self, config: AdapterConfig, test_output_dir: Path | None = None):
        super().__init__(config)
        self.test_output_dir = test_output_dir

    # Required abstract methods (minimal implementations)
    async def initialize(self) -> None:
        pass

    async def health_check(self) -> HealthResponse:
        return HealthResponse(
            status="healthy",
            framework_id=self.config.framework_id,
            version="1.0.0",
        )

    async def shutdown(self) -> None:
        pass

    async def get_framework_info(self) -> FrameworkInfo:
        raise NotImplementedError

    async def list_benchmarks(self) -> list[BenchmarkInfo]:
        raise NotImplementedError

    async def get_benchmark_info(self, benchmark_id: str) -> BenchmarkInfo | None:
        raise NotImplementedError

    async def submit_evaluation(self, request: EvaluationRequest) -> EvaluationJob:
        raise NotImplementedError

    async def get_job_status(self, job_id: str) -> EvaluationJob | None:
        return self._jobs.get(job_id)

    async def get_evaluation_results(self, job_id: str) -> EvaluationResponse | None:
        raise NotImplementedError

    async def cancel_job(self, job_id: str) -> bool:
        raise NotImplementedError

    # Only this method matters for persist tests
    async def job_files(self, job_id: str) -> EvaluationJobFilesLocation:
        """Override to return test output directory."""
        if self.test_output_dir and job_id in self._jobs:
            job = self._jobs[job_id]
            if job.status == JobStatus.COMPLETED:
                return EvaluationJobFilesLocation(
                    job_id=job_id,
                    path=str(self.test_output_dir),
                    metadata={"framework": "test", "benchmark": "test_benchmark"},
                )
        return EvaluationJobFilesLocation(job_id=job_id, path=None)


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    """Create test output directory with files."""
    output = tmp_path / "test_output"
    output.mkdir()
    (output / "results.json").write_text('{"score": 0.95}')
    (output / "metadata.txt").write_text("test metadata")
    return output


@pytest.fixture
def mock_adapter(test_output_dir: Path) -> MockAdapter:
    """Create mock adapter instance."""
    config = AdapterConfig(framework_id="test", adapter_name="Test Adapter")
    return MockAdapter(config, test_output_dir)


@pytest.fixture
def client(mock_adapter: MockAdapter) -> TestClient:
    """Create test client with mock adapter."""
    router = create_adapter_api(mock_adapter)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


class TestPersistEndpoint:
    """Tests for POST /evaluations/{job_id}/persist endpoint."""

    def test_persist_job_not_found(self, client: TestClient) -> None:
        """Test persist with non-existent job."""
        response = client.post(
            "/api/v1/evaluations/nonexistent/persist",
            json={"oci_ref": "ghcr.io/test/repo:latest"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_persist_job_not_completed(
        self, client: TestClient, mock_adapter: MockAdapter
    ) -> None:
        """Test persist with job not in completed state."""
        # Create pending job
        job = EvaluationJob(
            job_id="pending_job",
            status=JobStatus.PENDING,
            request=EvaluationRequest(
                benchmark_id="test_benchmark", model=ModelConfig(name="test_model")
            ),
            submitted_at=datetime.now(timezone.utc),
        )
        mock_adapter._jobs["pending_job"] = job

        response = client.post(
            "/api/v1/evaluations/pending_job/persist",
            json={"oci_ref": "ghcr.io/test/repo:latest"},
        )

        assert response.status_code == 409
        assert "not completed" in response.json()["detail"].lower()

    def test_persist_job_running(
        self, client: TestClient, mock_adapter: MockAdapter
    ) -> None:
        """Test persist with running job."""
        job = EvaluationJob(
            job_id="running_job",
            status=JobStatus.RUNNING,
            request=EvaluationRequest(
                benchmark_id="test_benchmark", model=ModelConfig(name="test_model")
            ),
            submitted_at=datetime.now(timezone.utc),
        )
        mock_adapter._jobs["running_job"] = job

        response = client.post(
            "/api/v1/evaluations/running_job/persist",
            json={"oci_ref": "ghcr.io/test/repo:latest"},
        )

        assert response.status_code == 409

    def test_persist_completed_job_no_files(
        self, client: TestClient, mock_adapter: MockAdapter
    ) -> None:
        """Test persist with completed job but no files."""
        # Create completed job (but adapter will return path=None)
        job = EvaluationJob(
            job_id="no_files_job",
            status=JobStatus.COMPLETED,
            request=EvaluationRequest(
                benchmark_id="test_benchmark", model=ModelConfig(name="test_model")
            ),
            submitted_at=datetime.now(timezone.utc),
        )
        mock_adapter._jobs["no_files_job"] = job
        # Set test_output_dir to None so job_files returns path=None
        mock_adapter.test_output_dir = None

        response = client.post(
            "/api/v1/evaluations/no_files_job/persist",
            json={"oci_ref": "ghcr.io/test/repo:latest"},
        )

        assert response.status_code == 404
        assert "no files to persist" in response.json()["detail"].lower()

    def test_persist_completed_job_success(
        self, client: TestClient, mock_adapter: MockAdapter, test_output_dir: Path
    ) -> None:
        """Test successful persist of completed job."""
        job = EvaluationJob(
            job_id="completed_job",
            status=JobStatus.COMPLETED,
            request=EvaluationRequest(
                benchmark_id="test_benchmark", model=ModelConfig(name="test_model")
            ),
            submitted_at=datetime.now(timezone.utc),
        )
        mock_adapter._jobs["completed_job"] = job

        response = client.post(
            "/api/v1/evaluations/completed_job/persist",
            json={"oci_ref": "ghcr.io/test/repo:latest"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["job_id"] == "completed_job"
        assert data["digest"].startswith("sha256:")
        assert data["oci_ref"].startswith("ghcr.io/test/repo:latest@sha256:")
        assert data["files_count"] == 2  # results.json + metadata.txt
        assert data["metadata"]["placeholder"] is True

    def test_persist_invalid_request_missing_oci_ref(
        self, client: TestClient, mock_adapter: MockAdapter
    ) -> None:
        """Test persist with missing oci_ref in request body."""
        job = EvaluationJob(
            job_id="test_job",
            status=JobStatus.COMPLETED,
            request=EvaluationRequest(
                benchmark_id="test_benchmark", model=ModelConfig(name="test_model")
            ),
            submitted_at=datetime.now(timezone.utc),
        )
        mock_adapter._jobs["test_job"] = job

        response = client.post(
            "/api/v1/evaluations/test_job/persist",
            json={},  # Missing oci_ref
        )

        assert response.status_code == 422  # Validation error
