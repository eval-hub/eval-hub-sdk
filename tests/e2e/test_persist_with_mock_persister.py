"""E2E test for persist endpoint with custom mock persister."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import oras.client  # type: ignore
import pytest
from evalhub.adapter.api.endpoints import create_adapter_api
from evalhub.adapter.models.framework import AdapterConfig, FrameworkAdapter
from evalhub.adapter.oci.persister import Persister
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
    OCICoordinate,
    PersistResponse,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestOrasPersister:
    """Test scope persister that uses oras-py internally (for now) and returns a controlled response."""

    async def persist(
        self,
        files_location: EvaluationJobFilesLocation,
        coordinate: OCICoordinate,
        job: EvaluationJob,
    ) -> PersistResponse:
        """Persist files using ORAS to upload OCI artifact."""
        if not files_location.path:
            raise ValueError("No files to persist")

        source_path = Path(files_location.path)
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

        registry_host = coordinate.oci_ref.split("/")[0]
        client = oras.client.OrasClient(hostname=registry_host, insecure=True)

        # Count files
        files_count = 0
        if source_path.is_file():
            files_count = 1
        else:
            files_count = sum(1 for f in source_path.rglob("*") if f.is_file())

        # Push using ORAS
        digest = None
        try:
            response = client.push(
                files=[str(source_path)],
                target=coordinate.oci_ref,
                disable_path_validation=True,
            )
            if "Docker-Content-Digest" in response.headers:
                digest = response.headers["Docker-Content-Digest"]
            else:
                # Fallback digest if not provided during testing
                digest = f"sha256:{'0' * 64}"
        except Exception as e:
            print(e)
            # If push fails (e.g., no registry running), use fallback digest; this allows E2E tests to work without a running registry for now
            digest = f"sha256:{'0' * 64}"

        return PersistResponse(
            job_id=job.job_id,
            oci_ref=f"{coordinate.oci_ref}@{digest}",
            digest=digest,
            files_count=files_count,
            metadata={
                "test_oras": True,
            },
        )


class TestAdapter(FrameworkAdapter):
    """Test adapter that uses custom persister."""

    def __init__(
        self,
        config: AdapterConfig,
        test_output_dir: Path | None = None,
        custom_persister: Persister | None = None,
    ):
        super().__init__(config)
        self.test_output_dir = test_output_dir
        self.custom_persister = custom_persister

    def _get_persister(self) -> Persister:
        """Override to return custom persister for testing."""
        if self.custom_persister:
            return self.custom_persister
        raise RuntimeError(
            "This test adapter should be used only to override default implementation."
        )

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
    """Create test output directory with sample files."""
    output = tmp_path / "test_output"
    output.mkdir()
    (output / "results.json").write_text('{"accuracy": 0.95, "f1_score": 0.92}')
    (output / "metrics.csv").write_text("metric,value\\naccuracy,0.95\\nf1,0.92")
    (output / "summary.txt").write_text("Test completed successfully")
    return output


@pytest.fixture
def oras_persister() -> TestOrasPersister:
    """Create ORAS persister instance."""
    return TestOrasPersister()


@pytest.fixture
def test_adapter(
    test_output_dir: Path, oras_persister: TestOrasPersister
) -> TestAdapter:
    """Create test adapter with custom ORAS persister."""
    config = AdapterConfig(framework_id="test-e2e", adapter_name="E2E Test Adapter")
    return TestAdapter(config, test_output_dir, custom_persister=oras_persister)


@pytest.fixture
def client(test_adapter: TestAdapter) -> TestClient:
    """Create test client with test adapter."""
    router = create_adapter_api(test_adapter)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


@pytest.mark.e2e
class TestPersistE2E:
    """E2E tests for persist endpoint with mock persister."""

    def test_persist_uses_custom_persister(
        self,
        client: TestClient,
        test_adapter: TestAdapter,
    ) -> None:
        """Test that persist endpoint uses custom persister."""
        # Setup: Create completed job
        job = EvaluationJob(
            job_id="e2e_test_job",
            status=JobStatus.COMPLETED,
            request=EvaluationRequest(
                benchmark_id="test_benchmark", model=ModelConfig(name="test_model")
            ),
            submitted_at=datetime.now(timezone.utc),
        )
        test_adapter._jobs["e2e_test_job"] = job

        # Execute: Call persist endpoint
        response = client.post(
            "/api/v1/evaluations/e2e_test_job/persist",
            json={"oci_ref": "localhost:5001/test/artifact:latest"},
        )

        # Assert: Response is successful
        assert response.status_code == 200
        data = response.json()

        # Verify response data
        assert data["job_id"] == "e2e_test_job"
        assert data["files_count"] == 3
        assert "@sha256:" in data["oci_ref"]

        # External verification with skopeo.

        # Remove tag if present before digest (e.g., :latest@sha256:... -> @sha256:...)
        oci_ref = data["oci_ref"]
        if "@" in oci_ref and ":" in oci_ref.split("@")[0].split("/")[-1]:
            # Split on @ to get [base:tag, digest], then remove :tag from base
            base, digest = oci_ref.split("@", 1)
            base = base.rsplit(":", 1)[0]  # Remove tag
            oci_ref = f"{base}@{digest}"

        result = subprocess.run(
            [
                "skopeo",
                "inspect",
                "--raw",
                f"docker://{oci_ref}",
                "--tls-verify=False",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        manifest = json.loads(result.stdout)

        # Verify it's an OCI manifest
        assert manifest["mediaType"] == "application/vnd.oci.image.manifest.v1+json"
