"""Integration tests for adapter functionality."""

from datetime import datetime, timezone

# typing imports removed - using PEP 604 union syntax
import pytest
from evalhub.adapter.client.adapter_client import AdapterClient
from evalhub.adapter.models.framework import AdapterConfig, FrameworkAdapter
from evalhub.adapter.server.app import create_adapter_app
from evalhub.models.api import (
    BenchmarkInfo,
    EvaluationJob,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    FrameworkInfo,
    HealthResponse,
    JobStatus,
)
from fastapi.testclient import TestClient


class IntegrationTestAdapter(FrameworkAdapter):
    """Test adapter for integration testing."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._jobs: dict[str, EvaluationJob] = {}
        self._job_counter = 0
        self._benchmarks = [
            BenchmarkInfo(
                benchmark_id="integration_test",
                name="Integration Test Benchmark",
                description="A benchmark for integration testing",
                category="testing",
                metrics=["accuracy", "precision", "recall"],
            ),
            BenchmarkInfo(
                benchmark_id="performance_test",
                name="Performance Test Benchmark",
                description="Performance testing benchmark",
                category="performance",
                metrics=["throughput", "latency"],
            ),
        ]

    async def initialize(self) -> None:
        """Initialize the adapter."""
        pass

    async def get_framework_info(self) -> FrameworkInfo:
        """Get framework information."""
        return FrameworkInfo(
            framework_id=self.config.framework_id,
            name=self.config.adapter_name,
            version=self.config.version,
            description="Integration test framework adapter",
            supported_benchmarks=self._benchmarks,
            supported_model_types=["test", "mock"],
            capabilities=["classification", "generation"],
        )

    async def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List available benchmarks."""
        return self._benchmarks.copy()

    async def get_benchmark_info(self, benchmark_id: str) -> BenchmarkInfo | None:
        """Get benchmark information."""
        for benchmark in self._benchmarks:
            if benchmark.benchmark_id == benchmark_id:
                return benchmark
        return None

    async def submit_evaluation(self, request: EvaluationRequest) -> EvaluationJob:
        """Submit evaluation job."""
        self._job_counter += 1
        job_id = f"integration_job_{self._job_counter}"

        job = EvaluationJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            request=request,
            submitted_at=datetime.now(timezone.utc),
        )

        self._jobs[job_id] = job
        return job

    async def get_job_status(self, job_id: str) -> EvaluationJob | None:
        """Get job status."""
        return self._jobs.get(job_id)

    async def get_evaluation_results(self, job_id: str) -> EvaluationResponse | None:
        """Get evaluation results."""
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.COMPLETED:
            return None

        # Generate mock results based on benchmark
        results = []
        if job.request.benchmark_id == "integration_test":
            results = [
                EvaluationResult(metric_name="accuracy", metric_value=0.92),
                EvaluationResult(metric_name="precision", metric_value=0.89),
                EvaluationResult(metric_name="recall", metric_value=0.95),
            ]
        elif job.request.benchmark_id == "performance_test":
            results = [
                EvaluationResult(metric_name="throughput", metric_value=1000),
                EvaluationResult(metric_name="latency", metric_value=50.5),
            ]

        return EvaluationResponse(
            job_id=job_id,
            benchmark_id=job.request.benchmark_id,
            model_name=job.request.model.name,
            results=results,
            overall_score=sum(
                float(r.metric_value)
                for r in results
                if isinstance(r.metric_value, int | float)
            )
            / len(results)
            if results
            else 0,
            num_examples_evaluated=job.request.num_examples or 100,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=60.0,
        )

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel job."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.CANCELLED
            return True
        return False

    async def health_check(self) -> HealthResponse:
        """Perform health check."""
        return HealthResponse(
            status="healthy",
            framework_id=self.config.framework_id,
            version=self.config.version,
            uptime_seconds=3600.0,
            dependencies={
                "test_service": {"status": "healthy", "response_time": 5.2},
            },
        )

    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        pass

    def _complete_job(self, job_id: str) -> None:
        """Helper to complete a job for testing."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.COMPLETED
            self._jobs[job_id].completed_at = datetime.now(timezone.utc)
            self._jobs[job_id].progress = 1.0


@pytest.mark.integration
class TestAdapterIntegration:
    """Integration tests for adapter functionality."""

    @pytest.fixture
    def config(self) -> AdapterConfig:
        """Create test configuration."""
        return AdapterConfig(
            framework_id="integration_test_framework",
            adapter_name="Integration Test Adapter",
            version="1.0.0",
        )

    @pytest.fixture
    def adapter(self, config: AdapterConfig) -> IntegrationTestAdapter:
        """Create test adapter."""
        return IntegrationTestAdapter(config)

    @pytest.fixture
    def test_client(self, adapter: IntegrationTestAdapter) -> TestClient:
        """Create test client."""
        app = create_adapter_app(adapter)
        return TestClient(app)

    def test_full_evaluation_workflow_via_http(
        self, test_client: TestClient, adapter: IntegrationTestAdapter
    ) -> None:
        """Test full evaluation workflow via HTTP endpoints."""
        # 1. Check health
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"

        # 2. Get framework info
        response = test_client.get("/api/v1/info")
        assert response.status_code == 200
        info_data = response.json()
        assert info_data["framework_id"] == "integration_test_framework"

        # 3. List benchmarks
        response = test_client.get("/api/v1/benchmarks")
        assert response.status_code == 200
        benchmarks_data = response.json()
        assert len(benchmarks_data) == 2

        # 4. Get specific benchmark
        response = test_client.get("/api/v1/benchmarks/integration_test")
        assert response.status_code == 200
        benchmark_data = response.json()
        assert benchmark_data["benchmark_id"] == "integration_test"

        # 5. Submit evaluation
        request_data = {
            "benchmark_id": "integration_test",
            "model": {
                "url": "http://localhost:8000/v1",
                "name": "test-model",
                "provider": "test",
                "parameters": {"temperature": 0.1},
            },
            "num_examples": 50,
            "experiment_name": "integration_test_run",
        }

        response = test_client.post("/api/v1/evaluations", json=request_data)
        assert response.status_code == 201
        job_data = response.json()
        job_id = job_data["job_id"]
        assert job_data["status"] == "pending"

        # 6. Check job status
        response = test_client.get(f"/api/v1/evaluations/{job_id}")
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["job_id"] == job_id

        # 7. Complete the job (simulate)
        adapter._complete_job(job_id)

        # 8. Get results
        response = test_client.get(f"/api/v1/evaluations/{job_id}/results")
        assert response.status_code == 200
        results_data = response.json()
        assert results_data["job_id"] == job_id
        assert len(results_data["results"]) == 3  # accuracy, precision, recall

        # 9. Cancel a job (create new one first)
        response = test_client.post("/api/v1/evaluations", json=request_data)
        assert response.status_code == 201
        new_job_data = response.json()
        new_job_id = new_job_data["job_id"]

        response = test_client.delete(f"/api/v1/evaluations/{new_job_id}")
        assert response.status_code == 200
        cancel_data = response.json()
        assert cancel_data["success"] is True

    @pytest.mark.asyncio
    async def test_full_evaluation_workflow_via_client(
        self, test_client: TestClient, adapter: IntegrationTestAdapter
    ) -> None:
        """Test full evaluation workflow via client SDK."""
        # Note: This would need a running server in a real integration test
        # For now, we'll test the client logic directly

        # Create mock responses for the client
        with test_client:
            base_url = "http://testserver"  # TestClient uses this as base

            # We would need to mock the httpx client or use a real server
            # For this example, we'll just verify the client can be created
            adapter_client = AdapterClient(base_url)
            assert adapter_client.base_url == base_url

    def test_error_handling_integration(self, test_client: TestClient) -> None:
        """Test error handling across the integration."""
        # Test 404 for non-existent benchmark
        response = test_client.get("/api/v1/benchmarks/non_existent")
        assert response.status_code == 404

        # Test 404 for non-existent job
        response = test_client.get("/api/v1/evaluations/non_existent")
        assert response.status_code == 404

        # Test validation error for invalid request
        invalid_request = {
            "benchmark_id": "",  # Empty benchmark ID
            "model": {
                "url": "http://localhost:8000/v1",
                "name": "",
            },  # Empty model name
        }
        response = test_client.post("/api/v1/evaluations", json=invalid_request)
        assert response.status_code == 422

    def test_concurrent_evaluations(
        self, test_client: TestClient, adapter: IntegrationTestAdapter
    ) -> None:
        """Test handling multiple concurrent evaluations."""
        request_data = {
            "benchmark_id": "performance_test",
            "model": {"url": "http://localhost:8000/v1", "name": "test-model"},
            "num_examples": 10,
        }

        job_ids = []

        # Submit multiple jobs
        for i in range(3):
            response = test_client.post("/api/v1/evaluations", json=request_data)
            assert response.status_code == 201
            job_data = response.json()
            job_ids.append(job_data["job_id"])

        # Verify all jobs exist and have unique IDs
        assert len(set(job_ids)) == 3

        # Complete all jobs
        for job_id in job_ids:
            adapter._complete_job(job_id)

        # Verify all results can be retrieved
        for job_id in job_ids:
            response = test_client.get(f"/api/v1/evaluations/{job_id}/results")
            assert response.status_code == 200
            results_data = response.json()
            assert results_data["job_id"] == job_id

    def test_benchmark_specific_results(
        self, test_client: TestClient, adapter: IntegrationTestAdapter
    ) -> None:
        """Test that different benchmarks produce different results."""
        # Submit evaluation for integration_test benchmark
        request1 = {
            "benchmark_id": "integration_test",
            "model": {"url": "http://localhost:8000/v1", "name": "test-model"},
        }
        response = test_client.post("/api/v1/evaluations", json=request1)
        job1_data = response.json()
        job1_id = job1_data["job_id"]
        adapter._complete_job(job1_id)

        # Submit evaluation for performance_test benchmark
        request2 = {
            "benchmark_id": "performance_test",
            "model": {"url": "http://localhost:8000/v1", "name": "test-model"},
        }
        response = test_client.post("/api/v1/evaluations", json=request2)
        job2_data = response.json()
        job2_id = job2_data["job_id"]
        adapter._complete_job(job2_id)

        # Get results for both
        response1 = test_client.get(f"/api/v1/evaluations/{job1_id}/results")
        results1 = response1.json()

        response2 = test_client.get(f"/api/v1/evaluations/{job2_id}/results")
        results2 = response2.json()

        # Verify different metrics
        assert len(results1["results"]) == 3  # accuracy, precision, recall
        assert len(results2["results"]) == 2  # throughput, latency

        metric_names1 = [r["metric_name"] for r in results1["results"]]
        metric_names2 = [r["metric_name"] for r in results2["results"]]

        assert set(metric_names1) == {"accuracy", "precision", "recall"}
        assert set(metric_names2) == {"throughput", "latency"}
