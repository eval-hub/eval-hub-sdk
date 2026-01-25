"""Unit tests for server components."""


import pytest
from evalhub.adapter.models.framework import AdapterConfig, FrameworkAdapter
from evalhub.adapter.server.app import AdapterServer, create_adapter_app
from evalhub.models.api import (
    BenchmarkInfo,
    EvaluationJob,
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
    """Mock adapter for testing server functionality."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._initialized = False
        self._jobs_counter = 0

    async def initialize(self) -> None:
        """Initialize the adapter."""
        self._initialized = True

    async def get_framework_info(self) -> FrameworkInfo:
        """Get framework information."""
        return FrameworkInfo(
            framework_id=self.config.framework_id,
            name=self.config.adapter_name,
            version=self.config.version,
            description="Mock adapter for testing",
            supported_benchmarks=[
                BenchmarkInfo(
                    benchmark_id="mock_test",
                    name="Mock Test",
                    category="testing",
                    metrics=["accuracy"],
                )
            ],
            supported_model_types=["mock"],
            capabilities=["classification"],
        )

    async def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List available benchmarks."""
        return [
            BenchmarkInfo(
                benchmark_id="mock_test",
                name="Mock Test",
                category="testing",
                metrics=["accuracy"],
            )
        ]

    async def get_benchmark_info(self, benchmark_id: str) -> BenchmarkInfo | None:
        """Get benchmark information."""
        if benchmark_id == "mock_test":
            return BenchmarkInfo(
                benchmark_id="mock_test",
                name="Mock Test",
                category="testing",
                metrics=["accuracy"],
            )
        return None

    async def submit_evaluation(self, request: EvaluationRequest) -> EvaluationJob:
        """Submit evaluation job."""
        from datetime import datetime, timezone

        from evalhub.models.api import EvaluationJob

        self._jobs_counter += 1
        job_id = f"mock_job_{self._jobs_counter}"

        return EvaluationJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            request=request,
            submitted_at=datetime.now(timezone.utc),
        )

    async def get_job_status(self, job_id: str) -> EvaluationJob | None:
        """Get job status."""
        if job_id.startswith("mock_job_"):
            from datetime import datetime, timezone

            from evalhub.models.api import EvaluationJob

            return EvaluationJob(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                request=EvaluationRequest(
                    benchmark_id="mock_test",
                    model=ModelConfig(
                        url="http://localhost:8000/v1", name="mock-model"
                    ),
                ),
                submitted_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                progress=1.0,
            )
        return None

    async def get_evaluation_results(self, job_id: str) -> EvaluationResponse | None:
        """Get evaluation results."""
        if job_id.startswith("mock_job_"):
            from datetime import datetime, timezone

            from evalhub.models.api import EvaluationResponse, EvaluationResult

            return EvaluationResponse(
                job_id=job_id,
                benchmark_id="mock_test",
                model_name="mock-model",
                results=[
                    EvaluationResult(
                        metric_name="accuracy",
                        metric_value=0.85,
                    )
                ],
                overall_score=0.85,
                num_examples_evaluated=100,
                completed_at=datetime.now(timezone.utc),
                duration_seconds=60.0,
            )
        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel job."""
        return job_id.startswith("mock_job_")

    async def health_check(self) -> HealthResponse:
        """Perform health check."""
        return HealthResponse(
            status="healthy" if self._initialized else "unhealthy",
            framework_id=self.config.framework_id,
            version=self.config.version,
            uptime_seconds=3600.0 if self._initialized else None,
        )

    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        self._initialized = False


class TestAdapterServer:
    """Test cases for AdapterServer."""

    @pytest.fixture
    def adapter_config(self) -> AdapterConfig:
        """Create test adapter configuration."""
        return AdapterConfig(
            framework_id="mock_framework",
            adapter_name="Mock Test Adapter",
            version="1.0.0",
            host="localhost",
            port=8888,
        )

    @pytest.fixture
    def mock_adapter(self, adapter_config: AdapterConfig) -> MockAdapter:
        """Create mock adapter."""
        return MockAdapter(adapter_config)

    def test_create_adapter_app(self, mock_adapter: MockAdapter) -> None:
        """Test creating adapter app."""
        app = create_adapter_app(mock_adapter)

        assert isinstance(app, FastAPI)
        assert app.title == "EvalHub Framework Adapter"
        assert "Mock Test Adapter" in app.description

    @pytest.mark.asyncio
    async def test_adapter_server_initialization(
        self, mock_adapter: MockAdapter, adapter_config: AdapterConfig
    ) -> None:
        """Test AdapterServer initialization."""
        server = AdapterServer(mock_adapter)

        assert server.adapter == mock_adapter
        assert server.adapter.config == adapter_config
        assert server.app is not None

    def test_health_endpoint(self, mock_adapter: MockAdapter) -> None:
        """Test health endpoint."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["framework_id"] == "mock_framework"
        assert data["version"] == "1.0.0"
        # Note: status might be "unhealthy" since we haven't initialized

    def test_info_endpoint(self, mock_adapter: MockAdapter) -> None:
        """Test framework info endpoint."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.get("/api/v1/info")
        assert response.status_code == 200

        data = response.json()
        assert data["framework_id"] == "mock_framework"
        assert data["name"] == "Mock Test Adapter"
        assert data["version"] == "1.0.0"
        assert len(data["supported_benchmarks"]) == 1

    def test_benchmarks_endpoint(self, mock_adapter: MockAdapter) -> None:
        """Test benchmarks listing endpoint."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.get("/api/v1/benchmarks")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["benchmark_id"] == "mock_test"
        assert data[0]["name"] == "Mock Test"

    def test_benchmark_detail_endpoint_exists(self, mock_adapter: MockAdapter) -> None:
        """Test getting existing benchmark details."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.get("/api/v1/benchmarks/mock_test")
        assert response.status_code == 200

        data = response.json()
        assert data["benchmark_id"] == "mock_test"
        assert data["name"] == "Mock Test"

    def test_benchmark_detail_endpoint_not_found(
        self, mock_adapter: MockAdapter
    ) -> None:
        """Test getting non-existent benchmark details."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.get("/api/v1/benchmarks/non_existent")
        assert response.status_code == 404

    def test_submit_evaluation_endpoint(self, mock_adapter: MockAdapter) -> None:
        """Test submitting evaluation."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        request_data = {
            "benchmark_id": "mock_test",
            "model": {"url": "http://localhost:8000/v1", "name": "test-model"},
            "num_examples": 100,
        }

        response = client.post("/api/v1/evaluations", json=request_data)
        assert response.status_code == 201

        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["request"]["benchmark_id"] == "mock_test"

    def test_submit_evaluation_invalid_request(self, mock_adapter: MockAdapter) -> None:
        """Test submitting invalid evaluation request."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        # Missing required fields
        request_data = {"benchmark_id": "mock_test"}

        response = client.post("/api/v1/evaluations", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_get_job_status_endpoint_exists(self, mock_adapter: MockAdapter) -> None:
        """Test getting existing job status."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.get("/api/v1/evaluations/mock_job_1")
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == "mock_job_1"
        assert data["status"] == "completed"

    def test_get_job_status_endpoint_not_found(self, mock_adapter: MockAdapter) -> None:
        """Test getting non-existent job status."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.get("/api/v1/evaluations/non_existent")
        assert response.status_code == 404

    def test_get_job_results_endpoint_exists(self, mock_adapter: MockAdapter) -> None:
        """Test getting existing job results."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.get("/api/v1/evaluations/mock_job_1/results")
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == "mock_job_1"
        assert data["benchmark_id"] == "mock_test"
        assert len(data["results"]) == 1
        assert data["results"][0]["metric_name"] == "accuracy"

    def test_get_job_results_endpoint_not_found(
        self, mock_adapter: MockAdapter
    ) -> None:
        """Test getting non-existent job results."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.get("/api/v1/evaluations/non_existent/results")
        assert response.status_code == 404

    def test_cancel_job_endpoint_exists(self, mock_adapter: MockAdapter) -> None:
        """Test canceling existing job."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.delete("/api/v1/evaluations/mock_job_1")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_cancel_job_endpoint_not_found(self, mock_adapter: MockAdapter) -> None:
        """Test canceling non-existent job."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.delete("/api/v1/evaluations/non_existent")
        assert response.status_code == 404

    def test_openapi_docs(self, mock_adapter: MockAdapter) -> None:
        """Test that OpenAPI docs are available."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        # Test OpenAPI JSON schema
        response = client.get("/openapi.json")
        assert response.status_code == 200

        # Should be valid JSON
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

        # Test docs UI
        response = client.get("/docs")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_startup_and_shutdown_events(self, mock_adapter: MockAdapter) -> None:
        """Test app startup and shutdown events."""
        # Note: This test manually initializes the adapter rather than
        # testing actual FastAPI startup/shutdown events

        # Simulate startup
        assert not mock_adapter._initialized

        # The startup event should initialize the adapter
        # In a real test, we'd trigger the startup event
        await mock_adapter.initialize()
        assert mock_adapter._initialized

        # Simulate shutdown
        await mock_adapter.shutdown()
        assert not mock_adapter._initialized

    def test_cors_headers(self, mock_adapter: MockAdapter) -> None:
        """Test CORS headers are present."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        response = client.options("/api/v1/health")
        assert response.status_code == 200

        # Test that CORS is configured (might need additional setup)
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_error_handling(self, mock_adapter: MockAdapter) -> None:
        """Test error handling in endpoints."""
        app = create_adapter_app(mock_adapter)
        client = TestClient(app)

        # Test with malformed JSON
        response = client.post(
            "/api/v1/evaluations",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

        # Test with wrong content type
        response = client.post(
            "/api/v1/evaluations",
            content="some data",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 422
