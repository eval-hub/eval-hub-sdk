"""Unit tests for FrameworkAdapter."""

from datetime import datetime, timezone

# typing imports removed - using PEP 604 union syntax
import pytest
from evalhub.adapter.models.framework import AdapterConfig, FrameworkAdapter
from evalhub.models.api import (
    BenchmarkInfo,
    EvaluationJob,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    FrameworkInfo,
    HealthResponse,
    JobStatus,
    ModelConfig,
)


class MockFrameworkAdapter(FrameworkAdapter):
    """Mock implementation of FrameworkAdapter for testing."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._jobs: dict[str, EvaluationJob] = {}
        self._job_counter = 0
        self._benchmarks = [
            BenchmarkInfo(
                benchmark_id="mock_benchmark",
                name="Mock Benchmark",
                description="A mock benchmark for testing",
                category="testing",
                metrics=["accuracy", "f1_score"],
            )
        ]

    async def initialize(self) -> None:
        """Initialize the mock adapter."""
        pass

    async def get_framework_info(self) -> FrameworkInfo:
        """Get framework information."""
        return FrameworkInfo(
            framework_id=self.config.framework_id,
            name=self.config.adapter_name,
            version=self.config.version,
            description="Mock framework for testing",
            supported_benchmarks=self._benchmarks,
            supported_model_types=["mock"],
            capabilities=["text-classification"],
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
        job_id = f"mock_job_{self._job_counter}"

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

        results = [
            EvaluationResult(metric_name="accuracy", metric_value=0.85),
            EvaluationResult(metric_name="f1_score", metric_value=0.82),
        ]

        return EvaluationResponse(
            job_id=job_id,
            benchmark_id=job.request.benchmark_id,
            model_name=job.request.model.name,
            results=results,
            overall_score=0.835,
            num_examples_evaluated=100,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=120.0,
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
        )

    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        pass

    def _complete_job(self, job_id: str) -> None:
        """Helper method to mark job as completed for testing."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.COMPLETED
            self._jobs[job_id].completed_at = datetime.now(timezone.utc)
            self._jobs[job_id].progress = 1.0


class TestAdapterConfig:
    """Test cases for AdapterConfig."""

    def test_basic_config(self) -> None:
        """Test basic AdapterConfig creation."""
        config = AdapterConfig(
            framework_id="test_framework",
            adapter_name="Test Adapter",
        )
        assert config.framework_id == "test_framework"
        assert config.adapter_name == "Test Adapter"
        assert config.version == "1.0.0"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.max_concurrent_jobs == 10
        assert config.job_timeout_seconds == 3600
        assert config.log_level == "INFO"
        assert config.framework_config == {}

    def test_full_config(self) -> None:
        """Test AdapterConfig with all fields."""
        framework_config = {"model_cache_dir": "/models", "device": "cuda"}

        config = AdapterConfig(
            framework_id="custom_framework",
            adapter_name="Custom Adapter",
            version="2.0.0",
            host="localhost",
            port=9000,
            workers=4,
            max_concurrent_jobs=5,
            job_timeout_seconds=7200,
            log_level="DEBUG",
            framework_config=framework_config,
        )

        assert config.framework_id == "custom_framework"
        assert config.adapter_name == "Custom Adapter"
        assert config.version == "2.0.0"
        assert config.host == "localhost"
        assert config.port == 9000
        assert config.workers == 4
        assert config.max_concurrent_jobs == 5
        assert config.job_timeout_seconds == 7200
        assert config.log_level == "DEBUG"
        assert config.framework_config == framework_config


class TestMockFrameworkAdapter:
    """Test cases for MockFrameworkAdapter."""

    @pytest.fixture
    def config(self) -> AdapterConfig:
        """Create test configuration."""
        return AdapterConfig(
            framework_id="mock_framework",
            adapter_name="Mock Test Adapter",
        )

    @pytest.fixture
    def adapter(self, config: AdapterConfig) -> MockFrameworkAdapter:
        """Create mock adapter."""
        return MockFrameworkAdapter(config)

    @pytest.mark.asyncio
    async def test_initialization(self, adapter: MockFrameworkAdapter) -> None:
        """Test adapter initialization."""
        await adapter.initialize()
        # Should not raise any exceptions
        assert True

    @pytest.mark.asyncio
    async def test_get_framework_info(self, adapter: MockFrameworkAdapter) -> None:
        """Test getting framework information."""
        info = await adapter.get_framework_info()

        assert info.framework_id == "mock_framework"
        assert info.name == "Mock Test Adapter"
        assert info.version == "1.0.0"
        assert info.description == "Mock framework for testing"
        assert len(info.supported_benchmarks) == 1
        assert info.supported_model_types == ["mock"]
        assert info.capabilities == ["text-classification"]

    @pytest.mark.asyncio
    async def test_list_benchmarks(self, adapter: MockFrameworkAdapter) -> None:
        """Test listing benchmarks."""
        benchmarks = await adapter.list_benchmarks()

        assert len(benchmarks) == 1
        assert benchmarks[0].benchmark_id == "mock_benchmark"
        assert benchmarks[0].name == "Mock Benchmark"
        assert benchmarks[0].category == "testing"
        assert benchmarks[0].metrics == ["accuracy", "f1_score"]

    @pytest.mark.asyncio
    async def test_get_benchmark_info_exists(
        self, adapter: MockFrameworkAdapter
    ) -> None:
        """Test getting existing benchmark information."""
        benchmark = await adapter.get_benchmark_info("mock_benchmark")

        assert benchmark is not None
        assert benchmark.benchmark_id == "mock_benchmark"
        assert benchmark.name == "Mock Benchmark"

    @pytest.mark.asyncio
    async def test_get_benchmark_info_not_exists(
        self, adapter: MockFrameworkAdapter
    ) -> None:
        """Test getting non-existent benchmark information."""
        benchmark = await adapter.get_benchmark_info("non_existent")
        assert benchmark is None

    @pytest.mark.asyncio
    async def test_submit_evaluation(self, adapter: MockFrameworkAdapter) -> None:
        """Test submitting evaluation."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(
            benchmark_id="mock_benchmark",
            model=model,
            num_examples=100,
        )

        job = await adapter.submit_evaluation(request)

        assert job.id == "mock_job_1"
        assert job.status == JobStatus.PENDING
        assert job.request.benchmark_id == "mock_benchmark"
        assert job.request.model.name == "test-model"
        assert job.submitted_at is not None

    @pytest.mark.asyncio
    async def test_get_job_status_exists(self, adapter: MockFrameworkAdapter) -> None:
        """Test getting existing job status."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="mock_benchmark", model=model)

        # Submit job first
        job = await adapter.submit_evaluation(request)
        job_id = job.id

        # Get job status
        retrieved_job = await adapter.get_job_status(job_id)

        assert retrieved_job is not None
        assert retrieved_job.id == job_id
        assert retrieved_job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_job_status_not_exists(
        self, adapter: MockFrameworkAdapter
    ) -> None:
        """Test getting non-existent job status."""
        job = await adapter.get_job_status("non_existent_job")
        assert job is None

    @pytest.mark.asyncio
    async def test_get_evaluation_results_completed(
        self, adapter: MockFrameworkAdapter
    ) -> None:
        """Test getting results for completed job."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="mock_benchmark", model=model)

        # Submit and complete job
        job = await adapter.submit_evaluation(request)
        adapter._complete_job(job.id)

        # Get results
        results = await adapter.get_evaluation_results(job.id)

        assert results is not None
        assert results.job_id == job.id
        assert results.benchmark_id == "mock_benchmark"
        assert results.model_name == "test-model"
        assert len(results.results) == 2
        assert results.overall_score == 0.835
        assert results.num_examples_evaluated == 100

    @pytest.mark.asyncio
    async def test_get_evaluation_results_not_completed(
        self, adapter: MockFrameworkAdapter
    ) -> None:
        """Test getting results for non-completed job."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="mock_benchmark", model=model)

        # Submit but don't complete job
        job = await adapter.submit_evaluation(request)

        # Try to get results
        results = await adapter.get_evaluation_results(job.id)
        assert results is None

    @pytest.mark.asyncio
    async def test_cancel_job_exists(self, adapter: MockFrameworkAdapter) -> None:
        """Test canceling existing job."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="mock_benchmark", model=model)

        # Submit job
        job = await adapter.submit_evaluation(request)

        # Cancel job
        result = await adapter.cancel_job(job.id)
        assert result is True

        # Check job status
        cancelled_job = await adapter.get_job_status(job.id)
        assert cancelled_job is not None
        assert cancelled_job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_job_not_exists(self, adapter: MockFrameworkAdapter) -> None:
        """Test canceling non-existent job."""
        result = await adapter.cancel_job("non_existent_job")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check(self, adapter: MockFrameworkAdapter) -> None:
        """Test health check."""
        health = await adapter.health_check()

        assert health.status == "healthy"
        assert health.framework_id == "mock_framework"
        assert health.version == "1.0.0"
        assert health.uptime_seconds == 3600.0

    @pytest.mark.asyncio
    async def test_shutdown(self, adapter: MockFrameworkAdapter) -> None:
        """Test adapter shutdown."""
        await adapter.shutdown()
        # Should not raise any exceptions
        assert True

    @pytest.mark.asyncio
    async def test_multiple_jobs(self, adapter: MockFrameworkAdapter) -> None:
        """Test submitting multiple jobs."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="mock_benchmark", model=model)

        # Submit multiple jobs
        job1 = await adapter.submit_evaluation(request)
        job2 = await adapter.submit_evaluation(request)

        assert job1.id == "mock_job_1"
        assert job2.id == "mock_job_2"
        assert job1.id != job2.id

        # Both jobs should be retrievable
        retrieved1 = await adapter.get_job_status(job1.id)
        retrieved2 = await adapter.get_job_status(job2.id)

        assert retrieved1 is not None
        assert retrieved2 is not None
        assert retrieved1.id != retrieved2.id


class TestFrameworkAdapterAbstract:
    """Test abstract FrameworkAdapter behavior."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that FrameworkAdapter cannot be instantiated directly."""
        config = AdapterConfig(framework_id="test", adapter_name="Test")

        with pytest.raises(TypeError):
            FrameworkAdapter(config)  # type: ignore

    def test_config_attribute(self) -> None:
        """Test that adapter stores config correctly."""
        config = AdapterConfig(
            framework_id="test_framework",
            adapter_name="Test Adapter",
            version="2.0.0",
        )

        adapter = MockFrameworkAdapter(config)
        assert adapter.config == config
        assert adapter.config.framework_id == "test_framework"
        assert adapter.config.adapter_name == "Test Adapter"
        assert adapter.config.version == "2.0.0"
