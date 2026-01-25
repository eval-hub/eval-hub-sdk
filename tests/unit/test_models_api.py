"""Unit tests for API models."""

from datetime import datetime, timezone
from typing import Any

import pytest
from evalhub.models.api import (
    BenchmarkInfo,
    ErrorResponse,
    EvaluationJob,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    EvaluationStatus,
    FrameworkInfo,
    HealthResponse,
    JobStatus,
    ModelConfig,
)
from pydantic import ValidationError


class TestModelConfig:
    """Test cases for ModelConfig model."""

    def test_basic_model_config(self) -> None:
        """Test basic ModelConfig creation."""
        config = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        assert config.name == "test-model"
        assert config.url == "http://localhost:8000/v1"

    def test_full_model_config(self) -> None:
        """Test ModelConfig with extra fields (extra='allow')."""
        config = ModelConfig(
            url="http://localhost:8000/v1",
            name="gpt-4",
            provider="openai",
            parameters={"temperature": 0.1, "max_tokens": 100},
            device="cuda:0",
            batch_size=8,
        )
        assert config.name == "gpt-4"
        assert config.url == "http://localhost:8000/v1"
        assert config.provider == "openai"
        assert config.parameters == {"temperature": 0.1, "max_tokens": 100}
        assert config.device == "cuda:0"
        assert config.batch_size == 8

    def test_model_config_validation(self) -> None:
        """Test ModelConfig validation."""
        with pytest.raises(ValidationError):
            ModelConfig(
                url="http://localhost:8000/v1", name=""
            )  # Empty name should fail
        with pytest.raises(ValidationError):
            ModelConfig(url="", name="test-model")  # Empty URL should fail


class TestBenchmarkInfo:
    """Test cases for BenchmarkInfo model."""

    def test_basic_benchmark_info(self) -> None:
        """Test basic BenchmarkInfo creation."""
        benchmark = BenchmarkInfo(
            benchmark_id="test_benchmark",
            name="Test Benchmark",
            description="A test benchmark",
            category="testing",
            metrics=["accuracy"],
        )
        assert benchmark.benchmark_id == "test_benchmark"
        assert benchmark.name == "Test Benchmark"
        assert benchmark.description == "A test benchmark"
        assert benchmark.category == "testing"
        assert benchmark.metrics == ["accuracy"]

    def test_benchmark_info_optional_fields(self) -> None:
        """Test BenchmarkInfo with optional fields."""
        benchmark = BenchmarkInfo(
            benchmark_id="minimal",
            name="Minimal",
        )
        assert benchmark.benchmark_id == "minimal"
        assert benchmark.name == "Minimal"
        assert benchmark.description is None
        assert benchmark.category is None
        assert benchmark.metrics == []

    def test_benchmark_info_validation(self) -> None:
        """Test BenchmarkInfo validation."""
        with pytest.raises(ValidationError):
            BenchmarkInfo(benchmark_id="", name="test")  # Empty ID should fail

        with pytest.raises(ValidationError):
            BenchmarkInfo(benchmark_id="test", name="")  # Empty name should fail


class TestEvaluationRequest:
    """Test cases for EvaluationRequest model."""

    def test_basic_evaluation_request(self) -> None:
        """Test basic EvaluationRequest creation."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(
            benchmark_id="test_bench",
            model=model,
        )
        assert request.benchmark_id == "test_bench"
        assert request.model.name == "test-model"
        assert request.num_examples is None
        assert request.num_few_shot is None
        assert request.benchmark_config == {}
        assert request.experiment_name is None

    def test_full_evaluation_request(self) -> None:
        """Test EvaluationRequest with all fields."""
        model = ModelConfig(
            url="http://localhost:8000/v1",
            name="gpt-4",
            provider="openai",
        )
        request = EvaluationRequest(
            benchmark_id="mmlu",
            model=model,
            num_examples=100,
            num_few_shot=5,
            benchmark_config={"subset": "college_math"},
            experiment_name="test_run_1",
        )
        assert request.benchmark_id == "mmlu"
        assert request.model.name == "gpt-4"
        assert request.num_examples == 100
        assert request.num_few_shot == 5
        assert request.benchmark_config == {"subset": "college_math"}
        assert request.experiment_name == "test_run_1"


class TestEvaluationJob:
    """Test cases for EvaluationJob model."""

    def test_basic_evaluation_job(self) -> None:
        """Test basic EvaluationJob creation."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="test", model=model)
        now = datetime.now(timezone.utc)

        job = EvaluationJob(
            job_id="job_123",
            status=JobStatus.PENDING,
            request=request,
            submitted_at=now,
        )
        assert job.id == "job_123"
        assert job.status == JobStatus.PENDING
        assert job.request.benchmark_id == "test"
        assert job.submitted_at == now
        assert job.started_at is None
        assert job.completed_at is None
        assert job.progress is None
        assert job.error_message is None

    def test_completed_evaluation_job(self) -> None:
        """Test completed EvaluationJob."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="test", model=model)
        now = datetime.now(timezone.utc)

        job = EvaluationJob(
            job_id="job_456",
            status=JobStatus.COMPLETED,
            request=request,
            submitted_at=now,
            started_at=now,
            completed_at=now,
            progress=1.0,
        )
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 1.0
        assert job.completed_at == now

    def test_failed_evaluation_job(self) -> None:
        """Test failed EvaluationJob."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="test", model=model)
        now = datetime.now(timezone.utc)

        job = EvaluationJob(
            job_id="job_error",
            status=JobStatus.FAILED,
            request=request,
            submitted_at=now,
            error_message="Model not found",
        )
        assert job.status == JobStatus.FAILED
        assert job.error_message == "Model not found"


class TestEvaluationResult:
    """Test cases for EvaluationResult model."""

    def test_float_result(self) -> None:
        """Test EvaluationResult with float value."""
        result = EvaluationResult(
            metric_name="accuracy",
            metric_value=0.85,
            metric_type="float",
            num_samples=1000,
        )
        assert result.metric_name == "accuracy"
        assert result.metric_value == 0.85
        assert result.metric_type == "float"
        assert result.num_samples == 1000

    def test_string_result(self) -> None:
        """Test EvaluationResult with string value."""
        result = EvaluationResult(
            metric_name="grade",
            metric_value="A+",
            metric_type="string",
        )
        assert result.metric_name == "grade"
        assert result.metric_value == "A+"
        assert result.metric_type == "string"
        assert result.num_samples is None

    def test_boolean_result(self) -> None:
        """Test EvaluationResult with boolean value."""
        result = EvaluationResult(
            metric_name="passed",
            metric_value=True,
            metric_type="bool",
        )
        assert result.metric_name == "passed"
        assert result.metric_value is True
        assert result.metric_type == "bool"


class TestEvaluationResponse:
    """Test cases for EvaluationResponse model."""

    def test_evaluation_response(self) -> None:
        """Test EvaluationResponse creation."""
        results = [
            EvaluationResult(metric_name="accuracy", metric_value=0.85),
            EvaluationResult(metric_name="f1_score", metric_value=0.82),
        ]
        now = datetime.now(timezone.utc)

        response = EvaluationResponse(
            job_id="job_123",
            benchmark_id="test_benchmark",
            model_name="test-model",
            results=results,
            overall_score=0.835,
            num_examples_evaluated=1000,
            completed_at=now,
            duration_seconds=300.5,
        )
        assert response.job_id == "job_123"
        assert response.benchmark_id == "test_benchmark"
        assert response.model_name == "test-model"
        assert len(response.results) == 2
        assert response.overall_score == 0.835
        assert response.num_examples_evaluated == 1000
        assert response.completed_at == now
        assert response.duration_seconds == 300.5

    def test_evaluation_response_without_overall_score(self) -> None:
        """Test EvaluationResponse without overall score."""
        results = [
            EvaluationResult(metric_name="accuracy", metric_value=0.85),
        ]
        now = datetime.now(timezone.utc)

        response = EvaluationResponse(
            job_id="job_123",
            benchmark_id="test_benchmark",
            model_name="test-model",
            results=results,
            num_examples_evaluated=1000,
            completed_at=now,
            duration_seconds=300.5,
        )
        assert response.overall_score is None


class TestFrameworkInfo:
    """Test cases for FrameworkInfo model."""

    def test_framework_info(self) -> None:
        """Test FrameworkInfo creation."""
        benchmarks = [
            BenchmarkInfo(benchmark_id="test1", name="Test 1"),
            BenchmarkInfo(benchmark_id="test2", name="Test 2"),
        ]

        info = FrameworkInfo(
            framework_id="test_framework",
            name="Test Framework",
            version="1.0.0",
            description="A test framework",
            supported_benchmarks=benchmarks,
            supported_model_types=["gpt", "claude"],
            capabilities=["text-generation", "classification"],
        )
        assert info.framework_id == "test_framework"
        assert info.name == "Test Framework"
        assert info.version == "1.0.0"
        assert info.description == "A test framework"
        assert len(info.supported_benchmarks) == 2
        assert info.supported_model_types == ["gpt", "claude"]
        assert info.capabilities == ["text-generation", "classification"]


class TestHealthResponse:
    """Test cases for HealthResponse model."""

    def test_healthy_response(self) -> None:
        """Test healthy HealthResponse."""
        deps: dict[str, Any] = {
            "database": {"status": "healthy", "latency": 5.2},
            "redis": {"status": "healthy", "connected": True},
        }

        health = HealthResponse(
            status="healthy",
            framework_id="test_framework",
            version="1.0.0",
            uptime_seconds=3600.0,
            dependencies=deps,
        )
        assert health.status == "healthy"
        assert health.framework_id == "test_framework"
        assert health.version == "1.0.0"
        assert health.uptime_seconds == 3600.0
        assert health.dependencies == deps

    def test_unhealthy_response(self) -> None:
        """Test unhealthy HealthResponse."""
        health = HealthResponse(
            status="unhealthy",
            framework_id="test_framework",
            version="1.0.0",
            error_message="Database connection failed",
        )
        assert health.status == "unhealthy"
        assert health.error_message == "Database connection failed"
        assert health.uptime_seconds is None
        assert health.dependencies is None


class TestErrorResponse:
    """Test cases for ErrorResponse model."""

    def test_error_response(self) -> None:
        """Test ErrorResponse creation."""
        error = ErrorResponse(
            error="ValidationError",
            message="Invalid benchmark ID",
            details={"field": "benchmark_id", "value": ""},
        )
        assert error.error == "ValidationError"
        assert error.message == "Invalid benchmark ID"
        assert error.details == {"field": "benchmark_id", "value": ""}

    def test_simple_error_response(self) -> None:
        """Test simple ErrorResponse without details."""
        error = ErrorResponse(
            error="NotFound",
            message="Benchmark not found",
        )
        assert error.error == "NotFound"
        assert error.message == "Benchmark not found"
        assert error.details is None


class TestEnums:
    """Test cases for enum types."""

    def test_job_status_enum(self) -> None:
        """Test JobStatus enum values."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_evaluation_status_enum(self) -> None:
        """Test EvaluationStatus enum values."""
        assert EvaluationStatus.QUEUED == "queued"
        assert EvaluationStatus.INITIALIZING == "initializing"
        assert EvaluationStatus.RUNNING == "running"
        assert EvaluationStatus.POST_PROCESSING == "post_processing"
        assert EvaluationStatus.COMPLETED == "completed"
        assert EvaluationStatus.FAILED == "failed"
        assert EvaluationStatus.CANCELLED == "cancelled"
