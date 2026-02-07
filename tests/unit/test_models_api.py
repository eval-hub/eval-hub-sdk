"""Unit tests for API models."""

from datetime import UTC, datetime
from typing import Any

import pytest
from evalhub.models.api import (
    BenchmarkInfo,
    BenchmarksList,
    CollectionList,
    ErrorInfo,
    ErrorResponse,
    EvaluationJob,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    EvaluationStatus,
    FrameworkInfo,
    HealthResponse,
    JobsList,
    JobStatus,
    ModelConfig,
    ProviderList,
)
from pydantic import ValidationError


class TestModelConfig:
    """Test cases for ModelConfig model."""

    def test_basic_model_config(self) -> None:
        """Test basic ModelConfig creation."""
        config = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        assert config.name == "test-model"
        assert config.url == "http://localhost:8000/v1"

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
        assert "num_few_shot" not in request.benchmark_config.keys()
        assert request.benchmark_config == {}
        assert request.experiment_name is None

    def test_full_evaluation_request(self) -> None:
        """Test EvaluationRequest with all fields."""
        model_payload: dict[str, Any] = {
            "url": "http://localhost:8000/v1",
            "name": "gpt-4",
        }
        model = ModelConfig(**model_payload)
        request = EvaluationRequest(
            benchmark_id="mmlu",
            model=model,
            num_examples=100,
            benchmark_config={"subset": "college_math"},
            experiment_name="test_run_1",
        )
        assert request.benchmark_id == "mmlu"
        assert request.model.name == "gpt-4"
        assert request.num_examples == 100
        assert "num_few_shot" not in request.benchmark_config.keys()
        assert request.benchmark_config == {"subset": "college_math"}
        assert request.experiment_name == "test_run_1"


class TestEvaluationJob:
    """Test cases for EvaluationJob model."""

    def test_basic_evaluation_job(self) -> None:
        """Test basic EvaluationJob creation."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="test", model=model)
        now = datetime.now(UTC)

        job = EvaluationJob(
            id="job_123",
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
        assert job.error is None

    def test_completed_evaluation_job(self) -> None:
        """Test completed EvaluationJob."""
        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        request = EvaluationRequest(benchmark_id="test", model=model)
        now = datetime.now(UTC)

        job = EvaluationJob(
            id="job_456",
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
        now = datetime.now(UTC)

        job = EvaluationJob(
            id="job_error",
            status=JobStatus.FAILED,
            request=request,
            submitted_at=now,
            error=ErrorInfo(message="Model not found", message_code="model_not_found"),
        )
        assert job.status == JobStatus.FAILED
        assert job.error is not None
        assert job.error.message == "Model not found"
        assert job.error.message_code == "model_not_found"


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
        now = datetime.now(UTC)

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
        now = datetime.now(UTC)

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
            error=ErrorInfo(
                message="Database connection failed",
                message_code="database_connection_failed",
            ),
        )
        assert health.status == "unhealthy"
        assert health.error is not None
        assert health.error.message == "Database connection failed"
        assert health.error.message_code == "database_connection_failed"
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


class TestListModelsServerCompatibility:
    """Test cases for list models to ensure server API compatibility.

    These tests verify that the client models correctly parse server responses
    using the consistent naming that the Go API uses (items, total_count).
    """

    def test_provider_list_with_server_fields(self) -> None:
        """Test ProviderList parses server response format."""
        # Go API returns 'items' and 'total_count'
        server_response = {
            "items": [
                {
                    "provider_id": "lm_eval",
                    "provider_name": "LM Evaluation Harness",
                    "description": "Evaluation harness for language models",
                    "provider_type": "lm_evaluation_harness",
                    "benchmarks": [],
                }
            ],
            "total_count": 1,
        }

        provider_list = ProviderList.model_validate(server_response)
        assert provider_list.total_count == 1
        assert len(provider_list.items) == 1
        assert provider_list.items[0].provider_id == "lm_eval"
        assert provider_list.items[0].provider_name == "LM Evaluation Harness"

    def test_benchmarks_list_with_server_fields(self) -> None:
        """Test BenchmarksList parses server response format."""
        # Go API returns 'items' and 'total_count'
        server_response = {
            "items": [
                {
                    "benchmark_id": "mmlu",
                    "provider_id": "lm_eval",
                    "name": "MMLU",
                    "description": "Massive Multitask Language Understanding",
                    "category": "knowledge",
                    "metrics": ["accuracy"],
                    "num_few_shot": 5,
                    "dataset_size": 1000,
                    "tags": [],
                }
            ],
            "total_count": 1,
        }

        benchmarks_list = BenchmarksList.model_validate(server_response)
        assert benchmarks_list.total_count == 1
        assert len(benchmarks_list.items) == 1
        assert benchmarks_list.items[0].benchmark_id == "mmlu"

    def test_collection_list_with_server_fields(self) -> None:
        """Test CollectionList parses server response format."""
        # Go API returns 'items' and 'total_count'
        server_response = {
            "items": [
                {
                    "resource": {
                        "id": "healthcare_v1",
                        "tenant": "default",
                        "created_at": "2026-01-27T12:00:00Z",
                        "updated_at": "2026-01-27T12:00:00Z",
                    },
                    "name": "Healthcare Safety v1",
                    "description": "Healthcare benchmarks",
                    "benchmarks": [
                        {"benchmark_id": "medqa", "provider_id": "lm_eval"},
                        {"benchmark_id": "pubmedqa", "provider_id": "lm_eval"},
                    ],
                }
            ],
            "total_count": 1,
        }

        collection_list = CollectionList.model_validate(server_response)
        assert collection_list.total_count == 1
        assert len(collection_list.items) == 1
        assert collection_list.items[0].resource.id == "healthcare_v1"

    def test_jobs_list_with_server_fields(self) -> None:
        """Test JobsList parses server response format."""
        # Go API returns 'items' and 'total_count'
        server_response = {
            "items": [
                {
                    "id": "job-123",
                    "status": "completed",
                    "request": {
                        "benchmark_id": "mmlu",
                        "model": {"name": "test-model", "url": "http://localhost:8000"},
                    },
                    "submitted_at": "2026-01-27T12:00:00Z",
                    "started_at": "2026-01-27T12:01:00Z",
                    "completed_at": "2026-01-27T12:30:00Z",
                }
            ],
            "total_count": 1,
        }

        jobs_list = JobsList.model_validate(server_response)
        assert jobs_list.total_count == 1
        assert len(jobs_list.items) == 1
        assert jobs_list.items[0].id == "job-123"

    def test_empty_provider_list(self) -> None:
        """Test ProviderList handles empty server response."""
        server_response = {
            "items": [],
            "total_count": 0,
        }

        provider_list = ProviderList.model_validate(server_response)
        assert provider_list.total_count == 0
        assert len(provider_list.items) == 0

    def test_empty_benchmarks_list(self) -> None:
        """Test BenchmarksList handles empty server response."""
        server_response = {
            "items": [],
            "total_count": 0,
        }

        benchmarks_list = BenchmarksList.model_validate(server_response)
        assert benchmarks_list.total_count == 0
        assert len(benchmarks_list.items) == 0
