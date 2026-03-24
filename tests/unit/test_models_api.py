"""Unit tests for API models."""

from datetime import UTC, datetime
from typing import Any

import pytest
from evalhub.models.api import (
    BenchmarkConfig,
    BenchmarkInfo,
    BenchmarksList,
    CollectionList,
    CollectionRef,
    ErrorInfo,
    ErrorResponse,
    EvaluationJob,
    EvaluationResponse,
    EvaluationResult,
    EvaluationStatus,
    FrameworkInfo,
    HealthResponse,
    JobsList,
    JobStatus,
    JobSubmissionRequest,
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


class TestEvaluationJob:
    """Test cases for EvaluationJob model."""

    def test_basic_evaluation_job(self) -> None:
        """Test basic EvaluationJob creation."""
        from evalhub.models.api import (
            BenchmarkConfig,
            EvaluationJobResource,
            EvaluationJobStatus,
        )

        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        now = datetime.now(UTC)

        job = EvaluationJob(
            resource=EvaluationJobResource(
                id="job_123",
                tenant="default",
                created_at=now,
                updated_at=now,
            ),
            name="test-eval",
            description="A test evaluation",
            tags=["unit-test"],
            status=EvaluationJobStatus(state=JobStatus.PENDING),
            model=model,
            benchmarks=[
                BenchmarkConfig(id="test", provider_id="test_provider", parameters={})
            ],
        )
        assert job.id == "job_123"
        assert job.state == JobStatus.PENDING
        assert job.name == "test-eval"
        assert job.description == "A test evaluation"
        assert job.tags == ["unit-test"]
        assert job.benchmarks is not None
        assert job.benchmarks[0].id == "test"
        assert job.resource.created_at == now

    def test_completed_evaluation_job(self) -> None:
        """Test completed EvaluationJob."""
        from evalhub.models.api import (
            BenchmarkConfig,
            EvaluationJobResource,
            EvaluationJobResults,
            EvaluationJobStatus,
        )

        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        now = datetime.now(UTC)

        job = EvaluationJob(
            resource=EvaluationJobResource(
                id="job_456",
                tenant="default",
                created_at=now,
                updated_at=now,
            ),
            name="completed-eval",
            status=EvaluationJobStatus(state=JobStatus.COMPLETED),
            model=model,
            benchmarks=[
                BenchmarkConfig(id="test", provider_id="test_provider", parameters={})
            ],
            results=EvaluationJobResults(
                benchmarks=[],
            ),
        )
        assert job.state == JobStatus.COMPLETED
        assert job.results is not None

    def test_failed_evaluation_job(self) -> None:
        """Test failed EvaluationJob."""
        from evalhub.models.api import (
            BenchmarkConfig,
            EvaluationJobResource,
            EvaluationJobStatus,
            MessageInfo,
        )

        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        now = datetime.now(UTC)

        job = EvaluationJob(
            resource=EvaluationJobResource(
                id="job_error",
                tenant="default",
                created_at=now,
                updated_at=now,
            ),
            name="failed-eval",
            status=EvaluationJobStatus(
                state=JobStatus.FAILED,
                message=MessageInfo(
                    message="Model not found", message_code="model_not_found"
                ),
            ),
            model=model,
            benchmarks=[
                BenchmarkConfig(id="test", provider_id="test_provider", parameters={})
            ],
        )
        assert job.state == JobStatus.FAILED
        assert job.status is not None
        assert job.status.message is not None
        assert job.status.message.message == "Model not found"
        assert job.status.message.message_code == "model_not_found"

    def test_evaluation_job_with_collection(self) -> None:
        """Test EvaluationJob created via collection reference."""
        from evalhub.models.api import (
            EvaluationJobResource,
            EvaluationJobStatus,
        )

        model = ModelConfig(url="http://localhost:8000/v1", name="test-model")
        now = datetime.now(UTC)

        job = EvaluationJob(
            resource=EvaluationJobResource(
                id="job_coll_1",
                tenant="default",
                created_at=now,
                updated_at=now,
            ),
            name="collection-eval",
            status=EvaluationJobStatus(state=JobStatus.PENDING),
            model=model,
            collection=CollectionRef(id="healthcare_v1"),
        )
        assert job.id == "job_coll_1"
        assert job.benchmarks is None
        assert job.collection is not None
        assert job.collection.id == "healthcare_v1"


class TestCollectionRef:
    """Test cases for CollectionRef model."""

    def test_basic_collection_ref(self) -> None:
        """Test CollectionRef with id only."""
        ref = CollectionRef(id="healthcare_v1")
        assert ref.id == "healthcare_v1"
        assert ref.benchmarks is None

    def test_collection_ref_with_benchmarks(self) -> None:
        """Test CollectionRef with optional benchmark subset."""
        ref = CollectionRef(
            id="healthcare_v1",
            benchmarks=[
                BenchmarkConfig(id="medqa", provider_id="lm_eval", parameters={}),
            ],
        )
        assert ref.id == "healthcare_v1"
        assert ref.benchmarks is not None
        assert len(ref.benchmarks) == 1
        assert ref.benchmarks[0].id == "medqa"


class TestJobSubmissionRequest:
    """Test cases for JobSubmissionRequest model."""

    def test_submission_with_benchmarks(self) -> None:
        """Test JobSubmissionRequest with direct benchmarks."""
        request = JobSubmissionRequest(
            name="test-eval",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            benchmarks=[
                BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
            ],
        )
        assert request.benchmarks is not None
        assert request.collection is None

    def test_submission_with_collection(self) -> None:
        """Test JobSubmissionRequest with collection reference."""
        request = JobSubmissionRequest(
            name="test-eval",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            collection=CollectionRef(id="healthcare_v1"),
        )
        assert request.benchmarks is None
        assert request.collection is not None
        assert request.collection.id == "healthcare_v1"

    def test_submission_with_collection_and_benchmark_subset(self) -> None:
        """Test JobSubmissionRequest with collection and benchmark subset."""
        request = JobSubmissionRequest(
            name="test-eval",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            collection=CollectionRef(
                id="healthcare_v1",
                benchmarks=[
                    BenchmarkConfig(id="medqa", provider_id="lm_eval", parameters={}),
                ],
            ),
        )
        assert request.collection is not None
        assert request.collection.benchmarks is not None
        assert len(request.collection.benchmarks) == 1

    def test_submission_rejects_both(self) -> None:
        """Test that specifying both benchmarks and collection is rejected."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            JobSubmissionRequest(
                name="test-eval",
                model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
                benchmarks=[
                    BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
                ],
                collection=CollectionRef(id="healthcare_v1"),
            )

    def test_submission_rejects_neither(self) -> None:
        """Test that specifying neither benchmarks nor collection is rejected."""
        with pytest.raises(ValidationError, match="Must specify either"):
            JobSubmissionRequest(
                name="test-eval",
                model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            )

    def test_submission_excludes_none_on_dump(self) -> None:
        """Test that model_dump(exclude_none=True) produces clean payloads."""
        request = JobSubmissionRequest(
            name="test-eval",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            collection=CollectionRef(id="healthcare_v1"),
        )
        dumped = request.model_dump(exclude_none=True)
        assert "benchmarks" not in dumped
        assert "collection" in dumped
        assert dumped["collection"]["id"] == "healthcare_v1"


class TestExperimentConfig:
    """Test cases for ExperimentConfig and ExperimentTag models."""

    def test_basic_experiment_config(self) -> None:
        """Test basic ExperimentConfig creation."""
        from evalhub.models.api import ExperimentConfig, ExperimentTag

        config = ExperimentConfig(
            name="my-experiment",
            tags=[ExperimentTag(key="team", value="ml-platform")],
            artifact_location="s3://my-bucket/artifacts",
        )
        assert config.name == "my-experiment"
        assert len(config.tags) == 1
        assert config.tags[0].key == "team"
        assert config.tags[0].value == "ml-platform"
        assert config.artifact_location == "s3://my-bucket/artifacts"

    def test_experiment_config_defaults(self) -> None:
        """Test ExperimentConfig with all defaults."""
        from evalhub.models.api import ExperimentConfig

        config = ExperimentConfig()
        assert config.name is None
        assert config.tags == []
        assert config.artifact_location is None

    def test_experiment_tag_validation(self) -> None:
        """Test ExperimentTag field constraints."""
        from evalhub.models.api import ExperimentTag

        tag = ExperimentTag(key="k", value="v")
        assert tag.key == "k"
        assert tag.value == "v"

        with pytest.raises(ValidationError):
            ExperimentTag(key="a" * 251, value="v")  # key too long

        with pytest.raises(ValidationError):
            ExperimentTag(key="k", value="v" * 5001)  # value too long

    def test_job_submission_with_experiment(self) -> None:
        """Test JobSubmissionRequest includes experiment field."""
        from evalhub.models.api import (
            BenchmarkConfig,
            ExperimentConfig,
            ExperimentTag,
            JobSubmissionRequest,
        )

        request = JobSubmissionRequest(
            name="eval-with-experiment",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            benchmarks=[
                BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
            ],
            experiment=ExperimentConfig(
                name="tracking-experiment",
                tags=[ExperimentTag(key="version", value="1.0")],
            ),
        )
        assert request.experiment is not None
        assert request.experiment.name == "tracking-experiment"
        assert len(request.experiment.tags) == 1

    def test_job_submission_without_experiment(self) -> None:
        """Test JobSubmissionRequest defaults experiment to None."""
        from evalhub.models.api import BenchmarkConfig, JobSubmissionRequest

        request = JobSubmissionRequest(
            name="eval-no-experiment",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            benchmarks=[
                BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
            ],
        )
        assert request.experiment is None

    def test_evaluation_job_with_experiment(self) -> None:
        """Test EvaluationJob includes experiment from API response."""
        from evalhub.models.api import (
            BenchmarkConfig,
            EvaluationJobResource,
            EvaluationJobStatus,
            ExperimentConfig,
            ExperimentTag,
        )

        now = datetime.now(UTC)
        job = EvaluationJob(
            resource=EvaluationJobResource(
                id="job_exp",
                tenant="default",
                created_at=now,
                updated_at=now,
            ),
            name="experiment-eval",
            status=EvaluationJobStatus(state=JobStatus.RUNNING),
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            benchmarks=[
                BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
            ],
            experiment=ExperimentConfig(
                name="my-exp",
                tags=[ExperimentTag(key="env", value="staging")],
                artifact_location="s3://bucket/path",
            ),
        )
        assert job.experiment is not None
        assert job.experiment.name == "my-exp"
        assert job.experiment.artifact_location == "s3://bucket/path"

    def test_experiment_serialization_roundtrip(self) -> None:
        """Test ExperimentConfig serializes and deserializes correctly."""
        from evalhub.models.api import ExperimentConfig, ExperimentTag

        config = ExperimentConfig(
            name="roundtrip",
            tags=[
                ExperimentTag(key="k1", value="v1"),
                ExperimentTag(key="k2", value="v2"),
            ],
        )
        data = config.model_dump()
        restored = ExperimentConfig.model_validate(data)
        assert restored.name == "roundtrip"
        assert len(restored.tags) == 2
        assert restored.tags[0].key == "k1"


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
        # Go API returns 'items' and 'total_count' with nested resource
        server_response = {
            "items": [
                {
                    "resource": {
                        "id": "lm_eval",
                        "tenant": "default",
                        "created_at": "2026-01-27T12:00:00Z",
                        "updated_at": "2026-01-27T12:00:00Z",
                    },
                    "name": "LM Evaluation Harness",
                    "description": "Evaluation harness for language models",
                    "benchmarks": [],
                }
            ],
            "total_count": 1,
        }

        provider_list = ProviderList.model_validate(server_response)
        assert provider_list.total_count == 1
        assert len(provider_list.items) == 1
        assert provider_list.items[0].resource.id == "lm_eval"
        assert provider_list.items[0].name == "LM Evaluation Harness"

    def test_benchmarks_list_with_server_fields(self) -> None:
        """Test BenchmarksList parses server response format."""
        # Go API returns 'items' and 'total_count'
        server_response = {
            "items": [
                {
                    "id": "mmlu",
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
        assert benchmarks_list.items[0].id == "mmlu"

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
                        {"id": "medqa", "provider_id": "lm_eval"},
                        {"id": "pubmedqa", "provider_id": "lm_eval"},
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
        # Go API returns 'items' and 'total_count' with nested structure
        server_response = {
            "items": [
                {
                    "resource": {
                        "id": "job-123",
                        "tenant": "default",
                        "created_at": "2026-01-27T12:00:00Z",
                        "updated_at": "2026-01-27T12:30:00Z",
                    },
                    "name": "mmlu-eval",
                    "status": {"state": JobStatus.COMPLETED.value},
                    "model": {"name": "test-model", "url": "http://localhost:8000"},
                    "benchmarks": [
                        {
                            "id": "mmlu",
                            "provider_id": "lm_eval",
                            "parameters": {},
                        }
                    ],
                }
            ],
            "total_count": 1,
        }

        jobs_list = JobsList.model_validate(server_response)
        assert jobs_list.total_count == 1
        assert len(jobs_list.items) == 1
        assert jobs_list.items[0].id == "job-123"
        assert jobs_list.items[0].state == JobStatus.COMPLETED

    def test_jobs_list_with_collection_ref(self) -> None:
        """Test JobsList parses server response with collection reference."""
        server_response = {
            "items": [
                {
                    "resource": {
                        "id": "job-456",
                        "tenant": "default",
                        "created_at": "2026-01-27T12:00:00Z",
                        "updated_at": "2026-01-27T12:30:00Z",
                    },
                    "name": "collection-eval",
                    "status": {"state": JobStatus.PENDING.value},
                    "model": {"name": "test-model", "url": "http://localhost:8000"},
                    "collection": {
                        "id": "healthcare_v1",
                        "benchmarks": [
                            {
                                "id": "medqa",
                                "provider_id": "lm_eval",
                                "parameters": {},
                            }
                        ],
                    },
                }
            ],
            "total_count": 1,
        }

        jobs_list = JobsList.model_validate(server_response)
        assert jobs_list.total_count == 1
        assert len(jobs_list.items) == 1
        job = jobs_list.items[0]
        assert job.id == "job-456"
        assert job.benchmarks is None
        assert job.collection is not None
        assert job.collection.id == "healthcare_v1"
        assert job.collection.benchmarks is not None
        assert len(job.collection.benchmarks) == 1

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
