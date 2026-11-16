"""Unit tests for API models."""

from datetime import UTC, datetime
from typing import Any

import pytest
from evalhub.models.api import (
    BenchmarkConfig,
    BenchmarkHardwareConfig,
    BenchmarkInfo,
    BenchmarkResult,
    BenchmarksList,
    CalibrationDataRef,
    CollectionList,
    CollectionRef,
    ConfidenceInterval,
    ErrorInfo,
    ErrorResponse,
    EvaluationExports,
    EvaluationExportsOCI,
    EvaluationJob,
    EvaluationJobResults,
    EvaluationResponse,
    EvaluationResult,
    EvaluationStatus,
    FrameworkInfo,
    GitTestDataRef,
    HardwareDRAConfig,
    HardwareResourceQuantity,
    HealthResponse,
    HFTestDataRef,
    JobsList,
    JobStatus,
    JobSubmissionRequest,
    MetricSchema,
    ModelConfig,
    OCIConnectionConfig,
    OCICoordinates,
    OCIDataRef,
    PassCriteria,
    PostProcessingCommon,
    PostProcessingResource,
    PostProcessingResults,
    PostProcessingResultsDataRef,
    PostProcessingStatus,
    PrimaryScore,
    ProviderList,
    PVCTestDataRef,
    QueueConfig,
    ResultsDataConfig,
    ResultsDataConfigEntry,
    ResultType,
    S3TestDataRef,
    StandaloneConfidenceIntervalConfig,
    StandalonePostProcessingRequest,
    TestDataRef,
)
from pydantic import ValidationError

pytestmark = pytest.mark.unit


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

    def test_model_config_empty_url_for_prerecorded(self) -> None:
        """Test ModelConfig accepts empty URL for pre-recorded evaluation jobs."""
        config = ModelConfig(url="", name="test-model")
        assert config.url == ""
        assert config.name == "test-model"

    def test_model_config_default_url_is_empty(self) -> None:
        """Test ModelConfig defaults url to empty string when omitted."""
        config = ModelConfig(name="test-model")
        assert config.url == ""


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
        assert job.hardware_config is None
        assert "hardware_config" not in job.model_dump(exclude_none=True)

    @pytest.mark.parametrize(
        "hardware_config",
        [
            {"hardware_profile_name": "gpu-profile"},
            {"cpu": {"request": "2"}, "queue": {"name": "eval-queue"}},
        ],
    )
    def test_hardware_config_response_round_trip(
        self, hardware_config: dict[str, Any]
    ) -> None:
        payload = {
            "resource": {
                "id": "job-hardware",
                "created_at": "2026-01-01T00:00:00Z",
            },
            "name": "hardware-eval",
            "model": {"name": "test-model"},
            "benchmarks": [{"id": "mmlu", "provider_id": "lm_eval"}],
            "hardware_config": hardware_config,
        }

        job = EvaluationJob.model_validate(payload)

        assert job.hardware_config == BenchmarkHardwareConfig.model_validate(
            hardware_config
        )
        assert job.benchmarks is not None
        assert job.benchmarks[0].hardware_config is None
        serialized = job.model_dump(mode="json", exclude_none=True)
        assert serialized["hardware_config"] == hardware_config
        assert EvaluationJob.model_validate(serialized) == job

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


class TestEffectiveState:
    """Test cases for EvaluationJob.effective_state property."""

    def _make_job(
        self,
        job_state: JobStatus,
        benchmark_states: list[JobStatus] | None = None,
    ) -> EvaluationJob:
        from evalhub.models.api import (
            BenchmarkConfig,
            BenchmarkStatus,
            EvaluationJobResource,
            EvaluationJobStatus,
        )

        now = datetime.now(UTC)
        benchmarks_status = []
        if benchmark_states is not None:
            for i, s in enumerate(benchmark_states):
                benchmarks_status.append(
                    BenchmarkStatus(
                        id=f"bench_{i}",
                        provider_id="test_provider",
                        benchmark_index=i,
                        status=s,
                    )
                )
        return EvaluationJob(
            resource=EvaluationJobResource(
                id="job_eff", tenant="default", created_at=now
            ),
            name="effective-state-test",
            status=EvaluationJobStatus(state=job_state, benchmarks=benchmarks_status),
            model=ModelConfig(url="http://localhost:8000/v1", name="m"),
            benchmarks=[
                BenchmarkConfig(id="b", provider_id="test_provider", parameters={})
            ],
        )

    def test_terminal_top_level_state_is_returned_directly(self) -> None:
        job = self._make_job(JobStatus.COMPLETED)
        assert job.effective_state == JobStatus.COMPLETED

    def test_running_with_no_benchmarks_stays_running(self) -> None:
        job = self._make_job(JobStatus.RUNNING)
        assert job.effective_state == JobStatus.RUNNING

    def test_running_with_all_benchmarks_completed(self) -> None:
        """The core bug scenario: server says running, but all benchmarks done."""
        job = self._make_job(
            JobStatus.RUNNING,
            [JobStatus.COMPLETED, JobStatus.COMPLETED],
        )
        assert job.effective_state == JobStatus.COMPLETED

    def test_running_with_some_benchmarks_still_pending(self) -> None:
        job = self._make_job(
            JobStatus.RUNNING,
            [JobStatus.COMPLETED, JobStatus.PENDING],
        )
        assert job.effective_state == JobStatus.RUNNING

    def test_running_with_one_benchmark_failed(self) -> None:
        job = self._make_job(
            JobStatus.RUNNING,
            [JobStatus.COMPLETED, JobStatus.FAILED],
        )
        assert job.effective_state == JobStatus.PARTIALLY_FAILED

    def test_running_with_all_benchmarks_failed(self) -> None:
        job = self._make_job(
            JobStatus.RUNNING,
            [JobStatus.FAILED, JobStatus.FAILED],
        )
        assert job.effective_state == JobStatus.FAILED

    def test_running_with_one_benchmark_cancelled(self) -> None:
        job = self._make_job(
            JobStatus.RUNNING,
            [JobStatus.COMPLETED, JobStatus.CANCELLED],
        )
        assert job.effective_state == JobStatus.PARTIALLY_FAILED

    def test_running_with_all_benchmarks_cancelled(self) -> None:
        job = self._make_job(
            JobStatus.RUNNING,
            [JobStatus.CANCELLED, JobStatus.CANCELLED],
        )
        assert job.effective_state == JobStatus.CANCELLED

    def test_partially_failed_from_server_is_terminal(self) -> None:
        job = self._make_job(JobStatus.PARTIALLY_FAILED)
        assert job.effective_state == JobStatus.PARTIALLY_FAILED

    def test_pending_with_no_status(self) -> None:
        from evalhub.models.api import (
            BenchmarkConfig,
            EvaluationJobResource,
        )

        now = datetime.now(UTC)
        job = EvaluationJob(
            resource=EvaluationJobResource(
                id="job_none", tenant="default", created_at=now
            ),
            name="no-status",
            status=None,
            model=ModelConfig(url="http://localhost:8000/v1", name="m"),
            benchmarks=[
                BenchmarkConfig(id="b", provider_id="test_provider", parameters={})
            ],
        )
        assert job.effective_state == JobStatus.PENDING


class TestBenchmarkConfig:
    """Test cases for BenchmarkConfig model — primary_score and pass_criteria overrides."""

    def test_defaults_are_none(self) -> None:
        cfg = BenchmarkConfig(id="hellaswag", provider_id="lighteval")
        assert cfg.primary_score is None
        assert cfg.pass_criteria is None

    def test_primary_score_accepted(self) -> None:
        cfg = BenchmarkConfig(
            id="hellaswag",
            provider_id="lighteval",
            primary_score=PrimaryScore(metric="hellaswag.em"),
        )
        assert cfg.primary_score is not None
        assert cfg.primary_score.metric == "hellaswag.em"
        assert cfg.primary_score.lower_is_better is False

    def test_pass_criteria_accepted(self) -> None:
        cfg = BenchmarkConfig(
            id="hellaswag",
            provider_id="lighteval",
            pass_criteria=PassCriteria(threshold=0.99),
        )
        assert cfg.pass_criteria is not None
        assert cfg.pass_criteria.threshold == 0.99

    def test_both_fields_serialized(self) -> None:
        """primary_score and pass_criteria must appear in model_dump output so the
        server receives the per-benchmark threshold override."""
        cfg = BenchmarkConfig(
            id="hellaswag",
            provider_id="lighteval",
            primary_score=PrimaryScore(metric="hellaswag.em"),
            pass_criteria=PassCriteria(threshold=0.99),
        )
        data = cfg.model_dump(exclude_none=True)
        assert data["primary_score"] == {
            "metric": "hellaswag.em",
            "lower_is_better": False,
        }
        assert data["pass_criteria"] == {"threshold": 0.99}

    def test_fields_omitted_when_none(self) -> None:
        cfg = BenchmarkConfig(id="hellaswag", provider_id="lighteval")
        data = cfg.model_dump(exclude_none=True)
        assert "primary_score" not in data
        assert "pass_criteria" not in data

    def test_roundtrip_via_job_submission_request(self) -> None:
        """Regression test: per-benchmark overrides must survive YAML→model→dump→server."""
        import yaml

        raw = yaml.safe_load(
            """
name: violations-demo
model:
  url: http://vllm:8000/v1
  name: my-model
benchmarks:
  - id: hellaswag
    provider_id: lighteval
    primary_score:
      metric: hellaswag.em
    pass_criteria:
      threshold: 0.99
    parameters:
      num_examples: 10
"""
        )
        req = JobSubmissionRequest(**raw)
        assert req.benchmarks is not None
        b = req.benchmarks[0]
        assert b.primary_score is not None
        assert b.primary_score.metric == "hellaswag.em"
        assert b.pass_criteria is not None
        assert b.pass_criteria.threshold == 0.99
        # Verify the fields survive serialisation (what the CLI sends to the server)
        payload = req.model_dump(exclude_none=True)
        bench_payload = payload["benchmarks"][0]
        assert bench_payload["primary_score"]["metric"] == "hellaswag.em"
        assert bench_payload["pass_criteria"]["threshold"] == 0.99


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

    @pytest.mark.parametrize("collection", [False, True])
    def test_hardware_config_submission(self, collection: bool) -> None:
        benchmark = BenchmarkConfig(
            id="mmlu",
            provider_id="lm_eval",
            hardware_config=BenchmarkHardwareConfig(
                hardware_profile_name="gpu-profile"
            ),
        )
        request = JobSubmissionRequest(
            name="hardware-eval",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            benchmarks=None if collection else [benchmark],
            collection=(
                CollectionRef(id="my-collection", benchmarks=[benchmark])
                if collection
                else None
            ),
            hardware_config=BenchmarkHardwareConfig.model_validate(
                {"cpu": {"request": "2"}, "queue": {"name": "eval-queue"}}
            ),
        )
        payload = request.model_dump(
            exclude_none=True, context={"for_submission": True}
        )
        assert payload["hardware_config"] == {
            "cpu": {"request": "2"},
            "queue": {"name": "eval-queue"},
        }
        benchmarks = (
            payload["collection"]["benchmarks"] if collection else payload["benchmarks"]
        )
        assert benchmarks[0]["hardware_config"] == {
            "hardware_profile_name": "gpu-profile"
        }
        assert JobSubmissionRequest.model_validate(payload) == request

    @pytest.mark.parametrize("level", ["evaluation", "benchmark"])
    def test_invalid_hardware_config(self, level: str) -> None:
        raw: dict[str, Any] = {
            "name": "hardware-eval",
            "model": {"url": "http://localhost:8000/v1", "name": "test-model"},
            "benchmarks": [{"id": "mmlu", "provider_id": "lm_eval"}],
        }
        hardware = {"hardware_profile_name": "gpu-profile", "cpu": {"request": "2"}}
        if level == "evaluation":
            raw["hardware_config"] = hardware
        else:
            raw["benchmarks"][0]["hardware_config"] = hardware
        with pytest.raises(ValidationError, match="hardware_profile_name"):
            JobSubmissionRequest.model_validate(raw)

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
        assert request.hardware_config is None
        assert request.benchmarks[0].hardware_config is None
        payload = request.model_dump(exclude_none=True)
        assert "hardware_config" not in payload
        assert "hardware_config" not in payload["benchmarks"][0]

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

    def test_submission_with_exports_oci(self) -> None:
        """Test JobSubmissionRequest with full OCI exports configuration."""
        request = JobSubmissionRequest(
            name="test-eval",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            benchmarks=[
                BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
            ],
            exports=EvaluationExports(
                oci=EvaluationExportsOCI(
                    coordinates=OCICoordinates(
                        oci_host="quay.io",
                        oci_repository="my-org/my-repo",
                        oci_tag="eval-123",
                        annotations={"model": "llama2"},
                    ),
                    k8s=OCIConnectionConfig(connection="my-pull-secret"),
                ),
            ),
        )
        assert request.exports is not None
        assert request.exports.oci is not None
        assert request.exports.oci.coordinates.oci_host == "quay.io"
        assert request.exports.oci.coordinates.oci_repository == "my-org/my-repo"
        assert request.exports.oci.coordinates.oci_tag == "eval-123"
        assert request.exports.oci.k8s is not None
        assert request.exports.oci.k8s.connection == "my-pull-secret"

    def test_submission_with_exports_oci_minimal(self) -> None:
        """Test JobSubmissionRequest with minimal OCI exports (required fields only)."""
        request = JobSubmissionRequest(
            name="test-eval",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            benchmarks=[
                BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
            ],
            exports=EvaluationExports(
                oci=EvaluationExportsOCI(
                    coordinates=OCICoordinates(
                        oci_host="quay.io",
                        oci_repository="my-org/my-repo",
                    ),
                ),
            ),
        )
        assert request.exports is not None
        assert request.exports.oci is not None
        assert request.exports.oci.coordinates.oci_tag is None
        assert request.exports.oci.k8s is None

    def test_submission_exports_excluded_when_none_on_dump(self) -> None:
        """Test that exports is excluded from dump when not set."""
        request = JobSubmissionRequest(
            name="test-eval",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            benchmarks=[
                BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
            ],
        )
        dumped = request.model_dump(exclude_none=True)
        assert "exports" not in dumped

    def test_submission_exports_oci_dump_matches_server_schema(self) -> None:
        """Test that serialized exports matches the server's expected JSON structure."""
        request = JobSubmissionRequest(
            name="test-eval",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            benchmarks=[
                BenchmarkConfig(id="mmlu", provider_id="lm_eval", parameters={})
            ],
            exports=EvaluationExports(
                oci=EvaluationExportsOCI(
                    coordinates=OCICoordinates(
                        oci_host="quay.io",
                        oci_repository="my-org/my-repo",
                        oci_tag="eval-123",
                    ),
                    k8s=OCIConnectionConfig(connection="my-pull-secret"),
                ),
            ),
        )
        dumped = request.model_dump(exclude_none=True)
        assert dumped["exports"] == {
            "oci": {
                "coordinates": {
                    "oci_host": "quay.io",
                    "oci_repository": "my-org/my-repo",
                    "oci_tag": "eval-123",
                    "annotations": {},
                },
                "k8s": {"connection": "my-pull-secret"},
            }
        }


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


class TestBenchmarkResult:
    """Test cases for BenchmarkResult model."""

    def test_additional_info_default_none(self) -> None:
        result = BenchmarkResult(id="mmlu", provider_id="lm_eval")
        assert result.additional_info is None

    def test_additional_info_with_nested_values(self) -> None:
        info: dict[str, Any] = {
            "dataset": [
                {
                    "hf_repo": "openai/gsm8k",
                    "hf_subset": "main",
                    "sha": "740312add88f",
                }
            ],
            "generation_parameters": {"temperature": 0},
            "zero_shot": 0.8,
        }
        result = BenchmarkResult(id="mmlu", provider_id="lm_eval", additional_info=info)
        assert result.additional_info == info

    def test_additional_info_roundtrip(self) -> None:
        info: dict[str, Any] = {
            "prompting_strategy": "chain-of-thought",
            "dataset_sha": "abc123",
        }
        result = BenchmarkResult(
            id="mmlu",
            provider_id="lm_eval",
            metrics={"accuracy": 0.85},
            additional_info=info,
        )
        data = result.model_dump(mode="json")
        assert data["additional_info"] == info
        restored = BenchmarkResult.model_validate(data)
        assert restored.additional_info == info

    def test_additional_info_excluded_when_none(self) -> None:
        result = BenchmarkResult(id="mmlu", provider_id="lm_eval")
        data = result.model_dump(mode="json", exclude_none=True)
        assert "additional_info" not in data

    def test_additional_info_in_job_results(self) -> None:
        info: dict[str, Any] = {"zero_shot": 0.8}
        results = EvaluationJobResults(
            benchmarks=[
                BenchmarkResult(
                    id="mmlu",
                    provider_id="lm_eval",
                    metrics={"accuracy": 0.85},
                    additional_info=info,
                ),
            ],
        )
        data = results.model_dump(mode="json")
        assert data["benchmarks"][0]["additional_info"] == info


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
        assert JobStatus.PARTIALLY_FAILED == "partially_failed"

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
                    "category": "healthcare",
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

    def test_collection_list_without_description(self) -> None:
        """Test CollectionList parses when server omits description."""
        server_response = {
            "items": [
                {
                    "resource": {
                        "id": "c566c23d-8f79-4643-9d61-a6392e810184",
                        "tenant": "dataplane",
                        "created_at": "2026-07-28T15:25:38Z",
                        "updated_at": "2026-07-28T15:25:38Z",
                    },
                    "name": "my-safety-suite",
                    "category": "safety",
                    "benchmarks": [
                        {
                            "id": "truthfulqa_mc1",
                            "provider_id": "lm_evaluation_harness",
                        },
                        {"id": "owasp_llm_top_10", "provider_id": "garak"},
                    ],
                }
            ],
            "total_count": 1,
        }

        collection_list = CollectionList.model_validate(server_response)
        assert collection_list.total_count == 1
        assert len(collection_list.items) == 1
        assert collection_list.items[0].description == ""

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


class TestQueueConfig:
    def test_basic_creation(self) -> None:
        q = QueueConfig(name="user-queue")
        assert q.name == "user-queue"
        assert q.kind is None

    def test_creation_with_kind(self) -> None:
        q = QueueConfig(kind="kueue", name="user-queue")
        assert q.kind == "kueue"
        assert q.name == "user-queue"

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            QueueConfig()  # type: ignore[call-arg]

    def test_serialization_omits_kind_when_none(self) -> None:
        q = QueueConfig(name="user-queue")
        data = q.model_dump(exclude_none=True)
        assert data == {"name": "user-queue"}
        assert "kind" not in data

    def test_serialization_includes_kind_when_set(self) -> None:
        q = QueueConfig(kind="kueue", name="user-queue")
        data = q.model_dump(exclude_none=True)
        assert data == {"kind": "kueue", "name": "user-queue"}

    def test_job_submission_request_with_queue(self) -> None:
        req = JobSubmissionRequest(
            name="my-eval",
            model=ModelConfig(url="http://vllm:8000/v1", name="llama3"),
            benchmarks=[BenchmarkConfig(id="mmlu", provider_id="lm_eval")],
            queue=QueueConfig(name="user-queue"),
        )
        assert req.queue is not None
        assert req.queue.name == "user-queue"
        data = req.model_dump(exclude_none=True)
        assert data["queue"] == {"name": "user-queue"}

    def test_job_submission_request_without_queue(self) -> None:
        req = JobSubmissionRequest(
            name="my-eval",
            model=ModelConfig(url="http://vllm:8000/v1", name="llama3"),
            benchmarks=[BenchmarkConfig(id="mmlu", provider_id="lm_eval")],
        )
        assert req.queue is None
        assert "queue" not in req.model_dump(exclude_none=True)


class TestBenchmarkStatusPhase:
    """Test cases for JobPhase support on BenchmarkStatus."""

    def test_benchmark_status_without_phase(self) -> None:
        """BenchmarkStatus should default phase to None for backward compat."""
        from evalhub.models.api import BenchmarkStatus

        status = BenchmarkStatus(
            id="mmlu",
            provider_id="lm_eval",
            status=JobStatus.RUNNING,
        )
        assert status.phase is None

    def test_benchmark_status_with_phase(self) -> None:
        """BenchmarkStatus should accept and expose a JobPhase value."""
        from evalhub.models.api import BenchmarkStatus, JobPhase

        status = BenchmarkStatus(
            id="mmlu",
            provider_id="lm_eval",
            status=JobStatus.RUNNING,
            phase=JobPhase.RUNNING_EVALUATION,
        )
        assert status.phase == JobPhase.RUNNING_EVALUATION

    def test_benchmark_status_phase_from_json(self) -> None:
        """BenchmarkStatus should deserialize phase from raw JSON (server response)."""
        from evalhub.models.api import BenchmarkStatus, JobPhase

        data: dict[str, Any] = {
            "id": "mmlu",
            "provider_id": "lm_eval",
            "status": "running",
            "phase": "loading_data",
        }
        status = BenchmarkStatus(**data)
        assert status.phase == JobPhase.LOADING_DATA

    def test_benchmark_status_phase_serializes(self) -> None:
        """BenchmarkStatus should serialize phase in model_dump output."""
        from evalhub.models.api import BenchmarkStatus, JobPhase

        status = BenchmarkStatus(
            id="mmlu",
            provider_id="lm_eval",
            status=JobStatus.RUNNING,
            phase=JobPhase.POST_PROCESSING,
        )
        data = status.model_dump(mode="json")
        assert data["phase"] == "post_processing"

    def test_benchmark_status_phase_excluded_when_none(self) -> None:
        """Phase should be absent from exclude_none dumps when unset."""
        from evalhub.models.api import BenchmarkStatus

        status = BenchmarkStatus(
            id="mmlu",
            provider_id="lm_eval",
            status=JobStatus.RUNNING,
        )
        data = status.model_dump(exclude_none=True)
        assert "phase" not in data

    def test_job_phase_enum_values(self) -> None:
        """Verify all expected JobPhase values exist in the common models."""
        from evalhub.models.api import JobPhase

        expected = {
            "initializing",
            "loading_data",
            "running_evaluation",
            "post_processing",
            "persisting_artifacts",
            "completed",
        }
        actual = {p.value for p in JobPhase}
        assert actual == expected

    def test_job_phase_importable_from_models_package(self) -> None:
        """JobPhase should be importable from evalhub.models."""
        from evalhub.models import JobPhase

        assert JobPhase.RUNNING_EVALUATION.value == "running_evaluation"

    def test_adapter_job_phase_is_same_as_common(self) -> None:
        """Adapter JobPhase should be the same class as the common models JobPhase."""
        from evalhub.adapter.models import JobPhase as AdapterJobPhase
        from evalhub.models import JobPhase as CommonJobPhase

        assert AdapterJobPhase is CommonJobPhase


class TestPVCTestDataRef:
    """Tests for PVCTestDataRef and TestDataRef PVC support."""

    def test_pvc_ref_basic(self) -> None:
        ref = PVCTestDataRef(claim_name="my-pvc")
        assert ref.claim_name == "my-pvc"
        assert ref.sub_path is None

    def test_pvc_ref_with_sub_path(self) -> None:
        ref = PVCTestDataRef(claim_name="my-pvc", sub_path="staging")
        assert ref.sub_path == "staging"

    def test_pvc_ref_serializes_correctly(self) -> None:
        ref = PVCTestDataRef(claim_name="my-pvc", sub_path="staging")
        data = ref.model_dump(exclude_none=True)
        assert data == {"claim_name": "my-pvc", "sub_path": "staging"}

    def test_pvc_ref_omits_none_sub_path(self) -> None:
        ref = PVCTestDataRef(claim_name="my-pvc")
        data = ref.model_dump(exclude_none=True)
        assert "sub_path" not in data

    def test_test_data_ref_with_pvc(self) -> None:
        ref = TestDataRef(pvc=PVCTestDataRef(claim_name="my-pvc"))
        assert ref.pvc is not None
        assert ref.s3 is None
        data = ref.model_dump(exclude_none=True)
        assert data == {"pvc": {"claim_name": "my-pvc"}}

    def test_test_data_ref_with_s3(self) -> None:
        ref = TestDataRef(s3=S3TestDataRef(bucket="b", key="k", secret_ref="s"))
        assert ref.s3 is not None
        assert ref.pvc is None

    def test_test_data_ref_rejects_both(self) -> None:
        with pytest.raises(ValidationError, match="Cannot specify more than one"):
            TestDataRef(
                s3=S3TestDataRef(bucket="b", key="k", secret_ref="s"),
                pvc=PVCTestDataRef(claim_name="my-pvc"),
            )

    def test_test_data_ref_rejects_neither(self) -> None:
        with pytest.raises(ValidationError, match="Must specify exactly one"):
            TestDataRef()

    def test_pvc_importable_from_models_package(self) -> None:
        from evalhub.models import PVCTestDataRef as Imported

        assert Imported is PVCTestDataRef

    def test_benchmark_config_with_pvc(self) -> None:
        cfg = BenchmarkConfig(
            id="arc_easy",
            provider_id="lm_evaluation_harness",
            test_data_ref=TestDataRef(
                pvc=PVCTestDataRef(claim_name="my-pvc", sub_path="staging")
            ),
        )
        data = cfg.model_dump(exclude_none=True)
        assert data["test_data_ref"]["pvc"]["claim_name"] == "my-pvc"
        assert data["test_data_ref"]["pvc"]["sub_path"] == "staging"


class TestGitTestDataRef:
    """Tests for GitTestDataRef and TestDataRef git support."""

    def test_git_ref_basic(self) -> None:
        ref = GitTestDataRef(url="https://github.com/org/repo.git", ref="main")
        assert ref.url == "https://github.com/org/repo.git"
        assert ref.ref == "main"
        assert ref.sub_path is None
        assert ref.secret_ref is None

    def test_git_ref_with_all_fields(self) -> None:
        ref = GitTestDataRef(
            url="https://github.com/org/repo.git",
            ref="v1.0",
            sub_path="arc_easy",
            secret_ref="my-git-secret",
        )
        assert ref.sub_path == "arc_easy"
        assert ref.secret_ref == "my-git-secret"

    def test_git_ref_http_without_credentials_allowed(self) -> None:
        ref = GitTestDataRef(url="http://github.com/org/repo.git", ref="main")
        assert ref.url.startswith("http://")

    def test_git_ref_rejects_non_http_scheme(self) -> None:
        with pytest.raises(ValidationError, match="http or https scheme"):
            GitTestDataRef(url="git@github.com:org/repo.git", ref="main")

    def test_git_ref_rejects_hostless_https_url(self) -> None:
        with pytest.raises(ValidationError, match="include a host"):
            GitTestDataRef(url="https://", ref="main")

    def test_git_ref_rejects_hostless_http_url(self) -> None:
        with pytest.raises(ValidationError, match="include a host"):
            GitTestDataRef(url="http://", ref="main")

    def test_git_ref_rejects_http_with_secret_ref(self) -> None:
        with pytest.raises(ValidationError, match="https URL"):
            GitTestDataRef(
                url="http://github.com/org/repo.git",
                ref="main",
                secret_ref="my-secret",
            )

    def test_test_data_ref_with_git(self) -> None:
        ref = TestDataRef(
            git=GitTestDataRef(url="https://github.com/org/repo.git", ref="main")
        )
        assert ref.git is not None
        assert ref.s3 is None
        assert ref.pvc is None

    def test_resolved_sha_visible_in_response(self) -> None:
        ref = TestDataRef(
            git=GitTestDataRef(url="https://github.com/org/repo.git", ref="main"),
            resolved_sha="abc123def456",
        )
        data = ref.model_dump(exclude_none=True)
        assert data["resolved_sha"] == "abc123def456"

    def test_resolved_sha_stripped_from_submission_payload(self) -> None:
        ref = TestDataRef(
            git=GitTestDataRef(url="https://github.com/org/repo.git", ref="main"),
            resolved_sha="abc123def456",
        )
        data = ref.model_dump(exclude_none=True, context={"for_submission": True})
        assert "resolved_sha" not in data

    def test_resolved_sha_absent_means_not_in_dump(self) -> None:
        ref = TestDataRef(
            git=GitTestDataRef(url="https://github.com/org/repo.git", ref="main"),
        )
        data = ref.model_dump(exclude_none=True, context={"for_submission": True})
        assert "resolved_sha" not in data

    def test_git_importable_from_models_package(self) -> None:
        from evalhub.models import GitTestDataRef as Imported

        assert Imported is GitTestDataRef


class TestHFTestDataRef:
    """Tests for HFTestDataRef and TestDataRef HF support."""

    def test_hf_ref_basic(self) -> None:
        ref = HFTestDataRef(repo_id="eval-hub-test/evalhub-offline-testdata")
        assert ref.repo_id == "eval-hub-test/evalhub-offline-testdata"
        assert ref.revision is None
        assert ref.sub_path is None
        assert ref.secret_ref is None

    def test_hf_ref_with_all_fields(self) -> None:
        ref = HFTestDataRef(
            repo_id="cais/mmlu",
            revision="main",
            sub_path="data/train",
            secret_ref="my-hf-credentials",
        )
        assert ref.revision == "main"
        assert ref.sub_path == "data/train"
        assert ref.secret_ref == "my-hf-credentials"

    def test_hf_ref_rejects_blank_repo_id(self) -> None:
        with pytest.raises(ValidationError, match="repo_id must not be blank"):
            HFTestDataRef(repo_id="   ")

    def test_test_data_ref_with_hf(self) -> None:
        ref = TestDataRef(
            hf=HFTestDataRef(repo_id="eval-hub-test/evalhub-offline-testdata")
        )
        assert ref.hf is not None
        assert ref.s3 is None
        assert ref.pvc is None
        assert ref.git is None

    def test_test_data_ref_rejects_hf_and_git(self) -> None:
        with pytest.raises(ValidationError, match="Cannot specify more than one"):
            TestDataRef(
                hf=HFTestDataRef(repo_id="org/dataset"),
                git=GitTestDataRef(url="https://github.com/org/repo.git", ref="main"),
            )

    def test_resolved_sha_visible_in_response(self) -> None:
        ref = TestDataRef(
            hf=HFTestDataRef(repo_id="org/dataset"),
            resolved_sha="abc123def456",
        )
        data = ref.model_dump(exclude_none=True)
        assert data["resolved_sha"] == "abc123def456"

    def test_resolved_sha_stripped_from_submission_payload(self) -> None:
        ref = TestDataRef(
            hf=HFTestDataRef(repo_id="org/dataset", revision="main"),
            resolved_sha="abc123def456",
        )
        data = ref.model_dump(exclude_none=True, context={"for_submission": True})
        assert "resolved_sha" not in data
        assert data["hf"]["repo_id"] == "org/dataset"
        assert data["hf"]["revision"] == "main"

    def test_hf_importable_from_models_package(self) -> None:
        from evalhub.models import HFTestDataRef as Imported

        assert Imported is HFTestDataRef

    def test_benchmark_config_with_hf(self) -> None:
        cfg = BenchmarkConfig(
            id="arc_easy",
            provider_id="lm_evaluation_harness",
            test_data_ref=TestDataRef(
                hf=HFTestDataRef(
                    repo_id="eval-hub-test/evalhub-offline-testdata",
                    revision="main",
                    sub_path="staging_sub_path",
                )
            ),
        )
        data = cfg.model_dump(exclude_none=True)
        assert data["test_data_ref"]["hf"]["repo_id"] == (
            "eval-hub-test/evalhub-offline-testdata"
        )
        assert data["test_data_ref"]["hf"]["sub_path"] == "staging_sub_path"


class TestResultTypeEnum:
    """Tests for the ResultType enum."""

    def test_all_values_present(self) -> None:
        values = {rt.value for rt in ResultType}
        assert values == {
            "numeric",
            "categorical",
            "array_ordered",
            "array_unordered",
            "time_series",
        }

    def test_str_subclass(self) -> None:
        assert ResultType.NUMERIC == "numeric"
        assert ResultType.TIME_SERIES == "time_series"

    def test_importable_from_models_package(self) -> None:
        from evalhub.models import ResultType as Imported

        assert Imported is ResultType


class TestMetricSchema:
    """Tests for the MetricSchema model."""

    def test_basic_construction(self) -> None:
        ms = MetricSchema(name="accuracy", type=ResultType.NUMERIC)
        assert ms.name == "accuracy"
        assert ms.type == ResultType.NUMERIC

    def test_all_result_types_accepted(self) -> None:
        for rt in ResultType:
            ms = MetricSchema(name="m", type=rt)
            assert ms.type == rt

    def test_serializes_type_as_string_value(self) -> None:
        ms = MetricSchema(name="score", type=ResultType.CATEGORICAL)
        data = ms.model_dump()
        assert data["type"] == "categorical"

    def test_roundtrip_from_dict(self) -> None:
        data: dict[str, Any] = {"name": "latency_p99", "type": "time_series"}
        ms = MetricSchema.model_validate(data)
        assert ms.name == "latency_p99"
        assert ms.type == ResultType.TIME_SERIES

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(Exception):
            MetricSchema(name="x", type="unknown_type")  # type: ignore[arg-type]

    def test_importable_from_models_package(self) -> None:
        from evalhub.models import MetricSchema as Imported

        assert Imported is MetricSchema


class TestPostProcessingResponseCompatibility:
    """Tests for forward-compatible post-processing response parsing."""

    def test_status_ignores_unknown_server_fields(self) -> None:
        status = PostProcessingStatus.model_validate(
            {"state": "running", "progress": 0.5}
        )

        assert status.state.value == "running"
        assert status.model_extra is None

    def test_confidence_interval_ignores_unknown_server_fields(self) -> None:
        interval = ConfidenceInterval.model_validate(
            {"lower": 0.1, "upper": 0.9, "confidence_level": 0.95}
        )

        assert interval.lower == 0.1
        assert interval.model_extra is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_labeled", -1),
            ("n_unlabeled", -1),
            ("alpha", -0.1),
            ("alpha", 0.0),
            ("alpha", 1.0),
            ("alpha", 1.1),
        ],
    )
    def test_results_preserve_out_of_range_adapter_metadata(
        self, field: str, value: int | float
    ) -> None:
        payload = {"confidence_interval": {"lower": 0.1, "upper": 0.9, field: value}}

        results = PostProcessingResults.model_validate(payload)

        assert results.model_dump(exclude_none=True) == payload


@pytest.fixture
def post_processing_config_data() -> dict[str, Any]:
    """Minimal confidence-interval configuration using stored job results."""
    return {
        "results_data_ref": {"eval_job": {"id": "eval-123"}},
        "calibration_data_ref": [
            {
                "pvc": {"claim_name": "calibration-data"},
                "data_config": {
                    "format": "jsonl",
                    "columns": {"label": "expected", "prediction": "actual"},
                },
            }
        ],
        "significance_level": 0.05,
    }


class TestPostProcessingContract:
    """Representative wire shapes and business rules, not field permutations."""

    @pytest.mark.parametrize("field", ["format", "path"])
    def test_empty_optional_result_config_strings(self, field: str) -> None:
        config = ResultsDataConfigEntry.model_validate({field: ""})

        assert getattr(config, field) == ""
        assert config.model_dump(exclude_none=True) == {field: ""}

    @pytest.mark.parametrize(
        "payload",
        [{"format": " "}, {"path": " "}, {"path": "/results.jsonl"}],
    )
    def test_invalid_nonempty_result_config_strings(
        self, payload: dict[str, str]
    ) -> None:
        with pytest.raises(ValidationError):
            ResultsDataConfigEntry.model_validate(payload)

    def test_minimal_request_wire_shape(
        self, post_processing_config_data: dict[str, Any]
    ) -> None:
        payload = {"operations": {"confidence_interval": post_processing_config_data}}
        request = StandalonePostProcessingRequest.model_validate(payload)

        assert request.model_dump(mode="json", exclude_none=True) == payload

    def test_external_request_wire_shape(
        self, post_processing_config_data: dict[str, Any]
    ) -> None:
        post_processing_config_data.update(
            results_data_ref={
                "mlflow": {"run_id": "run-123", "artifact_path": "results.jsonl"},
                "data_config": {
                    "format": "jsonl",
                    "columns": {"prediction": "score"},
                    "value_mappings": {"prediction": {"correct": 1.0}},
                },
            },
            primary_score={"metric": "accuracy", "lower_is_better": False},
        )
        payload = {
            "name": "accuracy-ci",
            "hardware_config": {"hardware_profile_name": "small-cpu"},
            "operation_order": ["confidence_interval"],
            "operations": {"confidence_interval": post_processing_config_data},
        }
        request = StandalonePostProcessingRequest.model_validate(payload)

        assert request.model_dump(mode="json", exclude_none=True) == payload

    @pytest.mark.parametrize(
        "source",
        [
            {"eval_job": {"id": "eval-123"}},
            {"s3": {"bucket": "results", "key": "scores.jsonl", "secret_ref": "s3"}},
            {"pvc": {"claim_name": "results"}},
            {"git": {"url": "https://example.com/results.git", "ref": "main"}},
            {"hf": {"repo_id": "org/results"}},
            {"mlflow": {"run_id": "run-123", "artifact_path": "scores.jsonl"}},
            {
                "oci": {
                    "coordinates": {
                        "oci_host": "quay.io",
                        "oci_repository": "org/results",
                    },
                    "artifact_path": "scores.jsonl",
                }
            },
        ],
        ids=["eval-job", "s3", "pvc", "git", "hf", "mlflow", "oci"],
    )
    def test_result_sources(self, source: dict[str, Any]) -> None:
        ref = PostProcessingResultsDataRef.model_validate(source)

        assert getattr(ref, next(iter(source))) is not None
        assert set(ref.model_dump(exclude_none=True)) == set(source)

    @pytest.mark.parametrize("multiple", [False, True], ids=["none", "multiple"])
    def test_source_exclusivity(
        self, post_processing_config_data: dict[str, Any], multiple: bool
    ) -> None:
        source = (
            {"pvc": {"claim_name": "data"}, "hf": {"repo_id": "org/data"}}
            if multiple
            else {}
        )
        with pytest.raises(ValidationError, match="exactly one result data source"):
            PostProcessingResultsDataRef.model_validate(source)

        with pytest.raises(
            ValidationError, match="exactly one calibration data source"
        ):
            CalibrationDataRef.model_validate(
                {
                    **source,
                    "data_config": post_processing_config_data["calibration_data_ref"][
                        0
                    ]["data_config"],
                }
            )

    @pytest.mark.parametrize("eval_job", [False, True], ids=["external", "eval-job"])
    def test_primary_score_source_rules(
        self, post_processing_config_data: dict[str, Any], eval_job: bool
    ) -> None:
        if eval_job:
            post_processing_config_data["primary_score"] = {"metric": "accuracy"}
        else:
            post_processing_config_data["results_data_ref"] = {
                "pvc": {"claim_name": "results"}
            }

        with pytest.raises(ValidationError, match="primary_score"):
            StandaloneConfidenceIntervalConfig.model_validate(
                post_processing_config_data
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("significance_level", 0),
            ("significance_level", 1),
            ("calibration_data_ref", []),
        ],
    )
    def test_configuration_constraints(
        self, post_processing_config_data: dict[str, Any], field: str, value: Any
    ) -> None:
        post_processing_config_data[field] = value
        with pytest.raises(ValidationError) as error:
            StandaloneConfidenceIntervalConfig.model_validate(
                post_processing_config_data
            )
        assert error.value.errors()[0]["loc"] == (field,)

    @pytest.mark.parametrize("per_benchmark", [False, True], ids=["object", "array"])
    def test_data_config_wire_shape(self, per_benchmark: bool) -> None:
        entry = {"format": "csv", "selection": {"benchmark_id": "arc"}}
        payload = [entry] if per_benchmark else entry
        config = ResultsDataConfig.model_validate(payload)

        assert config.model_dump(mode="json", exclude_none=True) == payload
        assert (config.single is None) is per_benchmark
        assert (config.per_benchmark is not None) is per_benchmark

    @pytest.mark.parametrize("payload", [[], [{"format": "csv"}]])
    def test_data_config_array_requires_entries_with_selection(
        self, payload: Any
    ) -> None:
        with pytest.raises(ValidationError):
            ResultsDataConfig.model_validate(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            {"path": "/absolute.json"},
            {"columns": {"unknown": "score"}},
            {"columns": {"prediction": " "}},
            {"selection": {}},
            {"value_mappings": {"prediction": {"correct": True}}},
        ],
        ids=[
            "absolute-path",
            "unknown-column",
            "blank-column",
            "empty-selection",
            "boolean-mapping",
        ],
    )
    def test_invalid_data_mappings(self, payload: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            ResultsDataConfigEntry.model_validate(payload)

    def test_direct_hardware_and_conflicting_modes(self) -> None:
        direct = {
            "cpu": {"request": "1"},
            "gpu": {"name": "nvidia.com/gpu", "count": 1},
        }
        hardware = BenchmarkHardwareConfig.model_validate(direct)
        assert hardware.model_dump(exclude_none=True) == direct

        with pytest.raises(
            ValidationError, match="hardware_profile_name cannot be combined"
        ):
            BenchmarkHardwareConfig.model_validate(
                {"hardware_profile_name": "small", **direct}
            )
        with pytest.raises(ValidationError, match="gpu and dra are mutually exclusive"):
            BenchmarkHardwareConfig.model_validate(
                {**direct, "dra": {"resource_claim_template": "gpu-template"}}
            )

    @pytest.mark.parametrize("field_name", ["lambda", "lambda_"])
    def test_lambda_input_alias_and_wire_serialization(self, field_name: str) -> None:
        interval = ConfidenceInterval.model_validate(
            {"lower": 0.1, "upper": 0.9, field_name: 0.5}
        )
        assert interval.lambda_ == 0.5
        assert interval.model_dump(exclude_none=True) == {
            "lower": 0.1,
            "upper": 0.9,
            "lambda": 0.5,
        }

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_interval_values_are_rejected(self, value: float) -> None:
        with pytest.raises(ValidationError) as error:
            ConfidenceInterval(lower=value, upper=0.9)
        assert error.value.errors()[0]["type"] == "finite_number"

    @pytest.mark.parametrize(
        "result_kind", ["absent", "empty", "benchmarks", "aggregate", "both"]
    )
    def test_resource_response_shapes(
        self, post_processing_config_data: dict[str, Any], result_kind: str
    ) -> None:
        interval = {"lower": 0.7, "upper": 0.9, "lambda": 0.5, "future_metadata": True}
        results: dict[str, Any] = {}
        if result_kind in {"benchmarks", "both"}:
            results["benchmarks"] = [
                {
                    "id": "arc",
                    "provider_id": "lm_eval",
                    "benchmark_index": 0,
                    "confidence_interval": interval,
                }
            ]
        if result_kind in {"aggregate", "both"}:
            results["confidence_interval"] = interval
        payload = {
            "resource": {"id": "pp-123"},
            "operations": {
                "confidence_interval": {
                    **post_processing_config_data,
                    "future_option": True,
                }
            },
            "hardware_config": {
                "cpu": {"request": "1", "future_quantity": True},
                "future_hardware": True,
            },
            "status": {
                "state": "completed",
                "completed_at": "2026-01-01T00:00:00Z",
                "progress": 1,
            },
            "future_field": True,
        }
        if result_kind != "absent":
            payload["results"] = {**results, "future_result": True}
        response = PostProcessingResource.model_validate(payload)

        assert response.resource.id == "pp-123"
        assert response.status.completed_at == datetime(2026, 1, 1, tzinfo=UTC)
        assert response.model_extra is None
        if result_kind == "absent":
            assert response.results is None
        else:
            assert response.results is not None
            assert (response.results.benchmarks is not None) == (
                "benchmarks" in results
            )
            assert (response.results.confidence_interval is not None) == (
                "confidence_interval" in results
            )
            if response.results.benchmarks is not None:
                benchmark = response.results.benchmarks[0]
                assert (
                    benchmark.id,
                    benchmark.provider_id,
                    benchmark.benchmark_index,
                ) == ("arc", "lm_eval", 0)
                assert benchmark.confidence_interval.lower == 0.7
            if response.results.confidence_interval is not None:
                assert response.results.confidence_interval.lambda_ == 0.5

    def test_failed_response_without_results(
        self, post_processing_config_data: dict[str, Any]
    ) -> None:
        response = PostProcessingResource.model_validate(
            {
                "resource": {"id": "pp-123"},
                "operations": {"confidence_interval": post_processing_config_data},
                "status": {
                    "state": "failed",
                    "error_message": {
                        "message": "Artifact missing",
                        "message_code": "not_found",
                    },
                },
            }
        )

        assert response.status.state.value == "failed"
        assert response.status.error_message is not None
        assert response.status.error_message.message_code == "not_found"
        assert response.results is None


@pytest.mark.parametrize("name", ["Invalid_Name!", "a" * 64, ""])
def test_hardware_names_are_validated_by_server(name: str) -> None:
    hardware = BenchmarkHardwareConfig(hardware_profile_name=name)
    dra = HardwareDRAConfig(resource_claim_template=name, claim_name=name)

    assert hardware.model_dump(exclude_none=True) == {"hardware_profile_name": name}
    assert dra.model_dump(exclude_none=True) == {
        "resource_claim_template": name,
        "claim_name": name,
    }


class TestPostProcessingEmptyStringSemantics:
    """Test server-compatible handling of optional empty strings."""

    def test_empty_hardware_strings_are_preserved(self) -> None:
        hardware = BenchmarkHardwareConfig(
            hardware_profile_name="",
            cpu=HardwareResourceQuantity(request="1"),
        )
        dra = HardwareDRAConfig(
            resource_claim_template="claim-template",
            claim_name="",
        )

        assert hardware.hardware_profile_name == ""
        assert hardware.model_dump(exclude_none=True) == {
            "hardware_profile_name": "",
            "cpu": {"request": "1"},
        }
        assert dra.claim_name == ""
        assert dra.model_dump(exclude_none=True) == {
            "resource_claim_template": "claim-template",
            "claim_name": "",
        }

    def test_empty_digest_is_preserved(self) -> None:
        data_ref = OCIDataRef(
            coordinates=OCICoordinates(
                oci_host="quay.io", oci_repository="org/results"
            ),
            digest="",
            artifact_path="results.json",
        )

        assert data_ref.digest == ""
        assert data_ref.model_dump(exclude_none=True) == {
            "coordinates": {
                "oci_host": "quay.io",
                "oci_repository": "org/results",
                "annotations": {},
            },
            "artifact_path": "results.json",
            "digest": "",
        }

    def test_empty_name_is_preserved(self) -> None:
        common = PostProcessingCommon(name="")

        assert common.name == ""
        # Ensure their submission path uses exclude_none=True.
        assert common.model_dump(exclude_none=True) == {"name": ""}

    def test_nonempty_invalid_values_still_fail_validation(self) -> None:
        with pytest.raises(ValidationError):
            OCIDataRef(
                coordinates=OCICoordinates(
                    oci_host="quay.io", oci_repository="org/results"
                ),
                digest="not-a-sha256-digest",
                artifact_path="results.json",
            )


class TestBenchmarkResultMetricsSchema:
    """Tests for metrics_schema field on BenchmarkResult."""

    def test_metrics_schema_defaults_to_none(self) -> None:
        result = BenchmarkResult(id="mmlu", provider_id="lm_eval")
        assert result.metrics_schema is None

    def test_metrics_schema_round_trip(self) -> None:
        result = BenchmarkResult(
            id="mmlu",
            provider_id="lm_eval",
            metrics={"accuracy": 0.85, "f1": 0.82},
            metrics_schema=[
                MetricSchema(name="accuracy", type=ResultType.NUMERIC),
                MetricSchema(name="f1", type=ResultType.NUMERIC),
            ],
        )
        assert result.metrics_schema is not None
        assert len(result.metrics_schema) == 2
        assert result.metrics_schema[0].name == "accuracy"
        assert result.metrics_schema[1].type == ResultType.NUMERIC

    def test_metrics_schema_deserialized_from_server_response(self) -> None:
        data: dict[str, Any] = {
            "id": "mmlu",
            "provider_id": "lm_eval",
            "metrics": {"accuracy": 0.85},
            "metrics_schema": [{"name": "accuracy", "type": "numeric"}],
        }
        result = BenchmarkResult.model_validate(data)
        assert result.metrics_schema is not None
        assert result.metrics_schema[0].type == ResultType.NUMERIC

    def test_metrics_schema_absent_in_server_response_is_none(self) -> None:
        data: dict[str, Any] = {
            "id": "mmlu",
            "provider_id": "lm_eval",
            "metrics": {"accuracy": 0.85},
        }
        result = BenchmarkResult.model_validate(data)
        assert result.metrics_schema is None
