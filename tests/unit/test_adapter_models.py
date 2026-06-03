"""Unit tests for the simplified adapter models."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from evalhub.adapter import (
    CapabilityEvalEntry,
    EnvironmentCardMetadata,
    ErrorInfo,
    EvalCardMetadata,
    EvaluationResult,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    ModelConfig,
    OCIArtifactResult,
    OCIArtifactSpec,
    SafetyEvalEntry,
)


@pytest.fixture
def mock_job_spec_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary job spec file and set environment variable."""
    # Create test job spec
    job_spec = {
        "id": "test-job-001",
        "provider_id": "lm_evaluation_harness",
        "benchmark_id": "mmlu",
        "benchmark_index": 0,
        "model": {"url": "http://localhost:8000", "name": "test-model"},
        "num_examples": 10,
        "parameters": {"random_seed": 42},
        "callback_url": "http://localhost:8080",
    }

    # Write to temp file
    spec_file = tmp_path / "job.json"
    spec_file.write_text(json.dumps(job_spec))

    # Set environment variable
    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(spec_file))

    return spec_file


class TestJobSpec:
    """Tests for JobSpec model."""

    def test_job_spec_creation(self) -> None:
        """Test creating a valid JobSpec."""
        spec = JobSpec(
            id="test-job-001",
            provider_id="lm_evaluation_harness",
            benchmark_id="mmlu",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="test-model"),
            num_examples=10,
            parameters={"num_few_shot": 5, "random_seed": 42},
            callback_url="http://localhost:8080",
        )

        assert spec.id == "test-job-001"
        assert spec.provider_id == "lm_evaluation_harness"
        assert spec.benchmark_id == "mmlu"
        assert spec.model.name == "test-model"
        assert spec.num_examples == 10
        assert spec.parameters["num_few_shot"] == 5
        assert spec.parameters["random_seed"] == 42

    def test_creating_jobspec_with_minimal_fields(self) -> None:
        """Test creating JobSpec with minimal mandatory fields."""
        spec = JobSpec(
            id="test-job-002",
            provider_id="lm_evaluation_harness",
            benchmark_id="hellaswag",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="model"),
            parameters={},
            callback_url="http://localhost:8080",
        )

        assert spec.id == "test-job-002"
        assert spec.benchmark_id == "hellaswag"
        assert spec.num_examples is None
        assert spec.parameters == {}

    def test_jobspec_with_benchmarkspecific_configuration(self) -> None:
        """Test JobSpec with benchmark-specific configuration."""
        spec = JobSpec(
            id="test-job-003",
            provider_id="lm_evaluation_harness",
            benchmark_id="mmlu",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="model"),
            parameters={"subject": "physics", "difficulty": "hard"},
            callback_url="http://localhost:8080",
        )

        assert spec.parameters == {"subject": "physics", "difficulty": "hard"}

    def test_jobspec_with_custom_tags(self) -> None:
        """Test JobSpec with custom tags in list-of-dicts format (eval-hub/eval-hub#166)."""
        spec = JobSpec(
            id="test-job-004",
            provider_id="lm_evaluation_harness",
            benchmark_id="arc",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="model"),
            parameters={},
            callback_url="http://localhost:8080",
            tags=[
                {"key": "env", "value": "test"},
                {"key": "developer", "value": "alice"},
            ],
        )

        assert spec.tags == [
            {"key": "env", "value": "test"},
            {"key": "developer", "value": "alice"},
        ]

    def test_jobspec_default_tags_is_empty_list(self) -> None:
        """Test that tags default to an empty list."""
        spec = JobSpec(
            id="test-job-004b",
            provider_id="lm_evaluation_harness",
            benchmark_id="arc",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="model"),
            parameters={},
            callback_url="http://localhost:8080",
        )

        assert spec.tags == []

    def test_jobspec_from_file_with_list_tags(self, tmp_path: Path) -> None:
        """Test loading JobSpec from JSON file with new list-of-dicts tags format."""
        job_spec = {
            "id": "test-job-005",
            "provider_id": "lm_evaluation_harness",
            "benchmark_id": "mmlu",
            "benchmark_index": 0,
            "model": {"url": "http://localhost:8000", "name": "test-model"},
            "parameters": {},
            "callback_url": "http://localhost:8080",
            "tags": [
                {"key": "team", "value": "ml-platform"},
                {"key": "env", "value": "production"},
            ],
        }

        spec_file = tmp_path / "job.json"
        spec_file.write_text(json.dumps(job_spec))

        spec = JobSpec.from_file(spec_file)

        assert spec.tags == [
            {"key": "team", "value": "ml-platform"},
            {"key": "env", "value": "production"},
        ]

    def test_jobspec_can_be_serialized_to_json(self) -> None:
        """Test JobSpec can be serialized to JSON."""
        spec = JobSpec(
            id="test-job-005",
            provider_id="lm_evaluation_harness",
            benchmark_id="gsm8k",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="model"),
            parameters={},
            callback_url="http://localhost:8080",
            num_examples=50,
        )

        json_data = spec.model_dump()

        assert json_data["id"] == "test-job-005"
        assert json_data["benchmark_id"] == "gsm8k"
        assert json_data["num_examples"] == 50

        # Can recreate from JSON
        spec_2 = JobSpec(**json_data)
        assert spec_2.id == spec.id


class TestJobStatusUpdate:
    """Tests for JobStatusUpdate model."""

    def test_creating_a_status_update(self) -> None:
        """Test creating a status update."""
        update = JobStatusUpdate(
            status=JobStatus.RUNNING,
            phase=JobPhase.RUNNING_EVALUATION,
            progress=0.5,
            message=MessageInfo(
                message="Evaluating examples",
                message_code="status_update",
            ),
        )

        assert update.status == JobStatus.RUNNING
        assert update.phase == JobPhase.RUNNING_EVALUATION
        assert update.progress == 0.5
        assert update.message is not None
        assert update.message.message == "Evaluating examples"

    def test_status_update_with_only_required_fields(self) -> None:
        """Test status update with only required fields."""
        update = JobStatusUpdate(status=JobStatus.PENDING)

        assert update.status == JobStatus.PENDING
        assert update.phase is None
        assert update.progress is None
        assert update.message is not None
        assert update.message.message == "Status update"
        assert update.message.message_code == "status_update"

    def test_status_update_with_step_information(self) -> None:
        """Test status update with step information."""
        update = JobStatusUpdate(
            status=JobStatus.RUNNING,
            current_step="Processing batch 5",
            total_steps=10,
            completed_steps=5,
        )

        assert update.current_step == "Processing batch 5"
        assert update.total_steps == 10
        assert update.completed_steps == 5

    def test_status_update_with_error_information(self) -> None:
        """Test status update with error information."""
        update = JobStatusUpdate(
            status=JobStatus.FAILED,
            error=ErrorInfo(
                message="Model server unreachable",
                message_code="model_server_unreachable",
            ),
            error_details={"retry_count": 3},
        )

        assert update.status == JobStatus.FAILED
        assert update.error is not None
        assert update.error.message == "Model server unreachable"
        assert update.error.message_code == "model_server_unreachable"
        assert update.error_details is not None
        assert update.error_details["retry_count"] == 3

    def test_status_update_with_error_message(self) -> None:
        """Test status update with new error_message field."""
        update = JobStatusUpdate(
            status=JobStatus.FAILED,
            error_message=MessageInfo(
                message="Model server unreachable",
                message_code="model_server_unreachable",
            ),
        )

        assert update.error_message is not None
        assert update.error_message.message == "Model server unreachable"
        assert update.error_message.message_code == "model_server_unreachable"
        assert update.error is None

    def test_status_update_with_warning_message(self) -> None:
        """Test status update with warning_message field."""
        update = JobStatusUpdate(
            status=JobStatus.RUNNING,
            warning_message=MessageInfo(
                message="Slow model response",
                message_code="slow_response",
            ),
        )

        assert update.warning_message is not None
        assert update.warning_message.message == "Slow model response"
        assert update.warning_message.message_code == "slow_response"

    def test_resolved_error_prefers_error_message_over_error(self) -> None:
        """Test that resolved_error returns error_message when both are set."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            update = JobStatusUpdate(
                status=JobStatus.FAILED,
                error_message=MessageInfo(
                    message="New error",
                    message_code="new_error",
                ),
                error=ErrorInfo(
                    message="Old error",
                    message_code="old_error",
                ),
            )

        assert update.resolved_error is not None
        assert update.resolved_error.message == "New error"

    def test_resolved_error_falls_back_to_deprecated_error(self) -> None:
        """Test that resolved_error falls back to error when error_message is not set."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            update = JobStatusUpdate(
                status=JobStatus.FAILED,
                error=ErrorInfo(
                    message="Legacy error",
                    message_code="legacy_error",
                ),
            )

        assert update.resolved_error is not None
        assert update.resolved_error.message == "Legacy error"

    def test_resolved_error_returns_none_when_no_error(self) -> None:
        """Test that resolved_error returns None when neither field is set."""
        update = JobStatusUpdate(status=JobStatus.RUNNING)
        assert update.resolved_error is None

    def test_deprecated_error_field_emits_warning(self) -> None:
        """Test that using the deprecated error field emits a DeprecationWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            JobStatusUpdate(
                status=JobStatus.FAILED,
                error=ErrorInfo(
                    message="test",
                    message_code="test",
                ),
            )

        deprecation_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning)
            and "error is deprecated" in str(x.message)
        ]
        assert len(deprecation_warnings) == 1

    def test_deprecated_error_details_field_emits_warning(self) -> None:
        """Test that using the deprecated error_details field emits a DeprecationWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            JobStatusUpdate(
                status=JobStatus.FAILED,
                error_details={"exception_type": "ValueError"},
            )

        deprecation_warnings = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning)
            and "error_details is deprecated" in str(x.message)
        ]
        assert len(deprecation_warnings) == 1

    def test_that_timestamp_is_automatically_set(self) -> None:
        """Test that timestamp is automatically set."""
        update = JobStatusUpdate(status=JobStatus.RUNNING)

        assert update.timestamp is not None
        assert isinstance(update.timestamp, datetime)
        # Should be recent (within last second)
        now = datetime.now(UTC)
        assert (now - update.timestamp).total_seconds() < 1.0


class TestOCIArtifactSpec:
    """Tests for OCIArtifactSpec model."""

    def test_creating_an_oci_artifact_specification(self) -> None:
        """Test creating an OCI artifact specification."""
        from evalhub.models.api import OCICoordinates

        spec = OCIArtifactSpec(
            files_path=Path("/tmp/results"),
            coordinates=OCICoordinates(
                oci_host="ghcr.io",
                oci_repository="org/repo",
                oci_tag="eval-123",
            ),
        )

        assert spec.files_path == Path("/tmp/results")
        assert spec.coordinates.oci_host == "ghcr.io"
        assert spec.coordinates.oci_repository == "org/repo"
        assert spec.coordinates.oci_tag == "eval-123"

    def test_artifact_spec_with_annotations(self) -> None:
        """Test artifact spec with custom annotations on coordinates."""
        from evalhub.models.api import OCICoordinates

        spec = OCIArtifactSpec(
            files_path=Path("/tmp/job-001"),
            coordinates=OCICoordinates(
                oci_host="ghcr.io",
                oci_repository="org/repo",
                annotations={
                    "score": "0.85",
                    "framework": "lm-eval",
                },
            ),
        )

        assert spec.coordinates.annotations["score"] == "0.85"
        assert spec.coordinates.annotations["framework"] == "lm-eval"


class TestOCIArtifactResult:
    """Tests for OCIArtifactResult model."""

    def test_creating_an_oci_artifact_result(self) -> None:
        """Test creating an OCI artifact result."""
        result = OCIArtifactResult(
            digest="sha256:abc123...",
            reference="ghcr.io/eval-hub/results:test@sha256:abc123...",
        )

        assert result.digest == "sha256:abc123..."
        assert result.reference == "ghcr.io/eval-hub/results:test@sha256:abc123..."


class TestJobResults:
    """Tests for JobResults model."""

    def test_creating_job_results(self) -> None:
        """Test creating job results."""
        results = JobResults(
            id="test-job-001",
            benchmark_id="mmlu",
            benchmark_index=0,
            model_name="test-model",
            results=[
                EvaluationResult(
                    metric_name="accuracy", metric_value=0.85, metric_type="float"
                )
            ],
            num_examples_evaluated=100,
            duration_seconds=125.5,
        )

        assert results.id == "test-job-001"
        assert results.benchmark_id == "mmlu"
        assert results.model_name == "test-model"
        assert len(results.results) == 1
        assert results.results[0].metric_name == "accuracy"
        assert results.num_examples_evaluated == 100
        assert results.duration_seconds == 125.5

    def test_job_results_with_overall_score(self) -> None:
        """Test job results with overall score."""
        results = JobResults(
            id="test-job-001",
            benchmark_id="mmlu",
            benchmark_index=0,
            model_name="model",
            results=[],
            num_examples_evaluated=100,
            duration_seconds=60.0,
            overall_score=0.75,
        )

        assert results.overall_score == 0.75

    def test_job_results_with_evaluation_metadata(self) -> None:
        """Test job results with evaluation metadata."""
        results = JobResults(
            id="test-job-001",
            benchmark_id="mmlu",
            benchmark_index=0,
            model_name="model",
            results=[],
            num_examples_evaluated=100,
            duration_seconds=60.0,
            evaluation_metadata={
                "framework": "lm-eval",
                "framework_version": "0.4.0",
                "num_few_shot": 5,
            },
        )

        assert results.evaluation_metadata["framework"] == "lm-eval"
        assert results.evaluation_metadata["num_few_shot"] == 5

    def test_job_results_with_oci_artifact_information(self) -> None:
        """Test job results with OCI artifact information."""
        artifact = OCIArtifactResult(
            digest="sha256:abc123",
            reference="ghcr.io/eval-hub/results:test",
        )

        results = JobResults(
            id="test-job-001",
            benchmark_id="mmlu",
            benchmark_index=0,
            model_name="model",
            results=[],
            num_examples_evaluated=100,
            duration_seconds=60.0,
            oci_artifact=artifact,
        )

        assert results.oci_artifact is not None
        assert results.oci_artifact.digest == "sha256:abc123"

    def test_that_completed_at_is_automatically_set(self) -> None:
        """Test that completed_at is automatically set."""
        results = JobResults(
            id="test-job-001",
            benchmark_id="mmlu",
            benchmark_index=0,
            model_name="model",
            results=[],
            num_examples_evaluated=100,
            duration_seconds=60.0,
        )

        assert results.completed_at is not None
        assert isinstance(results.completed_at, datetime)


class TestJobCallbacks:
    """Tests for JobCallbacks interface."""

    def test_that_jobcallbacks_cannot_be_instantiated_directly(self) -> None:
        """Test that JobCallbacks cannot be instantiated directly."""
        with pytest.raises(TypeError):
            JobCallbacks()  # type: ignore

    def test_implementing_jobcallbacks_with_a_mock(self) -> None:
        """Test implementing JobCallbacks with a mock."""
        from evalhub.models.api import OCICoordinates

        class MockCallbacks(JobCallbacks):
            def __init__(self) -> None:
                self.status_updates: list[JobStatusUpdate] = []
                self.artifacts: list[OCIArtifactSpec] = []
                self.results: list[JobResults] = []

            def report_status(self, update: JobStatusUpdate) -> None:
                self.status_updates.append(update)

            def create_oci_artifact(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
                self.artifacts.append(spec)
                return OCIArtifactResult(
                    digest="sha256:test",
                    reference="test://artifact",
                )

            def report_results(self, results: JobResults) -> None:
                self.results.append(results)

            def report_metrics_to_mlflow(
                self, results: JobResults, job_spec: JobSpec
            ) -> None:
                pass

        # Should be able to instantiate the implementation
        callbacks = MockCallbacks()

        # Test report_status
        update = JobStatusUpdate(status=JobStatus.RUNNING, progress=0.5)
        callbacks.report_status(update)

        assert len(callbacks.status_updates) == 1
        assert callbacks.status_updates[0].status == JobStatus.RUNNING

        # Test create_oci_artifact
        spec = OCIArtifactSpec(
            files_path=Path("/tmp/test"),
            coordinates=OCICoordinates(
                oci_host="ghcr.io",
                oci_repository="org/repo",
            ),
        )
        result = callbacks.create_oci_artifact(spec)

        assert len(callbacks.artifacts) == 1
        assert result.digest == "sha256:test"


class TestFrameworkAdapter:
    """Tests for FrameworkAdapter base class."""

    def test_that_frameworkadapter_cannot_be_instantiated_directly(self) -> None:
        """Test that FrameworkAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FrameworkAdapter()  # type: ignore

    def test_implementing_frameworkadapter(self, mock_job_spec_file: Path) -> None:
        """Test implementing FrameworkAdapter."""

        class TestAdapter(FrameworkAdapter):
            def run_benchmark_job(
                self, config: JobSpec, callbacks: JobCallbacks
            ) -> JobResults:
                # Unused but required by interface
                _ = callbacks
                return JobResults(
                    id=config.id,
                    benchmark_id=config.benchmark_id,
                    benchmark_index=config.benchmark_index,
                    model_name=config.model.name,
                    results=[],
                    num_examples_evaluated=0,
                    duration_seconds=0.0,
                )

        # Should be able to instantiate the implementation
        adapter = TestAdapter()
        assert adapter is not None
        assert mock_job_spec_file.exists()  # Use fixture

    def test_running_a_benchmark_job_through_the_adapter(
        self, mock_job_spec_file: Path
    ) -> None:
        """Test running a benchmark job through the adapter."""

        class TestAdapter(FrameworkAdapter):
            def run_benchmark_job(
                self, config: JobSpec, callbacks: JobCallbacks
            ) -> JobResults:
                # Report initial status
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.INITIALIZING,
                        progress=0.0,
                    )
                )

                # Report progress
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.RUNNING_EVALUATION,
                        progress=0.5,
                    )
                )

                # Return results
                return JobResults(
                    id=config.id,
                    benchmark_id=config.benchmark_id,
                    benchmark_index=config.benchmark_index,
                    model_name=config.model.name,
                    results=[
                        EvaluationResult(
                            metric_name="accuracy",
                            metric_value=0.85,
                            metric_type="float",
                        )
                    ],
                    num_examples_evaluated=100,
                    duration_seconds=60.0,
                )

        # Create mock callbacks
        class MockCallbacks(JobCallbacks):
            def __init__(self) -> None:
                self.status_updates: list[JobStatusUpdate] = []
                self.results: list[JobResults] = []

            def report_status(self, update: JobStatusUpdate) -> None:
                self.status_updates.append(update)

            def create_oci_artifact(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
                # Unused but required by interface
                _ = spec
                return OCIArtifactResult(digest="sha256:test", reference="test")

            def report_results(self, results: JobResults) -> None:
                self.results.append(results)

            def report_metrics_to_mlflow(
                self, results: JobResults, job_spec: JobSpec
            ) -> None:
                pass

        # Run the adapter
        adapter = TestAdapter()
        callbacks = MockCallbacks()
        spec = JobSpec(
            id="test-job-001",
            provider_id="lm_evaluation_harness",
            benchmark_id="mmlu",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="test-model"),
            parameters={},
            callback_url="http://localhost:8080",
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        # Verify results
        assert results.id == "test-job-001"
        assert results.benchmark_id == "mmlu"
        assert len(results.results) == 1
        assert results.results[0].metric_value == 0.85

        # Verify status updates were sent
        assert len(callbacks.status_updates) == 2
        assert callbacks.status_updates[0].phase == JobPhase.INITIALIZING
        assert callbacks.status_updates[1].phase == JobPhase.RUNNING_EVALUATION
        assert mock_job_spec_file.exists()  # Use fixture


class TestLocalJobsBasePath:
    """Tests for FrameworkAdapter.local_jobs_base_path property."""

    _JOB_SPEC_DATA = {
        "id": "job-1",
        "provider_id": "prov",
        "benchmark_id": "bench",
        "benchmark_index": 0,
        "model": {"url": "http://localhost:8000", "name": "m"},
        "parameters": {},
        "callback_url": "http://localhost:8080",
    }

    class _Adapter(FrameworkAdapter):
        def run_benchmark_job(
            self, config: JobSpec, callbacks: JobCallbacks
        ) -> JobResults:
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=[],
                num_examples_evaluated=0,
                duration_seconds=0.0,
            )

    def _write_spec(self, spec_path: Path) -> None:
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(self._JOB_SPEC_DATA))

    def test_returns_parent_of_meta_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that local_jobs_base_path returns the parent of the meta/ directory."""
        base = tmp_path / "job-1" / "0" / "prov" / "bench"
        spec_path = base / "meta" / "job.json"
        self._write_spec(spec_path)

        monkeypatch.setenv("EVALHUB_MODE", "local")
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(spec_path))
        adapter = self._Adapter()

        assert adapter.local_jobs_base_path == base.resolve()

    def test_returns_none_in_k8s_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that local_jobs_base_path returns None when not in local mode."""
        spec_path = tmp_path / "meta" / "job.json"
        self._write_spec(spec_path)

        monkeypatch.setenv("EVALHUB_MODE", "k8s")
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(spec_path))
        adapter = self._Adapter()

        assert adapter.local_jobs_base_path is None

    def test_raises_when_path_does_not_end_with_meta_job_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that an AssertionError is raised if the path doesn't end with meta/job.json."""
        spec_path = tmp_path / "job.json"
        spec_path.write_text(json.dumps(self._JOB_SPEC_DATA))

        monkeypatch.setenv("EVALHUB_MODE", "local")
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(spec_path))
        adapter = self._Adapter()

        with pytest.raises(AssertionError, match="must end with 'meta/job.json'"):
            adapter.local_jobs_base_path

    def test_raises_when_job_spec_path_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that an AssertionError is raised if job_spec_path is None in local mode."""
        spec_path = tmp_path / "meta" / "job.json"
        self._write_spec(spec_path)

        monkeypatch.setenv("EVALHUB_MODE", "local")
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(spec_path))
        adapter = self._Adapter()
        # Manually clear to simulate missing path
        adapter._settings.job_spec_path = None

        with pytest.raises(AssertionError, match="must be set in local mode"):
            adapter.local_jobs_base_path


class TestEvalCardMetadata:
    """Tests for EvalCardMetadata model."""

    def test_creating_evalcard_with_all_sections(self) -> None:
        card = EvalCardMetadata(
            modalities_input=["text"],
            modalities_output=["text"],
            languages_count=1,
            languages=["en"],
            capability_evaluations=[
                CapabilityEvalEntry(
                    ability="knowledge",
                    benchmark="MMLU",
                    metric="exact_match",
                    zero_shot=0.61,
                    alt_prompting=0.71,
                    alt_prompting_description="5-Shot CoT",
                ),
            ],
            safety_evaluations=[
                SafetyEvalEntry(
                    feature="toxicity",
                    benchmark="ToxiGen",
                    metric="pass_rate",
                    zero_shot=0.95,
                ),
            ],
            developer_footnotes="MMLU evaluated on the test split.",
        )

        assert card.modalities_input == ["text"]
        assert card.languages_count == 1
        assert len(card.capability_evaluations) == 1
        assert card.capability_evaluations[0].ability == "knowledge"
        assert card.capability_evaluations[0].alt_prompting_description == "5-Shot CoT"
        assert len(card.safety_evaluations) == 1
        assert card.developer_footnotes is not None

    def test_evalcard_defaults_to_empty(self) -> None:
        card = EvalCardMetadata()

        assert card.modalities_input == []
        assert card.modalities_output == []
        assert card.languages_count is None
        assert card.languages == []
        assert card.capability_evaluations == []
        assert card.safety_evaluations == []
        assert card.developer_footnotes is None

    def test_evalcard_serialization_excludes_none(self) -> None:
        card = EvalCardMetadata(
            modalities_input=["text"],
            languages_count=1,
            languages=["en"],
        )
        dumped = card.model_dump(exclude_none=True)

        assert "modalities_input" in dumped
        assert "developer_footnotes" not in dumped

    def test_evalcard_roundtrip(self) -> None:
        card = EvalCardMetadata(
            modalities_input=["text", "image"],
            modalities_output=["text"],
            languages_count=2,
            languages=["en", "es"],
            capability_evaluations=[
                CapabilityEvalEntry(
                    ability="reasoning", benchmark="BBH", metric="exact_match"
                ),
            ],
        )
        dumped = card.model_dump(exclude_none=True)
        restored = EvalCardMetadata(**dumped)

        assert restored.modalities_input == card.modalities_input
        assert len(restored.capability_evaluations) == 1
        assert restored.capability_evaluations[0].ability == "reasoning"


class TestEnvironmentCardMetadata:
    """Tests for EnvironmentCardMetadata model."""

    def test_creating_envcard_with_hardware_fields(self) -> None:
        card = EnvironmentCardMetadata(
            gpu_model="NVIDIA A100-SXM4-80GB",
            gpu_count=4,
            gpu_driver_version="535.104.05",
            python_version="3.11.5",
            os_info="Linux-5.14.0",
        )

        assert card.gpu_model == "NVIDIA A100-SXM4-80GB"
        assert card.gpu_count == 4
        assert card.python_version == "3.11.5"

    def test_envcard_defaults_to_empty(self) -> None:
        card = EnvironmentCardMetadata()

        assert card.gpu_model is None
        assert card.key_packages == {}
        assert card.k8s_pod_labels == {}
        assert card.model_id is None
        assert card.capture_completeness is None

    def test_envcard_capture_returns_valid_card(self) -> None:
        card = EnvironmentCardMetadata.capture(
            framework_name="test-framework",
            framework_version="1.0.0",
        )

        assert card.python_version is not None
        assert card.os_info is not None
        assert card.framework_name == "test-framework"
        assert card.framework_version == "1.0.0"
        assert card.capture_completeness is not None
        assert 0.0 <= card.capture_completeness <= 1.0

    def test_envcard_capture_with_extra_packages(self) -> None:
        card = EnvironmentCardMetadata.capture(extra_packages=["pydantic"])

        assert "pydantic" in card.key_packages

    def test_envcard_capture_with_custom_kwargs(self) -> None:
        card = EnvironmentCardMetadata.capture(custom_field="custom_value")

        assert card.custom["custom_field"] == "custom_value"

    def test_envcard_completeness_scoring(self) -> None:
        empty_card = EnvironmentCardMetadata()
        assert empty_card._compute_completeness() == 0.0

        partial_card = EnvironmentCardMetadata(
            python_version="3.11.5",
            os_info="Linux",
            framework_name="lm-eval",
        )
        score = partial_card._compute_completeness()
        assert score == round(3 / 26, 2)

    def test_envcard_serialization_roundtrip(self) -> None:
        card = EnvironmentCardMetadata(
            python_version="3.11.5",
            gpu_model="A100",
            gpu_count=2,
            key_packages={"torch": "2.1.0"},
            capture_completeness=0.15,
        )
        dumped = card.model_dump(exclude_none=True)
        restored = EnvironmentCardMetadata(**dumped)

        assert restored.python_version == "3.11.5"
        assert restored.gpu_count == 2
        assert restored.key_packages["torch"] == "2.1.0"
        assert restored.capture_completeness == 0.15


class TestJobResultsWithCards:
    """Tests for JobResults with card fields."""

    def test_job_results_with_evalcard(self) -> None:
        results = JobResults(
            id="test-job",
            benchmark_id="mmlu",
            benchmark_index=0,
            model_name="model",
            results=[],
            num_examples_evaluated=100,
            duration_seconds=60.0,
            eval_card=EvalCardMetadata(
                modalities_input=["text"],
                modalities_output=["text"],
            ),
        )

        assert results.eval_card is not None
        assert results.eval_card.modalities_input == ["text"]

    def test_job_results_with_envcard(self) -> None:
        results = JobResults(
            id="test-job",
            benchmark_id="mmlu",
            benchmark_index=0,
            model_name="model",
            results=[],
            num_examples_evaluated=100,
            duration_seconds=60.0,
            env_card=EnvironmentCardMetadata(python_version="3.11.5"),
        )

        assert results.env_card is not None
        assert results.env_card.python_version == "3.11.5"

    def test_job_results_cards_default_to_none(self) -> None:
        results = JobResults(
            id="test-job",
            benchmark_id="mmlu",
            benchmark_index=0,
            model_name="model",
            results=[],
            num_examples_evaluated=100,
            duration_seconds=60.0,
        )

        assert results.eval_card is None
        assert results.env_card is None
