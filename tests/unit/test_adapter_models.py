"""Unit tests for the simplified adapter models."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from evalhub.adapter import (
    ErrorInfo,
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
        "benchmark_config": {"random_seed": 42},
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
            benchmark_config={"num_few_shot": 5, "random_seed": 42},
            callback_url="http://localhost:8080",
        )

        assert spec.id == "test-job-001"
        assert spec.provider_id == "lm_evaluation_harness"
        assert spec.benchmark_id == "mmlu"
        assert spec.model.name == "test-model"
        assert spec.num_examples == 10
        assert spec.benchmark_config["num_few_shot"] == 5
        assert spec.benchmark_config["random_seed"] == 42

    def test_creating_jobspec_with_minimal_fields(self) -> None:
        """Test creating JobSpec with minimal mandatory fields."""
        spec = JobSpec(
            id="test-job-002",
            provider_id="lm_evaluation_harness",
            benchmark_id="hellaswag",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="model"),
            benchmark_config={},
            callback_url="http://localhost:8080",
        )

        assert spec.id == "test-job-002"
        assert spec.benchmark_id == "hellaswag"
        assert spec.num_examples is None
        assert spec.benchmark_config == {}

    def test_jobspec_with_benchmarkspecific_configuration(self) -> None:
        """Test JobSpec with benchmark-specific configuration."""
        spec = JobSpec(
            id="test-job-003",
            provider_id="lm_evaluation_harness",
            benchmark_id="mmlu",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="model"),
            benchmark_config={"subject": "physics", "difficulty": "hard"},
            callback_url="http://localhost:8080",
        )

        assert spec.benchmark_config == {"subject": "physics", "difficulty": "hard"}

    def test_jobspec_with_custom_tags(self) -> None:
        """Test JobSpec with custom tags in list-of-dicts format (eval-hub/eval-hub#166)."""
        spec = JobSpec(
            id="test-job-004",
            provider_id="lm_evaluation_harness",
            benchmark_id="arc",
            benchmark_index=0,
            model=ModelConfig(url="http://localhost:8000", name="model"),
            benchmark_config={},
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
            benchmark_config={},
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
            "benchmark_config": {},
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
            benchmark_config={},
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
            benchmark_config={},
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
        "benchmark_config": {},
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
