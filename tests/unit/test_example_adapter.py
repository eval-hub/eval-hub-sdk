"""Unit tests for the example adapter implementation."""

from typing import Any
from unittest.mock import patch

import pytest
from evalhub.adapter import (
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    ModelConfig,
    OCIArtifactResult,
    OCIArtifactSpec,
)

try:
    from evalhub.adapter.examples import ExampleAdapter
except ImportError:
    # For type checking when module isn't installed
    ExampleAdapter = None  # type: ignore


class MockCallbacks(JobCallbacks):
    """Mock callbacks for testing."""

    def __init__(self) -> None:
        self.status_updates: list[JobStatusUpdate] = []
        self.artifacts: list[OCIArtifactSpec] = []
        self.results: list[JobResults] = []

    def report_status(self, update: JobStatusUpdate) -> None:
        """Record status update."""
        self.status_updates.append(update)

    def create_oci_artifact(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
        """Record artifact creation and return mock result."""
        self.artifacts.append(spec)
        return OCIArtifactResult(
            digest=f"sha256:mock-{spec.job_id}",
            reference=f"mock://localhost/{spec.job_id}",
            size_bytes=1024,
        )

    def report_results(self, results: JobResults) -> None:
        """Record results reporting."""
        self.results.append(results)


@pytest.mark.skipif(ExampleAdapter is None, reason="ExampleAdapter not available")
class TestExampleAdapter:
    """Tests for ExampleAdapter."""

    def test_adapter_creation(self) -> None:
        """Test creating an ExampleAdapter instance."""
        adapter = ExampleAdapter()
        assert adapter is not None

    def test_run_benchmark_job_success(self) -> None:
        """Test successful benchmark job execution."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-001",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000/v1", name="test-model"),
            num_examples=10,
            num_few_shot=5,
            random_seed=42,
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        # Verify results
        assert results.job_id == "test-job-001"
        assert results.benchmark_id == "mmlu"
        assert results.model_name == "test-model"
        assert results.num_examples_evaluated > 0
        assert results.duration_seconds > 0
        assert len(results.results) > 0

    def test_status_updates_sent(self) -> None:
        """Test that status updates are sent during execution."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-002",
            benchmark_id="hellaswag",
            model=ModelConfig(url="http://localhost:8000", name="model"),
            num_examples=5,
        )

        adapter.run_benchmark_job(spec, callbacks)

        # Should have multiple status updates
        assert len(callbacks.status_updates) > 0

        # Verify phases
        phases = [u.phase for u in callbacks.status_updates]
        assert JobPhase.INITIALIZING in phases
        assert JobPhase.LOADING_DATA in phases
        assert JobPhase.RUNNING_EVALUATION in phases

    def test_progress_increases(self) -> None:
        """Test that progress increases throughout execution."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-003",
            benchmark_id="arc",
            model=ModelConfig(url="http://localhost:8000", name="model"),
        )

        adapter.run_benchmark_job(spec, callbacks)

        # Get progress values (excluding None)
        progresses = [
            u.progress for u in callbacks.status_updates if u.progress is not None
        ]

        # Progress should increase
        assert len(progresses) > 1
        for i in range(len(progresses) - 1):
            assert progresses[i] <= progresses[i + 1]

    def test_artifacts_created(self) -> None:
        """Test that artifacts are created."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-004",
            benchmark_id="gsm8k",
            model=ModelConfig(url="http://localhost:8000", name="model"),
        )

        with patch("evalhub.adapter.examples.simple_adapter.Path.mkdir"):
            with patch("builtins.open", create=True):
                results = adapter.run_benchmark_job(spec, callbacks)

        # Should have created an artifact
        assert len(callbacks.artifacts) > 0
        artifact_spec = callbacks.artifacts[0]
        assert artifact_spec.job_id == "test-job-004"
        assert artifact_spec.benchmark_id == "gsm8k"

        # Results should reference the artifact
        assert results.oci_artifact is not None
        assert results.oci_artifact.digest.startswith("sha256:")

    def test_num_examples_parameter(self) -> None:
        """Test that num_examples parameter is respected."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-005",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000", name="model"),
            num_examples=20,
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        # Should evaluate exactly the requested number
        assert results.num_examples_evaluated == 20

    def test_benchmark_config_used(self) -> None:
        """Test that benchmark config is passed through."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-006",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000", name="model"),
            benchmark_config={"subject": "physics", "difficulty": "hard"},
        )

        # Should not raise any errors
        results = adapter.run_benchmark_job(spec, callbacks)
        assert results.job_id == "test-job-006"

    def test_error_in_validation(self) -> None:
        """Test that validation errors are raised."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        # Missing required fields
        spec = JobSpec(
            job_id="test-job-error",
            benchmark_id="",  # Invalid empty benchmark_id
            model=ModelConfig(url="http://localhost:8000", name="model"),
        )

        with pytest.raises(ValueError, match="benchmark_id"):
            adapter.run_benchmark_job(spec, callbacks)

    def test_metrics_returned(self) -> None:
        """Test that proper metrics are returned."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-007",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000", name="model"),
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        # Should have some metrics
        assert len(results.results) > 0

        # Each metric should have required fields
        for metric in results.results:
            assert metric.metric_name
            assert metric.metric_value is not None
            assert metric.metric_type

    def test_overall_score_calculated(self) -> None:
        """Test that overall score is calculated."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-008",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000", name="model"),
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        # Should have an overall score
        assert results.overall_score is not None
        assert 0 <= results.overall_score <= 1

    def test_metadata_populated(self) -> None:
        """Test that evaluation metadata is populated."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-009",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000", name="model"),
            num_few_shot=5,
            random_seed=123,
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        # Should have metadata
        assert results.evaluation_metadata is not None
        assert "framework" in results.evaluation_metadata
        assert results.evaluation_metadata.get("num_few_shot") == 5
        assert results.evaluation_metadata.get("random_seed") == 123

    @patch("evalhub.adapter.examples.simple_adapter.time.sleep")
    def test_execution_timing(self, mock_sleep: Any) -> None:
        """Test that execution timing is recorded."""
        # Mock sleep to make test faster
        mock_sleep.return_value = None

        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-010",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000", name="model"),
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        # Duration should be recorded
        assert results.duration_seconds >= 0

        # Completed timestamp should be set
        assert results.completed_at is not None

    def test_different_benchmarks(self) -> None:
        """Test adapter works with different benchmark IDs."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        benchmarks = ["mmlu", "hellaswag", "arc", "gsm8k"]

        for benchmark in benchmarks:
            spec = JobSpec(
                job_id=f"test-{benchmark}",
                benchmark_id=benchmark,
                model=ModelConfig(url="http://localhost:8000", name="model"),
            )

            results = adapter.run_benchmark_job(spec, callbacks)
            assert results.benchmark_id == benchmark

    def test_model_configuration(self) -> None:
        """Test that model configuration is used."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-011",
            benchmark_id="mmlu",
            model=ModelConfig(
                url="http://custom-server:9000/v1",
                name="custom-model",
                provider="vllm",
                parameters={"temperature": 0.1, "max_tokens": 100},
            ),
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        assert results.model_name == "custom-model"

    def test_step_reporting(self) -> None:
        """Test that step information is reported."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="test-job-012",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000", name="model"),
        )

        adapter.run_benchmark_job(spec, callbacks)

        # Check for step information
        updates_with_steps = [
            u for u in callbacks.status_updates if u.total_steps is not None
        ]

        assert len(updates_with_steps) > 0

        for update in updates_with_steps:
            if update.total_steps is not None:
                assert update.total_steps > 0
                if update.completed_steps is not None:
                    assert update.completed_steps <= update.total_steps


class TestExampleAdapterIntegration:
    """Integration tests for ExampleAdapter with realistic scenarios."""

    def test_full_evaluation_flow(self) -> None:
        """Test complete evaluation flow from start to finish."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        spec = JobSpec(
            job_id="integration-test-001",
            benchmark_id="mmlu",
            model=ModelConfig(
                url="http://localhost:8000/v1",
                name="llama-2-7b",
                provider="vllm",
            ),
            num_examples=50,
            num_few_shot=5,
            random_seed=42,
            benchmark_config={"subject": "all"},
            experiment_name="integration-test",
            tags={"env": "test", "framework": "pytest"},
        )

        with patch("evalhub.adapter.examples.simple_adapter.Path.mkdir"):
            with patch("builtins.open", create=True):
                results = adapter.run_benchmark_job(spec, callbacks)

        # Verify all phases were executed
        phases = {u.phase for u in callbacks.status_updates if u.phase}
        expected_phases = {
            JobPhase.INITIALIZING,
            JobPhase.LOADING_DATA,
            JobPhase.RUNNING_EVALUATION,
            JobPhase.POST_PROCESSING,
            JobPhase.PERSISTING_ARTIFACTS,
        }
        assert expected_phases.issubset(phases)

        # Verify status transitions
        statuses = [u.status for u in callbacks.status_updates]
        assert statuses[0] == JobStatus.RUNNING
        assert statuses[-1] == JobStatus.RUNNING

        # Verify results completeness
        assert results.job_id == "integration-test-001"
        assert results.num_examples_evaluated == 50
        assert len(results.results) > 0
        assert results.overall_score is not None
        assert results.duration_seconds > 0
        assert results.oci_artifact is not None

    def test_error_handling_flow(self) -> None:
        """Test that errors are handled properly."""
        adapter = ExampleAdapter()
        callbacks = MockCallbacks()

        # Create spec with invalid data that should cause validation error
        spec = JobSpec(
            job_id="test-job-error",
            benchmark_id="",  # Empty benchmark_id should fail
            model=ModelConfig(url="http://localhost:8000", name="model"),
        )

        with pytest.raises(ValueError, match="benchmark_id"):
            adapter.run_benchmark_job(spec, callbacks)

        # Validation happens early, so there should be a status update
        # from the INITIALIZING phase before the validation error
        assert len(callbacks.status_updates) > 0

    def test_multiple_jobs_sequential(self) -> None:
        """Test running multiple jobs sequentially."""
        adapter = ExampleAdapter()

        for i in range(3):
            callbacks = MockCallbacks()
            spec = JobSpec(
                job_id=f"sequential-{i}",
                benchmark_id="mmlu",
                model=ModelConfig(url="http://localhost:8000", name="model"),
                num_examples=10,
            )

            results = adapter.run_benchmark_job(spec, callbacks)

            assert results.job_id == f"sequential-{i}"
            assert len(callbacks.status_updates) > 0
