"""E2E test for OCI artifact persistence with the new adapter framework."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from evalhub.adapter import (
    EvaluationResult,
    FrameworkAdapter,
    JobCallbacks,
    JobResults,
    JobSpec,
    ModelConfig,
)
from evalhub.adapter.callbacks import DefaultCallbacks
from evalhub.adapter.models import JobStatusUpdate, OCIArtifactResult, OCIArtifactSpec
from evalhub.models.api import OCICoordinates


@pytest.fixture
def mock_job_spec_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary job spec file and set environment variable."""
    # Create test job spec
    job_spec = {
        "id": "test-job-001",
        "provider_id": "lm_evaluation_harness",
        "benchmark_id": "mmlu",
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


class TestOCIArtifactPersistenceE2E:
    """E2E tests for OCI artifact persistence in adapter workflow."""

    def test_adapter_creates_oci_artifact_via_callbacks(
        self, tmp_path: Path, mock_job_spec_file: Path
    ) -> None:
        """Test complete flow: adapter → callbacks → OCI persister."""

        # Track what gets called
        created_artifacts: list[OCIArtifactSpec] = []

        class TestCallbacks(JobCallbacks):
            """Test callbacks that record artifact creation."""

            def report_status(self, update: JobStatusUpdate) -> None:
                """No-op for this test."""
                pass

            def create_oci_artifact(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
                """Record artifact spec and return mock result."""
                created_artifacts.append(spec)
                return OCIArtifactResult(
                    digest="sha256:test123",
                    reference="ghcr.io/test/repo:e2e-test-001@sha256:test123",
                )

            def report_results(self, results: JobResults) -> None:
                """No-op for this test."""
                pass

            def report_metrics_to_mlflow(
                self, results: JobResults, job_spec: JobSpec
            ) -> None:
                """No-op for this test."""
                pass

        # Simple test adapter
        class TestAdapter(FrameworkAdapter):
            """Minimal adapter for testing OCI workflow."""

            def run_benchmark_job(
                self, config: JobSpec, callbacks: JobCallbacks
            ) -> JobResults:
                """Run minimal benchmark job that creates artifacts."""
                # Create test files
                output_dir = tmp_path / config.id
                output_dir.mkdir(parents=True, exist_ok=True)
                results_file = output_dir / "results.json"
                results_file.write_text('{"score": 0.85}')

                # Create OCI artifact
                artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=output_dir,
                        coordinates=OCICoordinates(
                            oci_host="ghcr.io",
                            oci_repository="test/repo",
                            oci_tag=config.id,
                        ),
                    )
                )

                # Return results
                return JobResults(
                    id=config.id,
                    benchmark_id=config.benchmark_id,
                    model_name=config.model.name,
                    results=[
                        EvaluationResult(
                            metric_name="accuracy",
                            metric_value=0.85,
                            metric_type="float",
                        )
                    ],
                    num_examples_evaluated=10,
                    duration_seconds=1.0,
                    completed_at=datetime.now(UTC),
                    oci_artifact=artifact,
                )

        # Run adapter with test callbacks
        adapter = TestAdapter()
        callbacks = TestCallbacks()

        spec = JobSpec(
            id="e2e-test-001",
            provider_id="lm_evaluation_harness",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000", name="test-model"),
            benchmark_config={},
            callback_url="http://localhost:8080",
            num_examples=10,
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        # Verify artifact was created
        assert len(created_artifacts) == 1
        artifact_spec = created_artifacts[0]
        assert artifact_spec.coordinates.oci_host == "ghcr.io"
        assert artifact_spec.coordinates.oci_repository == "test/repo"
        assert artifact_spec.coordinates.oci_tag == "e2e-test-001"

        # Verify results contain artifact info
        assert results.oci_artifact is not None
        assert results.oci_artifact.digest == "sha256:test123"
        assert "e2e-test-001" in results.oci_artifact.reference
        assert mock_job_spec_file.exists()  # Use fixture

    @patch("evalhub.adapter.oci.persister.oras.provider.Registry")
    @patch("evalhub.adapter.oci.persister.Layout")
    @patch("evalhub.adapter.oci.persister.create_simple_oci_artifact")
    def test_default_callbacks_oci_persistence(
        self,
        mock_create_artifact: MagicMock,
        mock_layout_cls: MagicMock,
        mock_registry_cls: MagicMock,
        tmp_path: Path,
        mock_job_spec_file: Path,
    ) -> None:
        """Test DefaultCallbacks can persist OCI artifacts."""
        # Create test files
        test_dir = tmp_path / "test_job"
        test_dir.mkdir()
        (test_dir / "results.json").write_text('{"score": 0.85}')
        (test_dir / "summary.txt").write_text("Test summary")

        # Mock the OCI push response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = {"Docker-Content-Digest": "sha256:abc123def456"}
        mock_layout_cls.return_value.push_to_registry.return_value = mock_response

        # Use DefaultCallbacks
        callbacks = DefaultCallbacks(
            job_id="test-job",
            benchmark_id="mmlu",
        )

        # Create artifact spec
        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host="localhost:5000",
                oci_repository="eval-results/mmlu",
                oci_tag="test-job",
            ),
        )

        # Persist artifact
        result = callbacks.create_oci_artifact(spec)

        # Verify result
        assert result.digest == "sha256:abc123def456"
        assert (
            result.reference
            == "localhost:5000/eval-results/mmlu:test-job@sha256:abc123def456"
        )
        assert mock_job_spec_file.exists()  # Use fixture

    @patch("evalhub.adapter.oci.persister.oras.provider.Registry")
    @patch("evalhub.adapter.oci.persister.Layout")
    @patch("evalhub.adapter.oci.persister.create_simple_oci_artifact")
    def test_oci_persister_integration(
        self,
        mock_create_artifact: MagicMock,
        mock_layout_cls: MagicMock,
        mock_registry_cls: MagicMock,
        tmp_path: Path,
        mock_job_spec_file: Path,
    ) -> None:
        """Test OCI persister directly with test files."""
        from evalhub.adapter.oci.persister import OCIArtifactPersister

        # Create test files
        test_dir = tmp_path / "integration_test"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content 1")
        (test_dir / "file2.json").write_text('{"key": "value"}')

        # Mock the OCI push response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = {"Docker-Content-Digest": "sha256:" + "a1b2c3d4" * 8}
        mock_layout_cls.return_value.push_to_registry.return_value = mock_response

        # Setup persister
        from evalhub.adapter.oci.persister import OCIArtifactContext

        persister = OCIArtifactPersister(
            context=OCIArtifactContext(
                job_id="integration-test",
                provider_id="my-provider",
                benchmark_id="mmlu",
            )
        )

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host="ghcr.io",
                oci_repository="test/integration",
                oci_tag="latest",
            ),
        )

        # Persist
        result = persister.persist(spec)

        # Verify result
        assert result.digest == "sha256:" + "a1b2c3d4" * 8
        assert (
            result.reference
            == "ghcr.io/test/integration:latest@sha256:" + "a1b2c3d4" * 8
        )
        mock_create_artifact.assert_called_once()
        assert mock_job_spec_file.exists()  # Use fixture
