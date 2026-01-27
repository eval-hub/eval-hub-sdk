"""E2E test for OCI artifact persistence with the new adapter framework."""

from datetime import UTC, datetime
from pathlib import Path

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


class TestOCIArtifactPersistenceE2E:
    """E2E tests for OCI artifact persistence in adapter workflow."""

    def test_adapter_creates_oci_artifact_via_callbacks(self, tmp_path: Path) -> None:
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
                    reference=f"ghcr.io/test/{spec.job_id}@sha256:test123",
                    size_bytes=1024,
                )

            def report_results(self, results: JobResults) -> None:
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
                output_dir = tmp_path / config.job_id
                output_dir.mkdir(parents=True, exist_ok=True)
                results_file = output_dir / "results.json"
                results_file.write_text('{"score": 0.85}')

                # Create OCI artifact
                artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files=[results_file],
                        base_path=output_dir,
                        job_id=config.job_id,
                        benchmark_id=config.benchmark_id,
                        model_name=config.model.name,
                    )
                )

                # Return results
                return JobResults(
                    job_id=config.job_id,
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
            job_id="e2e-test-001",
            benchmark_id="mmlu",
            model=ModelConfig(url="http://localhost:8000", name="test-model"),
            num_examples=10,
        )

        results = adapter.run_benchmark_job(spec, callbacks)

        # Verify artifact was created
        assert len(created_artifacts) == 1
        artifact_spec = created_artifacts[0]
        assert artifact_spec.job_id == "e2e-test-001"
        assert artifact_spec.benchmark_id == "mmlu"
        assert artifact_spec.model_name == "test-model"

        # Verify results contain artifact info
        assert results.oci_artifact is not None
        assert results.oci_artifact.digest == "sha256:test123"
        assert "e2e-test-001" in results.oci_artifact.reference

    def test_default_callbacks_oci_persistence(self, tmp_path: Path) -> None:
        """Test DefaultCallbacks can persist OCI artifacts."""
        # Create test files
        test_dir = tmp_path / "test_job"
        test_dir.mkdir()
        (test_dir / "results.json").write_text('{"score": 0.85}')
        (test_dir / "summary.txt").write_text("Test summary")

        # Use DefaultCallbacks with mock registry
        callbacks = DefaultCallbacks(registry_url="localhost:5000", insecure=True)

        # Create artifact spec
        spec = OCIArtifactSpec(
            files=[test_dir / "results.json", test_dir / "summary.txt"],
            base_path=test_dir,
            job_id="test-job",
            benchmark_id="mmlu",
            model_name="test-model",
            title="Test Results",
        )

        # Persist artifact (uses placeholder implementation)
        result = callbacks.create_oci_artifact(spec)

        # Verify result
        assert result.digest.startswith("sha256:")
        assert "localhost:5000/eval-results/mmlu:test-job" in result.reference
        assert result.size_bytes > 0

    @pytest.mark.asyncio
    async def test_oci_persister_integration(self, tmp_path: Path) -> None:
        """Test OCI persister directly with test files."""
        from datetime import UTC, datetime

        from evalhub.adapter.oci.persister import (
            OCIArtifactPersister as OriginalPersister,
        )
        from evalhub.models.api import (
            EvaluationJob,
            EvaluationJobFilesLocation,
            EvaluationRequest,
            JobStatus,
            ModelConfig,
            OCICoordinate,
        )

        # Create test files
        test_dir = tmp_path / "integration_test"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content 1")
        (test_dir / "file2.json").write_text('{"key": "value"}')

        # Setup persister
        persister = OriginalPersister()
        files_location = EvaluationJobFilesLocation(
            job_id="integration-test", path=str(test_dir)
        )

        coordinate = OCICoordinate(oci_ref="ghcr.io/test/integration:latest")

        job = EvaluationJob(
            job_id="integration-test",
            status=JobStatus.COMPLETED,
            request=EvaluationRequest(
                benchmark_id="test",
                model=ModelConfig(url="http://localhost:8000", name="model"),
            ),
            submitted_at=datetime.now(UTC),
        )

        # Persist
        response = await persister.persist(files_location, coordinate, job)

        # Verify response
        assert response.id == "integration-test"
        assert response.files_count == 2
        assert response.digest.startswith("sha256:")
        assert response.oci_ref == "ghcr.io/test/integration:latest@sha256:" + "0" * 64
        assert response.metadata["placeholder"] is True
