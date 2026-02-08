"""Unit tests for OCI persister."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from evalhub.adapter.models import OCIArtifactResult, OCIArtifactSpec
from evalhub.adapter.oci import OCIArtifactPersister
from evalhub.models.api import (
    BenchmarkConfig,
    EvaluationJob,
    EvaluationJobFilesLocation,
    EvaluationJobResource,
    EvaluationJobStatus,
    JobStatus,
    ModelConfig,
    OCICoordinate,
)


class TestOCIArtifactPersisterBridge:
    """Tests for OCIArtifactPersister bridge adapter."""

    def test_persister_initialization(self) -> None:
        """Test persister can be initialized."""
        persister = OCIArtifactPersister(
            registry_url="ghcr.io",
            username="user",
            password="token",
            insecure=False,
        )
        assert persister.registry_url == "ghcr.io"

    def test_persister_default_registry(self) -> None:
        """Test persister uses default registry if not provided."""
        persister = OCIArtifactPersister()
        assert persister.registry_url == "localhost:5000"

    def test_persist_converts_spec_and_returns_result(self, tmp_path: Path) -> None:
        """Test persist converts new adapter spec to legacy format and returns result."""
        # Create test files
        test_dir = tmp_path / "test_job"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("test content 1")
        (test_dir / "file2.txt").write_text("test content 2")

        persister = OCIArtifactPersister(registry_url="ghcr.io")

        spec = OCIArtifactSpec(
            files=[test_dir / "file1.txt", test_dir / "file2.txt"],
            base_path=test_dir,
            id="test_job",
            benchmark_id="mmlu",
            model_name="test_model",
            title="Test Results",
        )

        result = persister.persist(spec)

        # Verify result format
        assert isinstance(result, OCIArtifactResult)
        assert result.digest.startswith("sha256:")
        assert "ghcr.io/eval-results/mmlu:test_job" in result.reference
        assert result.size_bytes > 0

    def test_persist_with_empty_directory(self, tmp_path: Path) -> None:
        """Test persist with empty directory."""
        test_dir = tmp_path / "empty"
        test_dir.mkdir()

        persister = OCIArtifactPersister()

        spec = OCIArtifactSpec(
            files=[],
            base_path=test_dir,
            id="test_job",
            benchmark_id="test",
            model_name="model",
        )

        result = persister.persist(spec)

        assert result.digest.startswith("sha256:")
        assert result.size_bytes == 0  # No files

    def test_persist_with_nested_directory(self, tmp_path: Path) -> None:
        """Test persist counts files in nested directories."""
        test_dir = tmp_path / "nested"
        test_dir.mkdir()

        # Create nested structure
        (test_dir / "result.json").write_text('{"score": 0.95}')
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("content")
        (subdir / "file3.json").write_text("{}")

        persister = OCIArtifactPersister()

        spec = OCIArtifactSpec(
            files=[test_dir / "result.json"],
            base_path=test_dir,
            id="test_job",
            benchmark_id="test",
            model_name="model",
        )

        result = persister.persist(spec)

        # Size should reflect file count (1024 bytes per file placeholder)
        assert result.size_bytes == 3 * 1024  # 3 files total in directory

    def test_persist_with_custom_registry(self, tmp_path: Path) -> None:
        """Test persist with custom registry URL."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        persister = OCIArtifactPersister(registry_url="custom.registry.io")

        spec = OCIArtifactSpec(
            files=[test_dir / "file.txt"],
            base_path=test_dir,
            id="job-123",
            benchmark_id="gsm8k",
            model_name="model",
        )

        result = persister.persist(spec)

        assert "custom.registry.io/eval-results/gsm8k:job-123" in result.reference


@pytest.mark.asyncio
class TestOriginalOCIPersister:
    """Tests for the original OCI persister (placeholder implementation)."""

    async def test_persister_no_op_returns_response(self, tmp_path: Path) -> None:
        """Test no-op persister returns valid response."""
        from evalhub.adapter.oci.persister import (
            OCIArtifactPersister as OriginalPersister,
        )

        # Create test directory with files
        test_dir = tmp_path / "test_job"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("test content 1")
        (test_dir / "file2.txt").write_text("test content 2")

        persister = OriginalPersister()
        files_location = EvaluationJobFilesLocation(id="test_job", path=str(test_dir))

        now = datetime.now(UTC)
        job = EvaluationJob(
            resource=EvaluationJobResource(
                id="test_job",
                tenant="default",
                created_at=now,
                updated_at=now,
            ),
            status=EvaluationJobStatus(state=JobStatus.COMPLETED),
            model=ModelConfig(url="http://localhost:8000/v1", name="test_model"),
            benchmarks=[
                BenchmarkConfig(
                    id="test_benchmark",
                    provider_id="test_provider",
                    parameters={},
                )
            ],
        )

        coordinate = OCICoordinate(oci_ref="ghcr.io/test/repo:latest")

        response = await persister.persist(
            files_location=files_location,
            coordinate=coordinate,
            job=job,
        )

        assert response.id == "test_job"
        assert response.files_count == 2
        assert response.digest.startswith("sha256:")
        assert response.oci_ref.startswith("ghcr.io/test/repo:latest@sha256:")
        assert response.metadata["placeholder"] is True
        assert "not yet implemented" in response.metadata["message"]

    async def test_persister_empty_directory(self, tmp_path: Path) -> None:
        """Test persister with empty directory."""
        from evalhub.adapter.oci.persister import (
            OCIArtifactPersister as OriginalPersister,
        )

        test_dir = tmp_path / "empty"
        test_dir.mkdir()

        persister = OriginalPersister()
        files_location = EvaluationJobFilesLocation(id="test_job", path=str(test_dir))

        now = datetime.now(UTC)
        job = EvaluationJob(
            resource=EvaluationJobResource(
                id="test_job",
                tenant="default",
                created_at=now,
                updated_at=now,
            ),
            status=EvaluationJobStatus(state=JobStatus.COMPLETED),
            model=ModelConfig(url="http://localhost:8000/v1", name="test_model"),
            benchmarks=[
                BenchmarkConfig(id="test", provider_id="test_provider", parameters={})
            ],
        )

        coordinate = OCICoordinate(oci_ref="ghcr.io/test/repo:latest")

        response = await persister.persist(
            files_location=files_location,
            coordinate=coordinate,
            job=job,
        )

        assert response.files_count == 0
        assert response.digest.startswith("sha256:")

    async def test_persister_nested_directory_structure(self, tmp_path: Path) -> None:
        """Test persister counts files in nested directories."""
        from evalhub.adapter.oci.persister import (
            OCIArtifactPersister as OriginalPersister,
        )

        test_dir = tmp_path / "nested"
        test_dir.mkdir()

        # Create nested structure
        test_file = test_dir / "result.json"
        test_file.write_text('{"score": 0.95}')
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("content")
        (subdir / "file3.json").write_text("{}")

        persister = OriginalPersister()
        files_location = EvaluationJobFilesLocation(id="test_job", path=str(test_dir))

        now = datetime.now(UTC)
        job = EvaluationJob(
            resource=EvaluationJobResource(
                id="test_job",
                tenant="default",
                created_at=now,
                updated_at=now,
            ),
            status=EvaluationJobStatus(state=JobStatus.COMPLETED),
            model=ModelConfig(url="http://localhost:8000/v1", name="test_model"),
            benchmarks=[
                BenchmarkConfig(id="test", provider_id="test_provider", parameters={})
            ],
        )

        coordinate = OCICoordinate(oci_ref="ghcr.io/test/repo:latest")

        response = await persister.persist(
            files_location=files_location,
            coordinate=coordinate,
            job=job,
        )

        assert response.files_count == 3
        assert response.digest == "sha256:" + "0" * 64  # Placeholder digest
