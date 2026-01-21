"""Unit tests for OCI persister."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from evalhub.adapter.oci.persister import OCIArtifactPersister
from evalhub.models.api import (
    EvaluationJob,
    EvaluationJobFilesLocation,
    EvaluationRequest,
    JobStatus,
    ModelConfig,
    OCICoordinate,
)


@pytest.mark.asyncio
class TestOCIArtifactPersister:
    """Tests for OCIArtifactPersister (no-op implementation coverage, will be removed later)."""

    async def test_persister_no_op_returns_response(self, tmp_path: Path) -> None:
        """Test no-op persister returns valid response."""
        # Create test directory with files
        test_dir = tmp_path / "test_job"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("test content 1")
        (test_dir / "file2.txt").write_text("test content 2")

        persister = OCIArtifactPersister()
        files_location = EvaluationJobFilesLocation(
            job_id="test_job", path=str(test_dir)
        )

        job = EvaluationJob(
            job_id="test_job",
            status=JobStatus.COMPLETED,
            request=EvaluationRequest(
                benchmark_id="test_benchmark",
                model=ModelConfig(name="test_model"),
            ),
            submitted_at=datetime.now(timezone.utc),
        )

        coordinate = OCICoordinate(oci_ref="ghcr.io/test/repo:latest")

        response = await persister.persist(
            files_location=files_location,
            coordinate=coordinate,
            job=job,
        )

        assert response.job_id == "test_job"
        assert response.files_count == 2
        assert response.digest.startswith("sha256:")
        assert response.oci_ref.startswith("ghcr.io/test/repo:latest@sha256:")
        assert response.metadata["placeholder"] is True
        assert "not yet implemented" in response.metadata["message"]

    async def test_persister_empty_directory(self, tmp_path: Path) -> None:
        """Test persister with empty directory."""
        test_dir = tmp_path / "empty"
        test_dir.mkdir()

        persister = OCIArtifactPersister()
        files_location = EvaluationJobFilesLocation(
            job_id="test_job", path=str(test_dir)
        )

        job = EvaluationJob(
            job_id="test_job",
            status=JobStatus.COMPLETED,
            request=EvaluationRequest(
                benchmark_id="test", model=ModelConfig(name="test_model")
            ),
            submitted_at=datetime.now(timezone.utc),
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
        test_dir = tmp_path / "nested"
        test_dir.mkdir()

        # Create nested structure
        test_file = test_dir / "result.json"
        test_file.write_text('{"score": 0.95}')
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("content")
        (subdir / "file3.json").write_text("{}")

        persister = OCIArtifactPersister()
        files_location = EvaluationJobFilesLocation(
            job_id="test_job", path=str(test_dir)
        )

        job = EvaluationJob(
            job_id="test_job",
            status=JobStatus.COMPLETED,
            request=EvaluationRequest(
                benchmark_id="test", model=ModelConfig(name="test_model")
            ),
            submitted_at=datetime.now(timezone.utc),
        )

        coordinate = OCICoordinate(oci_ref="ghcr.io/test/repo:latest")

        response = await persister.persist(
            files_location=files_location,
            coordinate=coordinate,
            job=job,
        )

        assert response.files_count == 3
        assert response.digest == "sha256:" + "0" * 64  # Placeholder digest
