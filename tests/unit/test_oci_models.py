"""Unit tests for OCI persistence models."""

import pytest
from evalhub.models.api import (
    EvaluationJobFilesLocation,
    OCICoordinate,
    PersistResponse,
)
from pydantic import ValidationError


class TestOCICoordinate:
    """Tests for OCICoordinate model."""

    def test_oci_coordinate_required_fields(self) -> None:
        """Test OCICoordinate with required fields only."""
        coord = OCICoordinate(oci_ref="ghcr.io/org/repo:tag")
        assert coord.oci_ref == "ghcr.io/org/repo:tag"
        assert coord.oci_subject is None

    def test_oci_coordinate_with_subject(self) -> None:
        """Test OCICoordinate with optional subject."""
        coord = OCICoordinate(
            oci_ref="ghcr.io/org/repo:tag", oci_subject="subject_value"
        )
        assert coord.oci_ref == "ghcr.io/org/repo:tag"
        assert coord.oci_subject == "subject_value"

    def test_oci_coordinate_missing_required_field(self) -> None:
        """Test OCICoordinate fails without required oci_ref."""
        with pytest.raises(ValidationError) as exc_info:
            OCICoordinate()  # type: ignore[call-arg]
        assert "oci_ref" in str(exc_info.value)

    def test_oci_coordinate_serialization(self) -> None:
        """Test OCICoordinate JSON serialization."""
        coord = OCICoordinate(
            oci_ref="ghcr.io/org/repo:tag", oci_subject="subject_value"
        )
        data = coord.model_dump()
        assert data["oci_ref"] == "ghcr.io/org/repo:tag"
        assert data["oci_subject"] == "subject_value"

    def test_oci_coordinate_deserialization(self) -> None:
        """Test OCICoordinate JSON deserialization."""
        data = {"oci_ref": "ghcr.io/org/repo:tag", "oci_subject": "subject_value"}
        coord = OCICoordinate(**data)
        assert coord.oci_ref == "ghcr.io/org/repo:tag"
        assert coord.oci_subject == "subject_value"


class TestEvaluationJobFilesLocation:
    """Tests for EvaluationJobFilesLocation model."""

    def test_evaluation_job_files_location_empty(self) -> None:
        """Test empty files location."""
        loc = EvaluationJobFilesLocation(job_id="test_job", path=None)
        assert loc.job_id == "test_job"
        assert loc.path is None
        assert loc.metadata == {}

    def test_evaluation_job_files_location_with_path(self) -> None:
        """Test files location with path and metadata."""
        loc = EvaluationJobFilesLocation(
            job_id="test_job",
            path="/tmp/output",
            metadata={"framework": "lighteval", "benchmark": "benchmark_id_value"},
        )
        assert loc.job_id == "test_job"
        assert loc.path == "/tmp/output"
        assert loc.metadata["framework"] == "lighteval"
        assert loc.metadata["benchmark"] == "benchmark_id_value"

    def test_evaluation_job_files_location_defaults(self) -> None:
        """Test default values for optional fields."""
        loc = EvaluationJobFilesLocation(job_id="test_job")
        assert loc.job_id == "test_job"
        assert loc.path is None
        assert loc.metadata == {}

    def test_evaluation_job_files_location_metadata_type(self) -> None:
        """Test metadata must be dict[str, str]."""
        # Valid: string values
        loc = EvaluationJobFilesLocation(
            job_id="test_job", metadata={"key1": "value1", "key2": "value2"}
        )
        assert loc.metadata == {"key1": "value1", "key2": "value2"}

        # Invalid: non-string values should fail validation
        with pytest.raises(ValidationError):
            EvaluationJobFilesLocation(
                job_id="test_job",
                metadata={"key": 123},  # type: ignore[dict-item]
            )

    def test_evaluation_job_files_location_serialization(self) -> None:
        """Test JSON serialization."""
        loc = EvaluationJobFilesLocation(
            job_id="test_job",
            path="/tmp/output",
            metadata={"framework": "lighteval"},
        )
        data = loc.model_dump()
        assert data["job_id"] == "test_job"
        assert data["path"] == "/tmp/output"
        assert data["metadata"]["framework"] == "lighteval"


class TestPersistResponse:
    """Tests for PersistResponse model."""

    def test_persist_response_required_fields(self) -> None:
        """Test PersistResponse with all required fields."""
        response = PersistResponse(
            job_id="test_job",
            oci_ref="ghcr.io/org/repo:tag@sha256:abc123",
            digest="sha256:abc123",
            files_count=42,
        )
        assert response.job_id == "test_job"
        assert response.oci_ref == "ghcr.io/org/repo:tag@sha256:abc123"
        assert response.digest == "sha256:abc123"
        assert response.files_count == 42
        assert response.metadata == {}

    def test_persist_response_with_metadata(self) -> None:
        """Test PersistResponse with metadata."""
        response = PersistResponse(
            job_id="test_job",
            oci_ref="ghcr.io/org/repo:tag@sha256:abc123",
            digest="sha256:abc123",
            files_count=42,
            metadata={"placeholder": True, "message": "Test message"},
        )
        assert response.metadata["placeholder"] is True
        assert response.metadata["message"] == "Test message"

    def test_persist_response_digest_format(self) -> None:
        """Test digest field accepts sha256 format."""
        response = PersistResponse(
            job_id="test_job",
            oci_ref="ghcr.io/org/repo:tag@sha256:" + "0" * 64,
            digest="sha256:" + "0" * 64,
            files_count=0,
        )
        assert response.digest.startswith("sha256:")
        assert len(response.digest) == 71  # "sha256:" (7 chars) + 64 hex chars

    def test_persist_response_zero_files(self) -> None:
        """Test PersistResponse with zero files."""
        response = PersistResponse(
            job_id="test_job",
            oci_ref="ghcr.io/org/repo:tag@sha256:abc123",
            digest="sha256:abc123",
            files_count=0,
        )
        assert response.files_count == 0

    def test_persist_response_missing_required_fields(self) -> None:
        """Test PersistResponse fails without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            PersistResponse(job_id="test_job")  # type: ignore[call-arg]
        error_str = str(exc_info.value)
        assert "oci_ref" in error_str
        assert "digest" in error_str
        assert "files_count" in error_str

    def test_persist_response_serialization(self) -> None:
        """Test JSON serialization."""
        response = PersistResponse(
            job_id="test_job",
            oci_ref="ghcr.io/org/repo:tag@sha256:abc123",
            digest="sha256:abc123",
            files_count=10,
            metadata={"key": "value"},
        )
        data = response.model_dump()
        assert data["job_id"] == "test_job"
        assert data["oci_ref"] == "ghcr.io/org/repo:tag@sha256:abc123"
        assert data["digest"] == "sha256:abc123"
        assert data["files_count"] == 10
        assert data["metadata"]["key"] == "value"
