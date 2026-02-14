"""Unit tests for OCI persister."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from evalhub.adapter.models import OCIArtifactResult, OCIArtifactSpec
from evalhub.adapter.oci import OCIArtifactPersister
from evalhub.models.api import OCICoordinates


class TestOCIArtifactPersisterInit:
    """Tests for OCIArtifactPersister initialization."""

    def test_persister_initialization(self) -> None:
        """Test persister can be initialized with required args."""
        persister = OCIArtifactPersister(job_id="job-123")
        assert persister.job_id == "job-123"
        assert persister.oci_auth_config_path is None
        assert persister.oci_insecure is False

    def test_persister_with_all_options(self, tmp_path: Path) -> None:
        """Test persister with all optional args."""
        auth_path = tmp_path / "auth.json"
        auth_path.write_text("{}")
        persister = OCIArtifactPersister(
            job_id="job-456",
            oci_auth_config_path=auth_path,
            oci_insecure=True,
        )
        assert persister.job_id == "job-456"
        assert persister.oci_auth_config_path == auth_path
        assert persister.oci_insecure is True


class TestOCIArtifactPersisterPersist:
    """Tests for OCIArtifactPersister.persist method."""

    def test_persist_raises_on_none_path(self) -> None:
        """Test that OCIArtifactSpec rejects None as files_path."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OCIArtifactSpec(
                files_path=None,  # type: ignore[arg-type]
                coordinates=OCICoordinates(
                    oci_host="ghcr.io", oci_repository="org/repo"
                ),
            )

    def test_persist_raises_on_nonexistent_path(self) -> None:
        """Test persist raises ValueError when path doesn't exist."""
        persister = OCIArtifactPersister(job_id="job-123")

        spec = OCIArtifactSpec(
            files_path=Path("/nonexistent/path"),
            coordinates=OCICoordinates(oci_host="ghcr.io", oci_repository="org/repo"),
        )

        with pytest.raises(ValueError, match="does not exists"):
            persister.persist(spec)

    @patch("evalhub.adapter.oci.persister.oras.provider.Registry")
    @patch("evalhub.adapter.oci.persister.Layout")
    @patch("evalhub.adapter.oci.persister.create_simple_oci_artifact")
    def test_persist_success(
        self,
        mock_create_artifact: MagicMock,
        mock_layout_cls: MagicMock,
        mock_registry_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test persist creates artifact and pushes to registry."""
        test_dir = tmp_path / "output"
        test_dir.mkdir()
        (test_dir / "result.json").write_text('{"score": 0.95}')

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = {"Docker-Content-Digest": "sha256:abc123"}
        mock_layout_cls.return_value.push_to_registry.return_value = mock_response

        persister = OCIArtifactPersister(job_id="job-123")

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host="ghcr.io",
                oci_repository="org/repo",
                oci_tag="eval-123",
            ),
        )

        result = persister.persist(spec)

        assert isinstance(result, OCIArtifactResult)
        assert result.digest == "sha256:abc123"
        assert result.reference == "ghcr.io/org/repo:eval-123@sha256:abc123"
        mock_create_artifact.assert_called_once()

    @patch("evalhub.adapter.oci.persister.oras.provider.Registry")
    @patch("evalhub.adapter.oci.persister.Layout")
    @patch("evalhub.adapter.oci.persister.create_simple_oci_artifact")
    def test_persist_uses_default_tag_from_job_id(
        self,
        mock_create_artifact: MagicMock,
        mock_layout_cls: MagicMock,
        mock_registry_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test persist uses job_id-based tag when oci_tag is not set."""
        test_dir = tmp_path / "output"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = {"Docker-Content-Digest": "sha256:def456"}
        mock_layout_cls.return_value.push_to_registry.return_value = mock_response

        persister = OCIArtifactPersister(job_id="my-job")

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host="ghcr.io",
                oci_repository="org/repo",
                # oci_tag not set
            ),
        )

        result = persister.persist(spec)

        assert result.reference == "ghcr.io/org/repo:evalhub-job-my-job@sha256:def456"

    @patch("evalhub.adapter.oci.persister.oras.provider.Registry")
    @patch("evalhub.adapter.oci.persister.Layout")
    @patch("evalhub.adapter.oci.persister.create_simple_oci_artifact")
    def test_persist_raises_on_push_failure(
        self,
        mock_create_artifact: MagicMock,
        mock_layout_cls: MagicMock,
        mock_registry_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test persist raises RuntimeError when push fails."""
        test_dir = tmp_path / "output"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_layout_cls.return_value.push_to_registry.return_value = mock_response

        persister = OCIArtifactPersister(job_id="job-123")

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host="ghcr.io",
                oci_repository="org/repo",
            ),
        )

        with pytest.raises(RuntimeError, match="Failed to push OCI artifact"):
            persister.persist(spec)
