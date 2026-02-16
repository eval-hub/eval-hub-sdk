"""Unit tests for OCI persister."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from evalhub.adapter.models import OCIArtifactResult, OCIArtifactSpec
from evalhub.adapter.oci import OCIArtifactContext, OCIArtifactPersister
from evalhub.adapter.oci.persister import default_tag_hasher
from evalhub.models.api import OCICoordinates


class TestOCIArtifactPersisterInit:
    """Tests for OCIArtifactPersister initialization."""

    def test_persister_initialization(self) -> None:
        """Test persister can be initialized with required args."""
        ctx = OCIArtifactContext(
            job_id="job-123", provider_id="lm-eval", benchmark_id="mmlu"
        )
        persister = OCIArtifactPersister(context=ctx)
        assert persister.context.job_id == "job-123"
        assert persister.context.provider_id == "lm-eval"
        assert persister.context.benchmark_id == "mmlu"
        assert persister.oci_auth_config_path is None
        assert persister.oci_insecure is False

    def test_persister_with_all_options(self, tmp_path: Path) -> None:
        """Test persister with all optional args."""
        auth_path = tmp_path / "auth.json"
        auth_path.write_text("{}")
        ctx = OCIArtifactContext(
            job_id="job-456", provider_id="lm-eval", benchmark_id="hellaswag"
        )
        persister = OCIArtifactPersister(
            context=ctx,
            oci_auth_config_path=auth_path,
            oci_insecure=True,
        )
        assert persister.context.job_id == "job-456"
        assert persister.context.benchmark_id == "hellaswag"
        assert persister.context.provider_id == "lm-eval"
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
        ctx = OCIArtifactContext(
            job_id="job-123", provider_id="lm-eval", benchmark_id="mmlu"
        )
        persister = OCIArtifactPersister(context=ctx)

        spec = OCIArtifactSpec(
            files_path=Path("/nonexistent/path"),
            coordinates=OCICoordinates(oci_host="ghcr.io", oci_repository="org/repo"),
        )

        with pytest.raises(ValueError, match="does not exist"):
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

        ctx = OCIArtifactContext(
            job_id="job-123", provider_id="lm-eval", benchmark_id="mmlu"
        )
        persister = OCIArtifactPersister(context=ctx)

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

        ctx = OCIArtifactContext(
            job_id="my-job", provider_id="lm-eval", benchmark_id="mmlu"
        )
        persister = OCIArtifactPersister(context=ctx)

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host="ghcr.io",
                oci_repository="org/repo",
                # oci_tag not set — should use hash-based tag
            ),
        )

        result = persister.persist(spec)

        expected_hash = hashlib.sha256(b"my-job:lm-eval:mmlu").hexdigest()
        expected_tag = f"evalhub-{expected_hash}"
        assert result.reference == f"ghcr.io/org/repo:{expected_tag}@sha256:def456"

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

        ctx = OCIArtifactContext(
            job_id="job-123", provider_id="lm-eval", benchmark_id="mmlu"
        )
        persister = OCIArtifactPersister(context=ctx)

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host="ghcr.io",
                oci_repository="org/repo",
            ),
        )

        with pytest.raises(RuntimeError, match="Failed to push OCI artifact"):
            persister.persist(spec)


class TestDefaultTagHasher:
    """Tests for the default_tag_hasher and tag generation."""

    def test_hash_is_deterministic(self) -> None:
        """Same context produces the same hash."""
        ctx = OCIArtifactContext(
            job_id="job-1", provider_id="lm-eval", benchmark_id="mmlu"
        )
        assert default_tag_hasher(ctx) == default_tag_hasher(ctx)

    def test_different_provider_gives_different_hash(self) -> None:
        """Different provider_id produces different hash."""
        ctx_a = OCIArtifactContext(
            job_id="job-1", provider_id="lm-eval", benchmark_id="mmlu"
        )
        ctx_b = OCIArtifactContext(
            job_id="job-1", provider_id="other-provider", benchmark_id="mmlu"
        )
        assert default_tag_hasher(ctx_a) != default_tag_hasher(ctx_b)

    def test_different_benchmark_gives_different_hash(self) -> None:
        """Different benchmark_id produces different hash."""
        ctx_a = OCIArtifactContext(
            job_id="job-1", provider_id="lm-eval", benchmark_id="mmlu"
        )
        ctx_b = OCIArtifactContext(
            job_id="job-1", provider_id="lm-eval", benchmark_id="hellaswag"
        )
        assert default_tag_hasher(ctx_a) != default_tag_hasher(ctx_b)

    def test_none_provider_handled(self) -> None:
        """provider_id=None produces a valid hash."""
        ctx = OCIArtifactContext(job_id="job-1", provider_id=None, benchmark_id="mmlu")
        result = default_tag_hasher(ctx)
        assert len(result) == 64  # SHA256 hex digest length
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_is_valid_oci_tag_chars(self) -> None:
        """Hash output only contains OCI-tag-safe characters."""
        ctx = OCIArtifactContext(
            job_id="job-1", provider_id="lm-eval", benchmark_id="mmlu"
        )
        tag = "evalhub-" + default_tag_hasher(ctx)
        assert len(tag) <= 128
        import re

        assert re.fullmatch(r"[a-zA-Z0-9_.-]+", tag)

    def test_custom_tag_hasher_is_used(self) -> None:
        """Custom tag_hasher callable overrides the default."""
        ctx = OCIArtifactContext(
            job_id="job-1", provider_id="lm-eval", benchmark_id="mmlu"
        )

        def custom_hasher(_ctx: OCIArtifactContext) -> str:
            return "custom-hash-value"

        persister = OCIArtifactPersister(context=ctx, tag_hasher=custom_hasher)
        assert persister.tag_hasher is custom_hasher
