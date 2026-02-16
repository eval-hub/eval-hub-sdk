"""E2E tests for OCI artifact persistence against a real local registry.

Requires an OCI registry running on localhost:5001 (no auth, insecure/HTTP).
Start one with: `make start-oci-registry`.
In GitHub/CI, one will be provided by the CI.
"""

from pathlib import Path

import oras.provider
import pytest
from evalhub.adapter.callbacks import DefaultCallbacks
from evalhub.adapter.models import OCIArtifactResult, OCIArtifactSpec
from evalhub.adapter.oci import OCIArtifactContext, OCIArtifactPersister
from evalhub.models.api import (
    OCI_ANNOTATION_BENCHMARK_ID,
    OCI_ANNOTATION_JOB_ID,
    OCI_ANNOTATION_PROVIDER_ID,
    OCICoordinates,
)

OCI_HOST = "localhost:5001"


@pytest.mark.e2e
class TestOCIPersisterAgainstRegistry:
    """Tests for OCIArtifactPersister pushing to a real local registry."""

    def test_persist_single_file(self, tmp_path: Path) -> None:
        """Test persisting a single file to the registry."""
        test_dir = tmp_path / "single"
        test_dir.mkdir()
        (test_dir / "results.json").write_text('{"accuracy": 0.95}')

        persister = OCIArtifactPersister(
            context=OCIArtifactContext(
                job_id="test-single", benchmark_id="mmlu", provider_id="test-provider"
            ),
            oci_insecure=True,
        )

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host=OCI_HOST,
                oci_repository="evalhub-test/single",
                oci_tag="v1",
            ),
        )

        result = persister.persist(spec)

        assert isinstance(result, OCIArtifactResult)
        assert result.digest.startswith("sha256:")
        assert f"{OCI_HOST}/evalhub-test/single:v1@sha256:" in result.reference

        registry = oras.provider.Registry(
            insecure=True
        )  # in E2E test and CI we have localhost:5001
        manifest = registry.get_manifest(
            f"{OCI_HOST}/evalhub-test/single:v1",
            allowed_media_type=["application/vnd.oci.image.manifest.v1+json"],
        )

        annotations = manifest.get("annotations", {})
        assert annotations[OCI_ANNOTATION_JOB_ID] == "test-single"
        assert annotations[OCI_ANNOTATION_PROVIDER_ID] == "test-provider"
        assert annotations[OCI_ANNOTATION_BENCHMARK_ID] == "mmlu"

    def test_persist_multiple_files(self, tmp_path: Path) -> None:
        """Test persisting a directory with multiple files."""
        test_dir = tmp_path / "multi"
        test_dir.mkdir()
        (test_dir / "results.json").write_text('{"accuracy": 0.95}')
        (test_dir / "metrics.csv").write_text("metric,value\naccuracy,0.95\n")
        subdir = test_dir / "logs"
        subdir.mkdir()
        (subdir / "eval.log").write_text("evaluation completed")

        persister = OCIArtifactPersister(
            context=OCIArtifactContext(
                job_id="test-multi", benchmark_id="mmlu", provider_id="test-provider"
            ),
            oci_insecure=True,
        )

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host=OCI_HOST,
                oci_repository="evalhub-test/multi",
                oci_tag="v1",
            ),
        )

        result = persister.persist(spec)

        assert result.digest.startswith("sha256:")
        assert f"{OCI_HOST}/evalhub-test/multi:v1@sha256:" in result.reference

    def test_persist_uses_default_tag_when_none(self, tmp_path: Path) -> None:
        """Test that persist falls back to job_id-based tag when oci_tag is None."""
        test_dir = tmp_path / "notag"
        test_dir.mkdir()
        (test_dir / "data.txt").write_text("some data")

        persister = OCIArtifactPersister(
            context=OCIArtifactContext(
                job_id="job-42", benchmark_id="mmlu", provider_id="test-provider"
            ),
            oci_insecure=True,
        )

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host=OCI_HOST,
                oci_repository="evalhub-test/notag",
            ),
        )

        result = persister.persist(spec)

        assert result.digest.startswith("sha256:")
        assert "evalhub-test/notag:evalhub-" in result.reference
        assert "@sha256:" in result.reference

    def test_persist_overwrites_same_tag(self, tmp_path: Path) -> None:
        """Test that pushing to the same tag twice succeeds (overwrites)."""
        persister = OCIArtifactPersister(
            context=OCIArtifactContext(
                job_id="test-overwrite",
                benchmark_id="mmlu",
                provider_id="test-provider",
            ),
            oci_insecure=True,
        )

        coordinates = OCICoordinates(
            oci_host=OCI_HOST,
            oci_repository="evalhub-test/overwrite",
            oci_tag="latest",
        )

        # First push
        dir1 = tmp_path / "v1"
        dir1.mkdir()
        (dir1 / "result.json").write_text('{"version": 1}')

        result1 = persister.persist(
            OCIArtifactSpec(files_path=dir1, coordinates=coordinates)
        )
        assert result1.digest.startswith("sha256:")

        # Second push to same tag with different content
        dir2 = tmp_path / "v2"
        dir2.mkdir()
        (dir2 / "result.json").write_text('{"version": 2}')

        result2 = persister.persist(
            OCIArtifactSpec(files_path=dir2, coordinates=coordinates)
        )
        assert result2.digest.startswith("sha256:")
        assert result2.digest != result1.digest


@pytest.mark.e2e
class TestDefaultCallbacksAgainstRegistry:
    """Tests for DefaultCallbacks.create_oci_artifact against a real registry."""

    def test_create_oci_artifact_via_default_callbacks(self, tmp_path: Path) -> None:
        """Test the full DefaultCallbacks → OCIArtifactPersister → registry flow."""
        test_dir = tmp_path / "callbacks_test"
        test_dir.mkdir()
        (test_dir / "results.json").write_text('{"score": 0.85}')
        (test_dir / "summary.txt").write_text("Evaluation summary")

        callbacks = DefaultCallbacks(
            job_id="cb-test-001",
            benchmark_id="mmlu",
            oci_insecure=True,
        )

        spec = OCIArtifactSpec(
            files_path=test_dir,
            coordinates=OCICoordinates(
                oci_host=OCI_HOST,
                oci_repository="evalhub-test/callbacks",
                oci_tag="cb-test-001",
            ),
        )

        result = callbacks.create_oci_artifact(spec)

        assert isinstance(result, OCIArtifactResult)
        assert result.digest.startswith("sha256:")
        assert (
            f"{OCI_HOST}/evalhub-test/callbacks:cb-test-001@sha256:" in result.reference
        )
