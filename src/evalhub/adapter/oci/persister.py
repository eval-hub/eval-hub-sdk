"""OCI artifact persistence for evaluation job files."""

import logging
import tempfile
from pathlib import Path

import oras.provider
from olot.oci_artifact import create_simple_oci_artifact
from oras.layout import Layout

from evalhub.adapter.models.job import OCIArtifactResult, OCIArtifactSpec
from evalhub.models.api import OCI_ARTIFACT_TYPE

logger = logging.getLogger(__name__)


class OCIArtifactPersister:
    def __init__(
        self,
        job_id: str,
        oci_auth_config_path: Path | None = None,
        oci_insecure: bool = False,
    ):
        self.job_id = job_id
        self.oci_auth_config_path = oci_auth_config_path
        self.oci_insecure = oci_insecure

    def persist(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
        """Persist OCI artifact.

        Args:
            spec: OCI Artifact specification

        Returns:
            OCIArtifactResult: Persistence result
        """
        if spec.files_path is None:
            raise ValueError("Invoked OCI persistence but files_path is empty.")
        if not spec.files_path.exists():
            raise ValueError(f"the specified path {spec.files_path} does not exist.")

        tag = (
            spec.coordinates.oci_tag
            if spec.coordinates.oci_tag
            else "evalhub-job-" + self.job_id
        )
        oci_ref = (
            spec.coordinates.oci_host
            + "/"
            + spec.coordinates.oci_repository
            + ":"
            + tag
        )

        temp_dir = tempfile.mkdtemp(prefix="oci_layout_")
        temp_path = Path(temp_dir)
        create_simple_oci_artifact(
            source_path=Path(spec.files_path),
            oci_layout_path=temp_path,
            artifact_type=OCI_ARTIFACT_TYPE,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Contents of temp_path (%s):", temp_path)
            for item in temp_path.rglob("*"):
                if item.is_file():
                    logger.debug("  File: %s", item.relative_to(temp_path))
                elif item.is_dir():
                    logger.debug("  Dir:  %s", item.relative_to(temp_path))

        provider = oras.provider.Registry(insecure=self.oci_insecure)
        provider.auth.hostname = spec.coordinates.oci_host
        if self.oci_auth_config_path:
            custom_auth_path = str(self.oci_auth_config_path.absolute())
            logger.debug("custom_auth_path: %s", custom_auth_path)
            provider.auth.load_configs(spec.coordinates.oci_host, [custom_auth_path])
        else:
            provider.auth.load_configs(spec.coordinates.oci_host)
        response = Layout(str(temp_path)).push_to_registry(
            provider=provider,
            target=oci_ref,
            tag="latest",  # note this is oci-layout tag on disk, not destination tag
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to push OCI artifact to {oci_ref}: "
                f"status {response.status_code}, response: {response.text}"
            )
        artifact_digest = response.headers.get("Docker-Content-Digest")
        artifact_reference = oci_ref + "@" + artifact_digest

        return OCIArtifactResult(digest=artifact_digest, reference=artifact_reference)
