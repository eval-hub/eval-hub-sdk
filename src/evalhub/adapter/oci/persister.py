"""OCI artifact persistence for evaluation job files."""

import hashlib
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import oras.provider
from olot.oci_artifact import create_simple_oci_artifact
from oras.layout import Layout

from evalhub.adapter.models.job import OCIArtifactResult, OCIArtifactSpec
from evalhub.models.api import (
    OCI_ANNOTATION_BENCHMARK_ID,
    OCI_ANNOTATION_JOB_ID,
    OCI_ANNOTATION_PROVIDER_ID,
    OCI_ARTIFACT_TYPE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCIArtifactContext:
    """Context identifying the evaluation job for OCI artifact tagging and annotations."""

    job_id: str
    benchmark_id: str
    # `provider_id`: while lenient if not existing, the JobSpec shall contain it
    provider_id: str | None = None


def default_tag_hasher(ctx: OCIArtifactContext) -> str:
    """Default tag hasher using SHA256.

    Produces a deterministic hash from the context fields.
    """
    components = f"{ctx.job_id}:{ctx.provider_id or ''}:{ctx.benchmark_id}"
    return hashlib.sha256(components.encode()).hexdigest()


class OCIArtifactPersister:
    def __init__(
        self,
        context: OCIArtifactContext,
        oci_auth_config_path: Path | None = None,
        oci_insecure: bool = False,
        tag_hasher: Callable[[OCIArtifactContext], str] | None = None,
    ):
        self.context = context
        self.oci_auth_config_path = oci_auth_config_path
        self.oci_insecure = oci_insecure
        self.tag_hasher = tag_hasher or default_tag_hasher

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
            else "evalhub-" + self.tag_hasher(self.context)
        )

        default_annotations = {
            OCI_ANNOTATION_JOB_ID: self.context.job_id,
            OCI_ANNOTATION_BENCHMARK_ID: self.context.benchmark_id,
        }
        if self.context.provider_id:
            default_annotations[OCI_ANNOTATION_PROVIDER_ID] = self.context.provider_id
        # User-provided annotations take precedence
        merged_annotations = {**default_annotations, **spec.coordinates.annotations}
        logger.debug("OCI artifact annotations: %s", merged_annotations)
        oci_ref = (
            spec.coordinates.oci_host
            + "/"
            + spec.coordinates.oci_repository
            + ":"
            + tag
        )

        with tempfile.TemporaryDirectory(prefix="oci_layout_") as temp_dir:
            temp_path = Path(temp_dir)
            create_simple_oci_artifact(
                source_path=Path(spec.files_path),
                oci_layout_path=temp_path,
                artifact_type=OCI_ARTIFACT_TYPE,
                annotations=merged_annotations,
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
                provider.auth.load_configs(
                    spec.coordinates.oci_host, [custom_auth_path]
                )
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
        if not artifact_digest:
            raise RuntimeError(
                f"Registry response for {oci_ref} did not include a "
                "Docker-Content-Digest header."
            )
        artifact_reference = oci_ref + "@" + artifact_digest

        return OCIArtifactResult(digest=artifact_digest, reference=artifact_reference)
