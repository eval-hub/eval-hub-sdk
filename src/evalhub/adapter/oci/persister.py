"""OCI artifact persistence for evaluation job files."""

import logging
from pathlib import Path
import tempfile
from typing import Protocol

from olot.oci_artifact import create_simple_oci_artifact
import oras.provider
from oras.layout import Layout, NewLayout

from evalhub.models.api import (
    EvaluationJob,
    EvaluationJobFilesLocation,
    OCICoordinate,
    PersistResponse,
)

logger = logging.getLogger(__name__)


class Persister(Protocol):
    """Protocol for OCI artifact persisters."""

    async def persist(
        self,
        files_location: EvaluationJobFilesLocation,
        coordinate: OCICoordinate,
        job: EvaluationJob,
    ) -> PersistResponse:
        """Persist evaluation job files as OCI artifact.

        Args:
            files_location: Files to persist
            coordinate: OCI coordinates
            job: The evaluation job

        Returns:
            PersistResponse: Persistence result
        """
        ...


class OCIArtifactPersister:
    """Placeholder OCI artifact persister."""

    def __init__(self, registry: str):
        self.registry = registry


    async def persist(
        self,
        files_location: EvaluationJobFilesLocation,
        coordinate: OCICoordinate,
        job: EvaluationJob,
    ) -> PersistResponse:
        """Persist evaluation job files as OCI artifact.

        Args:
            files_location: Files to persist
            coordinate: OCI coordinates
            job: Evaluation job

        Returns:
            PersistResponse: Persistence result
        """
        subject_info = (
            f" with subject '{coordinate.oci_subject}'"
            if coordinate.oci_subject
            else ""
        )
        logger.warning(
            f"OCI persister is a placeholder. "
            f"Would persist files from {files_location.path} to {coordinate.oci_ref}{subject_info}"
        )

        files_count = 0
        if files_location.path is not None:
            source = Path(files_location.path)
            if source.exists():
                if source.is_file():
                    files_count = 1
                elif source.is_dir():
                    files_count = sum(1 for f in source.rglob("*") if f.is_file())

        temp_dir = tempfile.mkdtemp(prefix="oci_layout_")
        temp_path = Path(temp_dir)
        if files_location.path is not None:
            create_simple_oci_artifact(
                source_path=Path(files_location.path),
                oci_layout_path=temp_path,
            )

        # Display the content of temp_path
        logger.info(f"Contents of temp_path ({temp_path}):")
        for item in temp_path.rglob("*"):
            if item.is_file():
                logger.info(f"  File: {item.relative_to(temp_path)}")
            elif item.is_dir():
                logger.info(f"  Dir:  {item.relative_to(temp_path)}")

        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True
        provider = oras.provider.Registry()
        provider.auth.hostname = self.registry
        provider.auth.load_configs(self.registry)
        response = Layout(str(temp_path)).push_to_registry(
            provider=provider,
            target="quay.io/mmortari/demo20260212:latest",
            tag="latest",
        )

        print(response.status_code)
        print(response.headers)

        return PersistResponse(
            id=job.id,
            oci_ref=f"{coordinate.oci_ref}@sha256:{'0' * 64}",
            digest=f"sha256:{'0' * 64}",
            files_count=files_count,
            metadata={
                "placeholder": True,
                "message": "OCI persistence not yet implemented",
            },
        )
