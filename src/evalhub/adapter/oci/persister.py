"""OCI artifact persistence for evaluation job files."""

import logging
from pathlib import Path
from typing import Protocol

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
