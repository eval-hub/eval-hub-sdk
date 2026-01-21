"""OCI artifact persistence for evaluation job files (placeholder implementation)."""

import logging
from pathlib import Path

from evalhub.models.api import (
    EvaluationJob,
    EvaluationJobFilesLocation,
    OCICoordinate,
    PersistResponse,
)

logger = logging.getLogger(__name__)


class OCIArtifactPersister:
    """Handles OCI artifact creation (no-op placeholder for now).

    Future implementation will integrate dependencies as needed for actual OCI artifact pushing.
    """

    async def persist(
        self,
        files_location: EvaluationJobFilesLocation,
        coordinate: OCICoordinate,
        job: EvaluationJob,
    ) -> PersistResponse:
        """Create and push OCI artifact with job files (no-op placeholder).

        Currently returns a mock PersistResponse without actually persisting.
        Future implementation will:
        1. Validate source paths exist
        2. Create temporary tarball with files
        3. Generate OCI manifest (with subject if provided)
        4. Push artifact using integrated dependencies
        5. Return persistence response with digest

        Args:
            files_location: Files to persist
            coordinate: OCI coordinates (reference and optional subject)
            job: The evaluation job

        Returns:
            PersistResponse: Mock response with placeholder values
        """
        subject_info = (
            f" with subject '{coordinate.oci_subject}'"
            if coordinate.oci_subject
            else ""
        )
        logger.warning(
            f"OCI persister is a no-op placeholder. "
            f"Would persist files from {files_location.path} to {coordinate.oci_ref}{subject_info}"
        )

        # Calculate number of files
        files_count = 0
        if files_location.path is not None:
            source = Path(files_location.path)
            if source.exists():
                if source.is_file():
                    files_count = 1
                elif source.is_dir():
                    files_count = sum(1 for f in source.rglob("*") if f.is_file())

        # Return mock response
        return PersistResponse(
            job_id=job.job_id,
            oci_ref=f"{coordinate.oci_ref}@sha256:{'0' * 64}",  # Placeholder digest
            digest=f"sha256:{'0' * 64}",
            files_count=files_count,
            metadata={
                "placeholder": True,
                "message": "OCI persistence not yet implemented",
            },
        )
