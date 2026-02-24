"""Configuration utilities for adapter SDK.

This module provides utilities for configuring the adapter, including
environment variable handling for job spec location and other settings.
"""

import os
from enum import StrEnum
from pathlib import Path


class EvalHubMode(StrEnum):
    """Execution mode for the adapter."""

    K8S = "k8s"
    LOCAL = "local"


# Default job spec location.
# - Kubernetes: /meta/job.json
# - Local dev: meta/job.json (repo-relative) for convenience
# - Job runs submitted via evalhub server in --local mode:
#     will always use EVALHUB_JOB_SPEC_PATH environment variable to find the job spec file
#     eg. EVALHUB_JOB_SPEC_PATH=/tmp/evalhub-jobs/{job_id}/{benchmark_index}/{provider_id}/{benchmark_id}/meta/job.json
DEFAULT_JOB_SPEC_PATH_K8S = "/meta/job.json"
DEFAULT_JOB_SPEC_PATH_LOCAL = "meta/job.json"

# Environment variable name for job spec location
JOB_SPEC_PATH_ENV = "EVALHUB_JOB_SPEC_PATH"


def get_job_spec_path() -> Path:
    """Get the job spec file path from environment or default.

    The job spec path can be configured via the EVALHUB_JOB_SPEC_PATH
    environment variable. This allows the SDK to work in different
    environments:

    - Kubernetes (production): /meta/job.json (default)
    - Local testing: ./meta/job.json or any custom path
    - Job runs submitted via evalhub server in --local mode:
         will always use EVALHUB_JOB_SPEC_PATH environment variable to find the job spec file
    - CI/CD: Custom paths as needed

    Returns:
        Path: Path to the job spec JSON file

    Raises:
        FileNotFoundError: If the job spec file does not exist

    Example:
        ```python
        # Use default location (Kubernetes)
        spec_path = get_job_spec_path()  # /meta/job.json

        # Set custom location for local testing
        os.environ["EVALHUB_JOB_SPEC_PATH"] = "./meta/job.json"
        spec_path = get_job_spec_path()  # ./meta/job.json
        ```

    Environment Variables:
        EVALHUB_JOB_SPEC_PATH: Path to job spec JSON file (optional)
            Default: /meta/job.json
    """
    # Env var always wins.
    path_str = os.getenv(JOB_SPEC_PATH_ENV)
    if not path_str:
        mode = os.getenv("EVALHUB_MODE", EvalHubMode.K8S).strip().lower()
        if mode == EvalHubMode.LOCAL:
            path_str = DEFAULT_JOB_SPEC_PATH_LOCAL
        else:
            path_str = DEFAULT_JOB_SPEC_PATH_K8S
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(
            f"Job spec file not found at {path}. "
            f"Set {JOB_SPEC_PATH_ENV} environment variable to specify a custom location."
        )

    return path
