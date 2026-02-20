"""Configuration utilities for adapter SDK.

This module provides utilities for configuring the adapter, including
environment variable handling for job spec location and other settings.
"""

from pathlib import Path

from .settings import JOB_SPEC_PATH_ENV


def get_job_spec_path() -> Path:
    """Get the job spec file path from environment or default.

    The job spec path can be configured via the EVALHUB_JOB_SPEC_PATH
    environment variable. This allows the SDK to work in different
    environments:

    - Kubernetes (production): /meta/job.json (default)
    - Local testing: ./meta/job.json or any custom path
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
    from .settings import AdapterSettings

    settings = AdapterSettings.from_env()
    path = settings.resolved_job_spec_path

    if not path.exists():
        raise FileNotFoundError(
            f"Job spec file not found at {path}. "
            f"Set {JOB_SPEC_PATH_ENV} environment variable to specify a custom location."
        )

    return path
