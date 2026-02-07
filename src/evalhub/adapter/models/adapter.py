"""Simplified framework adapter base class."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .job import JobCallbacks, JobResults, JobSpec

if TYPE_CHECKING:
    from ..settings import AdapterSettings

logger = logging.getLogger(__name__)


class FrameworkAdapter(ABC):
    """Simplified base class for framework adapters.

    This is a simple job runner that executes a single benchmark evaluation.
    The adapter does not manage jobs, expose APIs, or run as a server.

    The adapter automatically loads settings from env (or uses provided settings)
    and loads the JobSpec from the configured path.

    Job lifecycle:
    1. Service creates a Kubernetes Job with the adapter container + sidecar
    2. ConfigMap mounts JobSpec at pod startup (/meta/job.json by default)
    3. Adapter is initialized with settings and automatically loads JobSpec
    4. run_benchmark_job() is called with config (typically self.job_spec) and callbacks
    5. Adapter reports status via callbacks to localhost sidecar
    6. Sidecar forwards updates to eval-hub service
    7. Adapter returns JobResults when complete
    8. Pod terminates

    Framework adapters should:
    - Implement run_benchmark_job() to execute the benchmark
    - Use the config parameter for job configuration (passed as adapter.job_spec in production)
    - Access self.settings for runtime configuration (service_url, registry, etc.)
    - Use callbacks.report_status() to report progress
    - Use callbacks.create_oci_artifact() to persist results
    - Return JobResults with evaluation outcomes

    Example:
        ```python
        class MyAdapter(FrameworkAdapter):
            def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
                logger.info(f"Running {config.benchmark_id}")

                # Report progress
                callbacks.report_status(JobStatusUpdate(...))

                # Run evaluation
                results = evaluate(config.model, ...)

                return JobResults(...)

        # Production usage (auto-detects /meta/job.json in k8s)
        adapter = MyAdapter()

        # Local testing with hardcoded path (no env var needed)
        adapter = MyAdapter(job_spec_path="/custom/path/to/job.json")

        # Alternative: using environment variable
        # export EVALHUB_JOB_SPEC_PATH=/path/to/job.json
        adapter = MyAdapter()

        # Run the job
        callbacks = DefaultCallbacks(
            job_id=adapter.job_spec.id,
            sidecar_url=adapter.job_spec.callback_url,
            ...
        )
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        ```
    """

    def __init__(
        self,
        settings: AdapterSettings | None = None,
        job_spec_path: str | None = None,
    ) -> None:
        """Initialize adapter with settings and load JobSpec.

        Args:
            settings: Runtime settings. If None, loads from environment via
                      AdapterSettings.from_env().
            job_spec_path: Optional path to job spec JSON file.
                          If provided, overrides EVALHUB_JOB_SPEC_PATH env var.
                          This is a convenience parameter - you can also pass it
                          via settings.job_spec_path.

        Raises:
            FileNotFoundError: If job spec file does not exist
            ValueError: If job spec JSON is invalid
        """
        from pathlib import Path

        from ..settings import AdapterSettings as SettingsClass

        # If job_spec_path is provided, create or update settings
        if job_spec_path is not None:
            if settings is not None:
                # Update existing settings
                settings.job_spec_path = Path(job_spec_path)
                self._settings = settings
            else:
                # Create new settings from env with overridden path
                self._settings = SettingsClass.from_env()
                self._settings.job_spec_path = Path(job_spec_path)
        else:
            # Use provided settings or load from env
            self._settings = (
                settings if settings is not None else SettingsClass.from_env()
            )

        self._job_spec = self._load_job_spec()

    def _load_job_spec(self) -> JobSpec:
        """Load JobSpec from configured path.

        Returns:
            JobSpec: Loaded job specification

        Raises:
            FileNotFoundError: If job spec file does not exist
            ValueError: If job spec JSON is invalid
        """
        config_path = self._settings.resolved_job_spec_path
        logger.info(f"Loading job spec from {config_path}")
        return JobSpec.from_file(config_path)

    @property
    def settings(self) -> AdapterSettings:
        """Get the adapter settings.

        Returns:
            AdapterSettings: Runtime settings for this adapter
        """
        return self._settings

    @property
    def job_spec(self) -> JobSpec:
        """Get the loaded job specification.

        Returns:
            JobSpec: The job specification for this adapter instance
        """
        return self._job_spec

    @abstractmethod
    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Run a benchmark evaluation job.

        This is the single entry point for job execution. It runs synchronously
        and defines the complete job lifetime. The method should:

        1. Validate the job configuration
        2. Load the benchmark and model
        3. Report status updates via callbacks as the job progresses
        4. Execute the evaluation
        5. Persist results via callbacks.create_oci_artifact() if needed
        6. Return JobResults with outcomes

        The adapter automatically loads the JobSpec on initialization and makes it
        available via self.job_spec property. In production, you typically pass
        self.job_spec to this method. However, you can override it for testing:

        Production usage:
            adapter = MyAdapter()  # Loads from ConfigMap
            results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        Testing usage:
            adapter = MyAdapter()
            test_spec = JobSpec(job_id="test", benchmark_id="mmlu", ...)
            results = adapter.run_benchmark_job(test_spec, callbacks)

        Args:
            config: Job specification to execute (typically self.job_spec)
            callbacks: Callbacks for status updates and artifact persistence

        Returns:
            JobResults: Evaluation results and metadata

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If evaluation fails
        """
        pass
