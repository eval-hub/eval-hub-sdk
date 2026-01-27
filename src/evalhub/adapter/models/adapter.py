"""Simplified framework adapter base class."""

import logging
from abc import ABC, abstractmethod

from .job import JobCallbacks, JobResults, JobSpec

logger = logging.getLogger(__name__)


class FrameworkAdapter(ABC):
    """Simplified base class for framework adapters.

    This is a simple job runner that executes a single benchmark evaluation.
    The adapter does not manage jobs, expose APIs, or run as a server.

    Job lifecycle:
    1. Service creates a Kubernetes Job with the adapter container + sidecar
    2. ConfigMap mounts JobSpec at pod startup
    3. Adapter reads JobSpec and calls run_benchmark_job()
    4. Adapter reports status via callbacks to localhost sidecar
    5. Sidecar forwards updates to eval-hub service
    6. Adapter returns JobResults when complete
    7. Pod terminates

    Framework adapters should:
    - Load the JobSpec from the mounted ConfigMap
    - Implement run_benchmark_job() to execute the benchmark
    - Use callbacks.report_status() to report progress
    - Use callbacks.create_oci_artifact() to persist results
    - Return JobResults with evaluation outcomes
    """

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

        Args:
            config: Job specification from mounted ConfigMap
            callbacks: Callbacks for status updates and artifact persistence

        Returns:
            JobResults: Evaluation results and metadata

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If evaluation fails
        """
        pass
