"""Adapter SDK for building BYOF (Bring Your Own Framework) adapters.

The adapter SDK provides a simple interface for integrating evaluation frameworks
with the eval-hub service. Adapters are job runners that execute benchmarks in
Kubernetes pods and communicate with the service via a sidecar container.

Quick Start:
-----------
    from evalhub.adapter import (
        FrameworkAdapter,
        JobSpec,
        JobCallbacks,
        JobResults,
        JobStatus,
        JobPhase,
        JobStatusUpdate,
    )

    class MyAdapter(FrameworkAdapter):
        def run_benchmark_job(
            self, config: JobSpec, callbacks: JobCallbacks
        ) -> JobResults:
            # Report progress
            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.RUNNING_EVALUATION,
                progress=0.5
            ))

            # Run evaluation
            results = evaluate_benchmark(
                benchmark_id=config.benchmark_id,
                model=config.model,
                ...
            )

            # Persist artifacts if needed
            if output_files:
                artifact = callbacks.create_oci_artifact(OCIArtifactSpec(
                    files=output_files,
                    id=config.id,
                    benchmark_id=config.benchmark_id,
                    model_name=config.model.name
                ))

            # Return results
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                results=results,
                ...
            )

Architecture:
-----------
The adapter SDK uses a job runner architecture:

1. Service creates a Kubernetes Job with adapter container + sidecar
2. ConfigMap mounts JobSpec at pod startup
3. Adapter reads JobSpec and calls run_benchmark_job()
4. Adapter reports status via callbacks to localhost sidecar
5. Sidecar forwards updates to eval-hub service
6. Adapter returns JobResults when complete
7. Entrypoint reports results via callbacks.report_results()
8. Pod terminates
"""

# Simplified adapter API (current)
# Re-export common models from evalhub.models.api for convenience
from ..models.api import (
    EvaluationResult,
    JobStatus,
    ModelConfig,
)
from .callbacks import DefaultCallbacks
from .config import get_job_spec_path
from .models import (
    ErrorInfo,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactResult,
    OCIArtifactSpec,
)
from .oci import OCIArtifactPersister, Persister
from .settings import AdapterSettings

# Legacy API is available but deprecated
# from evalhub.adapter.legacy import ...

__all__ = [
    # Core adapter interface
    "FrameworkAdapter",
    # Job models
    "JobSpec",
    "JobCallbacks",
    "JobResults",
    "JobStatusUpdate",
    "JobPhase",
    "ErrorInfo",
    "MessageInfo",
    # OCI models
    "OCIArtifactSpec",
    "OCIArtifactResult",
    "Persister",
    "OCIArtifactPersister",
    # Callback implementation
    "DefaultCallbacks",
    # Configuration utilities
    "get_job_spec_path",
    "AdapterSettings",
    # Common models (re-exported for convenience)
    "JobStatus",
    "ModelConfig",
    "EvaluationResult",
]
