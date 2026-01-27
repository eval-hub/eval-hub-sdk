"""Example script for running the adapter locally without Kubernetes.

This demonstrates how to test your adapter implementation locally
before deploying it to Kubernetes.
"""

import logging

from evalhub.adapter import (
    JobCallbacks,
    JobResults,
    JobSpec,
    JobStatusUpdate,
    ModelConfig,
    OCIArtifactResult,
    OCIArtifactSpec,
)

# Import the example adapter from the local file
from simple_adapter import ExampleAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class LocalCallbacks(JobCallbacks):
    """Local callbacks for testing without sidecar."""

    def report_status(self, update: JobStatusUpdate) -> None:
        """Print status updates to console."""
        logger.info(
            f"Status: {update.status.value} | "
            f"Phase: {update.phase.value if update.phase else 'N/A'} | "
            f"Progress: {update.progress or 'N/A'} | "
            f"Message: {update.message or ''}"
        )

    def create_oci_artifact(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
        """Mock OCI artifact creation for local testing."""
        logger.info(
            f"Would create OCI artifact with {len(spec.files)} files "
            f"for job {spec.job_id}"
        )

        # In local mode, we just return a mock result
        # In production, this would push to an actual registry
        return OCIArtifactResult(
            digest="sha256:local-test",
            reference=f"localhost/eval-results/{spec.benchmark_id}:{spec.job_id}",
            size_bytes=sum(f.stat().st_size for f in spec.files if f.exists()),
        )

    def report_results(self, results: JobResults) -> None:
        """Print final results to console."""
        logger.info(
            f"Job {results.job_id} completed | "
            f"Benchmark: {results.benchmark_id} | "
            f"Model: {results.model_name} | "
            f"Overall Score: {results.overall_score} | "
            f"Examples: {results.num_examples_evaluated} | "
            f"Duration: {results.duration_seconds:.2f}s"
        )

        # Print individual metrics
        logger.info("Metrics:")
        for metric in results.results:
            logger.info(f"  {metric.metric_name}: {metric.metric_value}")


def main() -> None:
    """Run adapter locally for testing."""
    logger.info("Starting local adapter test")

    # Create job specification
    # In production, this would be loaded from /etc/eval-job/spec.json
    spec = JobSpec(
        job_id="local-test-001",
        benchmark_id="mmlu",
        model=ModelConfig(
            url="http://localhost:8000/v1",  # Your local model server
            name="test-model",
        ),
        num_examples=10,  # Small number for quick testing
        num_few_shot=5,
        random_seed=42,
        benchmark_config={
            "subject": "mathematics",
        },
        experiment_name="local-test",
        tags={
            "env": "local",
            "test": "true",
        },
    )

    # Create callbacks
    callbacks = LocalCallbacks()

    # Create and run adapter
    adapter = ExampleAdapter()

    logger.info(f"Running benchmark: {spec.benchmark_id}")
    logger.info(f"Model: {spec.model.name} at {spec.model.url}")
    logger.info(f"Examples: {spec.num_examples}")

    try:
        results = adapter.run_benchmark_job(spec, callbacks)

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Job ID: {results.job_id}")
        logger.info(f"Benchmark: {results.benchmark_id}")
        logger.info(f"Overall Score: {results.overall_score}")
        logger.info(f"Examples Evaluated: {results.num_examples_evaluated}")
        logger.info(f"Duration: {results.duration_seconds:.2f} seconds")

        if results.oci_artifact:
            logger.info(f"Artifact: {results.oci_artifact.reference}")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
