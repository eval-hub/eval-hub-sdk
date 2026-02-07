"""Example implementation of a framework adapter.

This example demonstrates how to implement a framework adapter. The adapter:

1. Reads JobSpec from a mounted ConfigMap
2. Executes a benchmark evaluation
3. Reports progress via callbacks
4. Persists results as OCI artifacts
5. Returns JobResults

This is a complete example showing all the key patterns for adapter development.
"""

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalhub.adapter import (
    ErrorInfo,
    EvaluationResult,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    ModelConfig,
    OCIArtifactResult,
    OCIArtifactSpec,
)

logger = logging.getLogger(__name__)


def _status_message(text: str, code: str = "status_update") -> MessageInfo:
    return MessageInfo(message=text, message_code=code)


class ExampleAdapter(FrameworkAdapter):
    """Example adapter demonstrating the adapter API.

    This adapter shows how to:
    - Load and validate job configuration
    - Report status updates at key points
    - Execute a benchmark evaluation
    - Persist results as OCI artifacts
    - Return structured results
    """

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Execute a benchmark evaluation job.

        Args:
            config: Job specification (typically self.job_spec, but can be overridden)
            callbacks: Callbacks for status updates and artifact persistence

        Returns:
            JobResults: Evaluation results and metadata

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If evaluation fails
        """
        start_time = time.time()
        logger.info(f"Starting job {config.id} for benchmark {config.benchmark_id}")

        try:
            # Phase 1: Initialize
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=_status_message(
                        f"Initializing {config.benchmark_id} evaluation"
                    ),
                )
            )

            self._validate_config(config)
            logger.info("Configuration validated")

            # Phase 2: Load data
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.1,
                    message=_status_message("Loading benchmark data"),
                    current_step="Loading dataset",
                    total_steps=4,
                    completed_steps=1,
                )
            )

            dataset = self._load_dataset(config.benchmark_id, config.num_examples)
            logger.info(f"Loaded {len(dataset)} examples")

            # Phase 3: Run evaluation
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=_status_message(f"Evaluating on {len(dataset)} examples"),
                    current_step="Running evaluation",
                    total_steps=4,
                    completed_steps=2,
                )
            )

            results = self._evaluate(config.model, dataset, config.benchmark_config)
            logger.info(f"Evaluation complete with {len(results)} metrics")

            # Phase 4: Post-processing
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=_status_message("Processing results"),
                    current_step="Post-processing",
                    total_steps=4,
                    completed_steps=3,
                )
            )

            overall_score = self._compute_overall_score(results)
            output_files = self._save_detailed_results(
                config.id, config.benchmark_id, results
            )
            logger.info(f"Results saved to {len(output_files)} files")

            # Phase 5: Persist artifacts
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.9,
                    message=_status_message("Persisting artifacts to OCI registry"),
                    current_step="Creating OCI artifact",
                    total_steps=4,
                    completed_steps=4,
                )
            )

            oci_artifact = None
            if output_files:
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files=output_files,
                        base_path=Path("/tmp/job_results"),
                        title=f"Evaluation results for {config.benchmark_id}",
                        description=f"Results from job {config.id}",
                        annotations={
                            "job_id": config.id,
                            "benchmark_id": config.benchmark_id,
                            "model_name": config.model.name,
                            "overall_score": str(overall_score)
                            if overall_score
                            else "N/A",
                        },
                        id=config.id,
                        benchmark_id=config.benchmark_id,
                        model_name=config.model.name,
                    )
                )
                logger.info(f"Artifact persisted: {oci_artifact.digest}")

            # Compute final metrics
            duration = time.time() - start_time

            # Return results
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                results=results,
                overall_score=overall_score,
                num_examples_evaluated=len(dataset),
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "simple_adapter",
                    "framework_version": "1.0.0",
                    "num_few_shot": config.benchmark_config.get("num_few_shot"),
                    "random_seed": config.benchmark_config.get("random_seed"),
                    "benchmark_config": config.benchmark_config,
                },
                oci_artifact=oci_artifact,
            )

        except Exception as e:
            logger.exception("Evaluation failed")
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=_status_message(
                        "Evaluation failed", code="evaluation_failed"
                    ),
                    error=ErrorInfo(
                        message=str(e),
                        message_code="evaluation_failed",
                    ),
                    error_details={"exception_type": type(e).__name__},
                )
            )
            raise

    def _validate_config(self, config: JobSpec) -> None:
        """Validate job configuration.

        Args:
            config: Job specification to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")

        if not config.model.url:
            raise ValueError("model.url is required")

        if not config.model.name:
            raise ValueError("model.name is required")

        logger.debug("Configuration is valid")

    def _load_dataset(
        self, benchmark_id: str, num_examples: int | None
    ) -> list[dict[str, Any]]:
        """Load benchmark dataset.

        Args:
            benchmark_id: Benchmark identifier
            num_examples: Number of examples to load (None = all)

        Returns:
            List of dataset examples
        """
        # This is a placeholder - real implementation would load actual data
        # from the benchmark framework
        logger.info(f"Loading dataset for benchmark: {benchmark_id}")

        # Simulate loading examples
        all_examples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(100)]

        if num_examples:
            return all_examples[:num_examples]
        return all_examples

    def _evaluate(
        self,
        model: ModelConfig,
        dataset: list[dict[str, Any]],
        benchmark_config: dict[str, Any],
    ) -> list[EvaluationResult]:
        """Run evaluation on the dataset.

        Args:
            model: Model configuration
            dataset: Dataset to evaluate on
            benchmark_config: Benchmark-specific configuration

        Returns:
            List of evaluation results
        """
        # This is a placeholder - real implementation would call the model
        # and compute metrics using the benchmark framework
        logger.info(
            f"Evaluating model {model.name} on {len(dataset)} examples "
            f"with config: {benchmark_config}"
        )

        # Simulate evaluation
        time.sleep(0.5)  # Simulate processing time

        # Return mock results
        return [
            EvaluationResult(
                metric_name="accuracy",
                metric_value=0.85,
                metric_type="float",
                confidence_interval=(0.82, 0.88),
                num_samples=len(dataset),
            ),
            EvaluationResult(
                metric_name="f1_score",
                metric_value=0.83,
                metric_type="float",
                num_samples=len(dataset),
            ),
            EvaluationResult(
                metric_name="latency_ms",
                metric_value=125.5,
                metric_type="float",
                metadata={"p50": 100, "p95": 200, "p99": 250},
            ),
        ]

    def _compute_overall_score(self, results: list[EvaluationResult]) -> float | None:
        """Compute overall score from individual metrics.

        Args:
            results: Individual evaluation results

        Returns:
            Overall score, or None if not applicable
        """
        # Simple example: average of all numeric metrics
        numeric_values = []
        for result in results:
            if isinstance(result.metric_value, int | float):
                # Normalize to 0-1 range (this is overly simplistic)
                value = float(result.metric_value)
                if value <= 1.0:  # Already normalized
                    numeric_values.append(value)

        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
        return None

    def _save_detailed_results(
        self, job_id: str, benchmark_id: str, results: list[EvaluationResult]
    ) -> list[Path]:
        """Save detailed results to files.

        Args:
            job_id: Job identifier
            benchmark_id: Benchmark identifier
            results: Evaluation results

        Returns:
            List of paths to saved files
        """
        output_dir = Path("/tmp/job_results") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        files = []

        # Save results as JSON
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "benchmark_id": benchmark_id,
                    "results": [
                        {
                            "metric_name": r.metric_name,
                            "metric_value": r.metric_value,
                            "metric_type": r.metric_type,
                            "confidence_interval": r.confidence_interval,
                            "num_samples": r.num_samples,
                            "metadata": r.metadata,
                        }
                        for r in results
                    ],
                },
                f,
                indent=2,
            )
        files.append(results_file)

        # Save summary
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Evaluation Results for {benchmark_id}\n")
            f.write("=" * 50 + "\n\n")
            for result in results:
                f.write(f"{result.metric_name}: {result.metric_value}\n")
                if result.confidence_interval:
                    f.write(f"  95% CI: {result.confidence_interval}\n")
        files.append(summary_file)

        logger.info(f"Saved {len(files)} result files to {output_dir}")
        return files


def main() -> None:
    """Example main function showing how to use the adapter.

    The adapter automatically loads the JobSpec from the mounted ConfigMap
    (default: /meta/job.json, configurable via EVALHUB_JOB_SPEC_PATH).

    In production, this would:
    1. Create adapter instance (automatically loads JobSpec)
    2. Create callbacks that communicate with localhost sidecar
    3. Call run_benchmark_job()
    4. Handle results
    """
    import sys

    # Define callbacks that communicate with sidecar
    # In production, these would make HTTP requests to localhost sidecar
    class SidecarCallbacks(JobCallbacks):
        def report_status(self, update: JobStatusUpdate) -> None:
            # In production: POST to http://localhost:8080/status
            logger.info(f"Status update: {update.status} - {update.message}")

        def create_oci_artifact(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
            # In production: POST to http://localhost:8080/artifacts
            logger.info(f"Creating OCI artifact with {len(spec.files)} files")
            # Return mock result for example
            return OCIArtifactResult(
                digest="sha256:abc123...",
                reference="ghcr.io/org/repo:job-123@sha256:abc123...",
                size_bytes=sum(f.stat().st_size for f in spec.files),
            )

        def report_results(self, results: JobResults) -> None:
            # In production: POST to http://localhost:8080/results
            logger.info(
                f"Job {results.id} completed with score {results.overall_score}"
            )

    try:
        # Create adapter (automatically loads JobSpec from ConfigMap)
        adapter = ExampleAdapter()
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")

        # Create callbacks
        callbacks = SidecarCallbacks()

        # Run benchmark job (pass self.job_spec or override with custom spec for testing)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info(f"Job completed successfully: {results.id}")
        logger.info(f"Overall score: {results.overall_score}")

        # Report final results
        callbacks.report_results(results)

        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"Job spec not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid job spec: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
