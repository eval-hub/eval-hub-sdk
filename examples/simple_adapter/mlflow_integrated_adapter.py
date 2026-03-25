"""Example adapter with MLflow integration.

This example demonstrates how to integrate MLflow into a framework adapter:

1. Read mlflow_experiment_id from JobSpec
2. Create MLflow run within the experiment
3. Log metrics, parameters, and tags to MLflow
4. Upload artifacts (reports, results, plots) to MLflow
5. Report mlflow_run_id back to EvalHub

This ensures:
- Artifacts persist beyond pod lifecycle
- MLflow UI can display run details
- Dashboard can embed MLflow UI
- Complete traceability from job to artifacts
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
    OCIArtifactSpec,
)
from evalhub.adapter.callbacks import DefaultCallbacks
from evalhub.adapter.mlflow import MlflowClient, Metric, Param

logger = logging.getLogger(__name__)


def _status_message(text: str, code: str = "status_update") -> MessageInfo:
    return MessageInfo(message=text, message_code=code)


class MLflowIntegratedAdapter(FrameworkAdapter):
    """Example adapter with MLflow integration.

    Demonstrates best practices for:
    - Creating MLflow runs from mlflow_experiment_id
    - Logging metrics, parameters, and tags
    - Uploading artifacts (JSON, plots, reports)
    - Reporting mlflow_run_id in results
    """

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Execute a benchmark evaluation job with MLflow tracking.

        Args:
            config: Job specification containing mlflow_experiment_id
            callbacks: Callbacks for status updates and artifact persistence

        Returns:
            JobResults: Evaluation results including mlflow_run_id

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If evaluation or MLflow tracking fails
        """
        start_time = time.time()
        logger.info(f"Starting job {config.id} for benchmark {config.benchmark_id}")

        # Initialize MLflow client if experiment ID is provided
        mlflow_client = None
        mlflow_run_id = None

        if config.mlflow_experiment_id:
            try:
                mlflow_client = MlflowClient()
                logger.info(
                    f"MLflow tracking enabled - experiment: {config.mlflow_experiment_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize MLflow client: {e}. "
                    "Continuing without MLflow tracking."
                )

        try:
            # Start MLflow run if client is initialized
            if mlflow_client and config.mlflow_experiment_id:
                with mlflow_client.start_run(
                    experiment_id=config.mlflow_experiment_id,
                    run_name=f"{config.benchmark_id}_{config.benchmark_index}",
                    tags={
                        "job_id": config.id,
                        "benchmark_id": config.benchmark_id,
                        "benchmark_index": str(config.benchmark_index),
                        "provider_id": config.provider_id,
                        "model_name": config.model.name,
                    },
                ) as run_id:
                    mlflow_run_id = run_id
                    logger.info(f"Created MLflow run: {mlflow_run_id}")

                    # Run evaluation with MLflow tracking
                    return self._run_with_mlflow(
                        config, callbacks, mlflow_client, mlflow_run_id, start_time
                    )
            else:
                # Run without MLflow tracking
                logger.info("Running without MLflow tracking")
                return self._run_without_mlflow(config, callbacks, start_time)

        except Exception as e:
            logger.exception("Evaluation failed")

            # Mark MLflow run as failed
            if mlflow_client and mlflow_run_id:
                try:
                    mlflow_client.set_tag(mlflow_run_id, "status", "failed")
                    mlflow_client.set_tag(mlflow_run_id, "error", str(e))
                except Exception:
                    logger.warning("Failed to update MLflow run status")

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

    def _run_with_mlflow(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
        mlflow_client: MlflowClient,
        mlflow_run_id: str,
        start_time: float,
    ) -> JobResults:
        """Run evaluation with MLflow tracking enabled."""

        # Phase 1: Initialize
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.INITIALIZING,
                progress=0.0,
                message=_status_message(
                    f"Initializing {config.benchmark_id} evaluation with MLflow tracking"
                ),
            )
        )

        self._validate_config(config)

        # Log configuration to MLflow
        mlflow_client.log_batch(
            mlflow_run_id,
            params=[
                Param("benchmark_id", config.benchmark_id),
                Param("model_name", config.model.name),
                Param("model_url", config.model.url),
                Param("provider_id", config.provider_id),
                Param("num_examples", str(config.num_examples or "all")),
            ],
        )

        # Log benchmark parameters
        for key, value in config.parameters.items():
            mlflow_client.log_param(mlflow_run_id, f"benchmark.{key}", str(value))

        logger.info("Configuration validated and logged to MLflow")

        # Phase 2: Load data
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.LOADING_DATA,
                progress=0.1,
                message=_status_message("Loading benchmark data"),
            )
        )

        dataset = self._load_dataset(config.benchmark_id, config.num_examples)
        logger.info(f"Loaded {len(dataset)} examples")

        # Log dataset size
        mlflow_client.log_metric(mlflow_run_id, "dataset_size", float(len(dataset)))

        # Phase 3: Run evaluation
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.RUNNING_EVALUATION,
                progress=0.3,
                message=_status_message(f"Evaluating on {len(dataset)} examples"),
            )
        )

        results = self._evaluate(config.model, dataset, config.parameters)
        logger.info(f"Evaluation complete with {len(results)} metrics")

        # Log metrics to MLflow
        mlflow_metrics = []
        for result in results:
            if isinstance(result.metric_value, (int, float)):
                mlflow_metrics.append(
                    Metric(key=result.metric_name, value=float(result.metric_value))
                )

        if mlflow_metrics:
            mlflow_client.log_batch(mlflow_run_id, metrics=mlflow_metrics)
            logger.info(f"Logged {len(mlflow_metrics)} metrics to MLflow")

        # Phase 4: Post-processing
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.POST_PROCESSING,
                progress=0.8,
                message=_status_message("Processing results"),
            )
        )

        overall_score = self._compute_overall_score(results)
        output_dir, output_files = self._save_detailed_results(
            config.id, config.benchmark_id, results
        )

        # Log overall score
        if overall_score is not None:
            mlflow_client.log_metric(mlflow_run_id, "overall_score", overall_score)

        # Upload artifacts to MLflow
        logger.info(f"Uploading {len(output_files)} artifacts to MLflow")
        for file_path in output_files:
            artifact_path = f"results/{file_path.name}"
            mlflow_client.upload_artifact_file(
                mlflow_run_id, artifact_path, file_path
            )
            logger.info(f"✅ Uploaded {artifact_path} to MLflow")

        # Phase 5: Persist artifacts (OCI)
        oci_artifact = None
        if config.exports and config.exports.oci:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.9,
                    message=_status_message("Persisting artifacts to OCI registry"),
                )
            )

            oci_artifact = callbacks.create_oci_artifact(
                OCIArtifactSpec(
                    files_path=output_dir,
                    coordinates=config.exports.oci.coordinates,
                )
            )
            logger.info(f"OCI artifact persisted: {oci_artifact.digest}")

            # Log OCI artifact info to MLflow
            mlflow_client.set_tag(mlflow_run_id, "oci_digest", oci_artifact.digest)
            mlflow_client.set_tag(mlflow_run_id, "oci_reference", oci_artifact.reference)

        # Mark run as successful
        mlflow_client.set_tag(mlflow_run_id, "status", "completed")

        duration = time.time() - start_time
        mlflow_client.log_metric(mlflow_run_id, "duration_seconds", duration)

        # Return results with mlflow_run_id
        return JobResults(
            id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=config.model.name,
            results=results,
            overall_score=overall_score,
            num_examples_evaluated=len(dataset),
            duration_seconds=duration,
            completed_at=datetime.now(UTC),
            evaluation_metadata={
                "framework": "mlflow_integrated_adapter",
                "framework_version": "1.0.0",
                "mlflow_tracking_enabled": True,
                "parameters": config.parameters,
            },
            oci_artifact=oci_artifact,
            mlflow_run_id=mlflow_run_id,  # ✅ Include MLflow run ID
        )

    def _run_without_mlflow(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
        start_time: float,
    ) -> JobResults:
        """Run evaluation without MLflow tracking (fallback mode)."""

        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.INITIALIZING,
                progress=0.0,
                message=_status_message(
                    f"Initializing {config.benchmark_id} evaluation (no MLflow)"
                ),
            )
        )

        self._validate_config(config)
        dataset = self._load_dataset(config.benchmark_id, config.num_examples)
        results = self._evaluate(config.model, dataset, config.parameters)
        overall_score = self._compute_overall_score(results)
        output_dir, _ = self._save_detailed_results(
            config.id, config.benchmark_id, results
        )

        oci_artifact = None
        if config.exports and config.exports.oci:
            oci_artifact = callbacks.create_oci_artifact(
                OCIArtifactSpec(
                    files_path=output_dir,
                    coordinates=config.exports.oci.coordinates,
                )
            )

        duration = time.time() - start_time

        return JobResults(
            id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=config.model.name,
            results=results,
            overall_score=overall_score,
            num_examples_evaluated=len(dataset),
            duration_seconds=duration,
            completed_at=datetime.now(UTC),
            evaluation_metadata={
                "framework": "mlflow_integrated_adapter",
                "framework_version": "1.0.0",
                "mlflow_tracking_enabled": False,
            },
            oci_artifact=oci_artifact,
            mlflow_run_id=None,
        )

    def _validate_config(self, config: JobSpec) -> None:
        """Validate job configuration."""
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")
        if not config.model.url:
            raise ValueError("model.url is required")
        if not config.model.name:
            raise ValueError("model.name is required")

    def _load_dataset(
        self, benchmark_id: str, num_examples: int | None
    ) -> list[dict[str, Any]]:
        """Load benchmark dataset."""
        all_examples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(100)]
        if num_examples:
            return all_examples[:num_examples]
        return all_examples

    def _evaluate(
        self,
        model: ModelConfig,
        dataset: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> list[EvaluationResult]:
        """Run evaluation on the dataset."""
        time.sleep(0.5)
        return [
            EvaluationResult(
                metric_name="accuracy",
                metric_value=0.85,
                metric_type="float",
                num_samples=len(dataset),
            ),
            EvaluationResult(
                metric_name="f1_score",
                metric_value=0.83,
                metric_type="float",
                num_samples=len(dataset),
            ),
        ]

    def _compute_overall_score(self, results: list[EvaluationResult]) -> float | None:
        """Compute overall score from individual metrics."""
        numeric_values = []
        for result in results:
            if isinstance(result.metric_value, (int, float)):
                value = float(result.metric_value)
                if value <= 1.0:
                    numeric_values.append(value)
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
        return None

    def _save_detailed_results(
        self, job_id: str, benchmark_id: str, results: list[EvaluationResult]
    ) -> tuple[Path, list[Path]]:
        """Save detailed results to files."""
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
        files.append(summary_file)

        return output_dir, files


def main() -> None:
    """Example main function showing MLflow-integrated adapter usage."""
    import sys

    class SidecarCallbacks(DefaultCallbacks):
        def report_status(self, update: JobStatusUpdate) -> None:
            logger.info(f"Status: {update.status} - {update.message.message}")

        def report_results(self, results: JobResults) -> None:
            logger.info(
                f"Job {results.id} completed with score {results.overall_score}"
            )
            if results.mlflow_run_id:
                logger.info(f"MLflow run ID: {results.mlflow_run_id}")

    try:
        adapter = MLflowIntegratedAdapter()
        logger.info(f"Loaded job {adapter.job_spec.id}")

        if adapter.job_spec.mlflow_experiment_id:
            logger.info(
                f"MLflow experiment ID: {adapter.job_spec.mlflow_experiment_id}"
            )

        callbacks = SidecarCallbacks(
            adapter.job_spec.id,
            adapter.job_spec.provider_id,
            adapter.job_spec.benchmark_id,
            adapter.job_spec.benchmark_index,
            sidecar_url=adapter.job_spec.callback_url,
            oci_auth_config_path=adapter.settings.oci_auth_config_path,
            oci_insecure=adapter.settings.oci_insecure,
        )

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_results(results)

        logger.info("✅ Job completed successfully!")
        sys.exit(0)
    except Exception:
        logger.exception("❌ Job failed")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
