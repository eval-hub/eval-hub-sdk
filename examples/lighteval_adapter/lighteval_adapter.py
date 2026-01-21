"""LightEval Framework Adapter for EvalHub SDK.

This adapter wraps the LightEval framework to work with the EvalHub SDK.
LightEval is a lightweight evaluation framework for language models.
"""

import asyncio

# LightEval imports - simplified for demo
import importlib.util
import json
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalhub.adapter.models.framework import AdapterConfig, FrameworkAdapter
from evalhub.models.api import (
    BenchmarkInfo,
    EvaluationJob,
    EvaluationJobFilesLocation,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    FrameworkInfo,
    HealthResponse,
    JobStatus,
    ModelConfig,
)

LIGHTEVAL_AVAILABLE = importlib.util.find_spec("lighteval") is not None

logger = logging.getLogger(__name__)


class LightEvalAdapter(FrameworkAdapter):
    """Adapter for the LightEval evaluation framework."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._jobs: dict[str, EvaluationJob] = {}
        self._job_results: dict[str, EvaluationResponse] = {}
        self._available_tasks: dict[str, BenchmarkInfo] = {}
        self._initialized = False

        # LightEval configuration
        self._cache_dir = Path(
            config.framework_config.get("cache_dir", "/tmp/lighteval_cache")
        )
        self._output_dir = Path(
            config.framework_config.get("output_dir", "/tmp/lighteval_output")
        )
        self._max_samples = config.framework_config.get("max_samples", 1000)
        self._batch_size = config.framework_config.get("batch_size", 1)

        # Ensure directories exist
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the LightEval adapter."""
        if not LIGHTEVAL_AVAILABLE:
            raise RuntimeError(
                "LightEval is not installed. " "Install it with: pip install lighteval"
            )

        logger.info("Initializing LightEval adapter...")

        # For demo purposes, use fallback tasks instead of complex LightEval integration
        logger.info("Using demo task set for simplified integration")
        self._load_fallback_tasks()

        self._initialized = True
        logger.info(
            f"LightEval adapter initialized with {len(self._available_tasks)} tasks"
        )

    def _load_fallback_tasks(self) -> None:
        """Load a fallback set of common LightEval tasks."""
        fallback_tasks = [
            ("hellaswag", "HellaSwag", "Commonsense reasoning", ["accuracy"]),
            ("arc:easy", "ARC Easy", "Scientific reasoning", ["accuracy"]),
            ("arc:challenge", "ARC Challenge", "Scientific reasoning", ["accuracy"]),
            ("piqa", "PIQA", "Physical commonsense", ["accuracy"]),
            ("winogrande", "WinoGrande", "Commonsense reasoning", ["accuracy"]),
            ("truthfulqa:mc", "TruthfulQA", "Truthfulness", ["accuracy"]),
        ]

        for task_id, name, category, metrics in fallback_tasks:
            benchmark = BenchmarkInfo(
                benchmark_id=task_id,
                name=name,
                description=f"LightEval task: {name}",
                category=category,
                metrics=metrics,
            )
            self._available_tasks[task_id] = benchmark

    def _get_task_category(self, task_name: str) -> str:
        """Determine task category based on task name."""
        if any(x in task_name.lower() for x in ["math", "gsm", "algebra"]):
            return "math"
        elif any(x in task_name.lower() for x in ["code", "humaneval", "mbpp"]):
            return "coding"
        elif any(x in task_name.lower() for x in ["arc", "science", "physics"]):
            return "science"
        elif any(x in task_name.lower() for x in ["truthful", "ethics", "bias"]):
            return "safety"
        elif any(x in task_name.lower() for x in ["reasoning", "logic"]):
            return "reasoning"
        else:
            return "general"

    def _get_task_metrics(self, task_name: str) -> list[str]:
        """Get metrics for a task based on task name."""
        # Common metrics for different task types
        if "generation" in task_name.lower():
            return ["exact_match", "bleu", "rouge"]
        elif "classification" in task_name.lower():
            return ["accuracy", "f1", "precision", "recall"]
        else:
            return ["accuracy"]  # Default metric

    async def get_framework_info(self) -> FrameworkInfo:
        """Get LightEval framework information."""
        return FrameworkInfo(
            framework_id=self.config.framework_id,
            name=self.config.adapter_name,
            version=self.config.version,
            description="LightEval is a lightweight evaluation framework for language models",
            supported_benchmarks=list(self._available_tasks.values()),
            supported_model_types=[
                "transformers",
                "vllm",
                "openai",
                "anthropic",
                "endpoint",
            ],
            capabilities=[
                "text-generation",
                "text-classification",
                "multiple-choice",
                "open-ended",
            ],
        )

    async def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List available LightEval benchmarks."""
        return list(self._available_tasks.values())

    async def get_benchmark_info(self, benchmark_id: str) -> BenchmarkInfo | None:
        """Get information about a specific benchmark."""
        return self._available_tasks.get(benchmark_id)

    async def submit_evaluation(self, request: EvaluationRequest) -> EvaluationJob:
        """Submit a LightEval evaluation job."""
        if not self._initialized:
            raise RuntimeError("Adapter not initialized")

        if request.benchmark_id not in self._available_tasks:
            raise ValueError(f"Unknown benchmark: {request.benchmark_id}")

        job_id = str(uuid.uuid4())

        job = EvaluationJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            request=request,
            submitted_at=datetime.now(timezone.utc),
        )

        self._jobs[job_id] = job

        # Start evaluation in background
        asyncio.create_task(self._run_evaluation(job_id))

        return job

    async def _run_evaluation(self, job_id: str) -> None:
        """Run the LightEval evaluation in the background."""
        job = self._jobs[job_id]
        request = job.request

        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            job.progress = 0.1

            # Create temporary config files
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                eval_config = self._create_evaluation_config(request)
                json.dump(eval_config, f, indent=2)
                config_path = f.name

            try:
                # Create model config
                model_config = self._create_model_config(request.model)

                # Update progress
                job.progress = 0.3

                # Run LightEval evaluation
                results = await self._execute_lighteval(
                    config_path=config_path,
                    model_config=model_config,
                    benchmark_id=request.benchmark_id,
                    num_examples=request.num_examples,
                )

                # Update progress
                job.progress = 0.9

                # Create response
                response = self._create_evaluation_response(
                    job_id=job_id,
                    request=request,
                    results=results,
                )

                # Store results
                self._job_results[job_id] = response

                # Update job status
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                job.progress = 1.0

                logger.info(f"Evaluation {job_id} completed successfully")

            finally:
                # Clean up config file
                Path(config_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Evaluation {job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)

    def _create_evaluation_config(self, request: EvaluationRequest) -> dict[str, Any]:
        """Create LightEval configuration from request."""
        return {
            "tasks": [request.benchmark_id],
            "model_args": f"pretrained={request.model.name}",
            "batch_size": self._batch_size,
            "max_samples": request.num_examples or self._max_samples,
            "output_base_path": str(self._output_dir),
            "cache_dir": str(self._cache_dir),
            "use_chat_template": False,
            "system_prompt": None,
        }

    def _create_model_config(self, model_config: ModelConfig) -> dict[str, Any]:
        """Create LightEval model configuration."""
        # Map EvalHub model config to LightEval format
        lighteval_config = {
            "model_name": model_config.name,
            "model_type": "transformers",  # Default
        }

        if model_config.provider:
            provider_map = {
                "openai": "openai",
                "anthropic": "anthropic",
                "huggingface": "transformers",
                "vllm": "vllm",
            }
            lighteval_config["model_type"] = provider_map.get(
                model_config.provider.lower(), "transformers"
            )

        if model_config.parameters:
            lighteval_config.update(model_config.parameters)

        return lighteval_config

    async def _execute_lighteval(
        self,
        config_path: str,
        model_config: dict[str, Any],
        benchmark_id: str,
        num_examples: int | None,
    ) -> dict[str, Any]:
        """Execute LightEval evaluation."""
        # This is a simplified version - in a real implementation,
        # you would call the actual LightEval evaluation pipeline

        # Simulate evaluation with mock results
        await asyncio.sleep(2)  # Simulate evaluation time

        # Generate realistic mock results based on benchmark type
        if "math" in benchmark_id.lower():
            accuracy = 0.65 + (hash(benchmark_id) % 20) / 100  # 0.65-0.84
        elif "science" in benchmark_id.lower():
            accuracy = 0.70 + (hash(benchmark_id) % 25) / 100  # 0.70-0.94
        elif "code" in benchmark_id.lower():
            accuracy = 0.45 + (hash(benchmark_id) % 30) / 100  # 0.45-0.74
        else:
            accuracy = 0.75 + (hash(benchmark_id) % 20) / 100  # 0.75-0.94

        return {
            "accuracy": round(accuracy, 4),
            "num_examples": num_examples or 100,
            "benchmark_id": benchmark_id,
            "model_name": model_config["model_name"],
        }

    def _create_evaluation_response(
        self,
        job_id: str,
        request: EvaluationRequest,
        results: dict[str, Any],
    ) -> EvaluationResponse:
        """Create evaluation response from results."""
        eval_results = []

        # Add accuracy result
        if "accuracy" in results:
            eval_results.append(
                EvaluationResult(
                    metric_name="accuracy",
                    metric_value=results["accuracy"],
                    metric_type="float",
                    num_samples=results.get("num_examples"),
                )
            )

        # Add additional metrics if available
        for metric_name, value in results.items():
            if metric_name not in [
                "accuracy",
                "num_examples",
                "benchmark_id",
                "model_name",
            ]:
                eval_results.append(
                    EvaluationResult(
                        metric_name=metric_name,
                        metric_value=value,
                        metric_type=type(value).__name__,
                    )
                )

        return EvaluationResponse(
            job_id=job_id,
            benchmark_id=request.benchmark_id,
            model_name=request.model.name,
            results=eval_results,
            overall_score=results.get("accuracy"),
            num_examples_evaluated=results.get("num_examples", 100),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=120.0,  # Mock duration
        )

    async def get_job_status(self, job_id: str) -> EvaluationJob | None:
        """Get the status of an evaluation job."""
        return self._jobs.get(job_id)

    async def get_evaluation_results(self, job_id: str) -> EvaluationResponse | None:
        """Get the results of a completed evaluation."""
        if job_id not in self._jobs:
            return None

        job = self._jobs[job_id]
        if job.status != JobStatus.COMPLETED:
            return None

        return self._job_results.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an evaluation job."""
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            return True

        return False

    async def health_check(self) -> HealthResponse:
        """Perform health check."""
        status = "healthy" if self._initialized and LIGHTEVAL_AVAILABLE else "unhealthy"

        dependencies = {}
        if LIGHTEVAL_AVAILABLE:
            dependencies["lighteval"] = {"status": "available"}
        else:
            dependencies["lighteval"] = {"status": "missing", "error": "Not installed"}

        error_message = None
        if not LIGHTEVAL_AVAILABLE:
            error_message = "LightEval is not installed"

        return HealthResponse(
            status=status,
            framework_id=self.config.framework_id,
            version=self.config.version,
            uptime_seconds=3600.0 if self._initialized else None,
            dependencies=dependencies,
            error_message=error_message,
        )

    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        logger.info("Shutting down LightEval adapter...")

        # Cancel any running jobs
        for job_id, job in self._jobs.items():
            if job.status == JobStatus.RUNNING:
                await self.cancel_job(job_id)

        self._initialized = False

    async def job_files(self, job_id: str) -> EvaluationJobFilesLocation:
        """Specify LightEval output directory for persistence.

        Args:
            job_id: The job identifier

        Returns:
            EvaluationJobFilesLocation: Files location for this job
        """
        job = self._jobs.get(job_id)
        if not job:
            return EvaluationJobFilesLocation(job_id=job_id, path=None)

        # Only persist completed jobs
        if job.status != JobStatus.COMPLETED:
            logger.info(f"Job {job_id} not completed, no files to persist")
            return EvaluationJobFilesLocation(job_id=job_id, path=None)

        output_path = self._output_dir / job_id

        # Return output directory path if it exists
        if output_path.exists():
            return EvaluationJobFilesLocation(
                job_id=job_id,
                path=str(output_path),
                metadata={
                    "framework": "lighteval",
                    "benchmark_id": job.request.benchmark_id,
                    "model_name": job.request.model.name,
                },
            )

        return EvaluationJobFilesLocation(job_id=job_id, path=None)


def create_lighteval_adapter() -> LightEvalAdapter:
    """Factory function to create LightEval adapter."""
    config = AdapterConfig(
        framework_id="lighteval",
        adapter_name="LightEval Framework Adapter",
        version="1.0.0",
        framework_config={
            "cache_dir": "/tmp/lighteval_cache",
            "output_dir": "/tmp/lighteval_output",
            "max_samples": 1000,
            "batch_size": 1,
        },
    )

    return LightEvalAdapter(config)


# CLI for running the adapter server
if __name__ == "__main__":
    import sys

    from evalhub.adapter.server.app import AdapterServer

    # Create adapter
    adapter = create_lighteval_adapter()

    # Update config from environment or command line
    if len(sys.argv) > 1:
        adapter.config.port = int(sys.argv[1])

    # Create and run server
    server = AdapterServer(adapter)
    server.run()
