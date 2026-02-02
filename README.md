# EvalHub SDK

**Framework Adapter SDK for EvalHub Integration**

The EvalHub SDK provides a standardized way to create framework adapters that can be consumed by EvalHub, enabling a "Bring Your Own Framework" (BYOF) approach for evaluation frameworks.

## Overview

The SDK creates a common API layer that allows EvalHub to communicate with ANY evaluation framework. Users only need to write minimal "glue" code to connect their framework to the standardized interface.

```
EvalHub → (Standard API) → Your Framework Adapter → Your Evaluation Framework
```

## Architecture

The adapter SDK uses a **job runner architecture**:

```mermaid
graph TB
    subgraph pod["Kubernetes Job Pod"]
        subgraph adapter["Adapter Container"]
            A1["1. Read JobSpec<br/>from ConfigMap"]
            A2["2. run_benchmark_job()"]
            A3["3. Report status<br/>via callbacks"]
            A4["4. Create OCI artifacts<br/>via callbacks"]
            A5["5. Report results<br/>via callbacks"]
            A6["6. Exit"]
        end

        subgraph sidecar["Sidecar Container"]
            S1["ConfigMap mounted<br/>/meta/job.json"]
            S2["Forward status to<br/>EvalHub service (HTTP)"]
            S4["Forward results to<br/>EvalHub service (HTTP)"]
        end

        A1 -.-> S1
        A3 --> S2
        A5 --> S4
    end

    S2 --> EvalHub["EvalHub Service"]
    S4 --> EvalHub
    A4 --> Registry["OCI Registry"]

    style pod fill:#f0f0f0,stroke:#333,stroke-width:2px
    style adapter fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style sidecar fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

### Package Organization

The SDK is organized into distinct, focused packages:

**Core (`evalhub.models`)** - Shared data models
- Request/response models for API communication
- Common data structures for evaluations and benchmarks

**Adapter SDK (`evalhub.adapter`)** - Framework adapter components
- `FrameworkAdapter` base class with `run_benchmark_job()` method
- Job specification models (`JobSpec`, `JobResults`)
- Callback interface for status updates and OCI artifacts
- Example implementations

**Client SDK (`evalhub.client`)** - REST API client for EvalHub service
- HTTP client for submitting evaluations to EvalHub
- Resource navigation (providers, benchmarks, collections)
- See [CLIENT_SDK_GUIDE.md](./CLIENT_SDK_GUIDE.md)

### Key Components

1. **JobSpec** - Job configuration loaded from ConfigMap at pod startup
2. **FrameworkAdapter** - Base class that implements `run_benchmark_job()` method
3. **JobCallbacks** - Interface for reporting status and persisting artifacts
4. **JobResults** - Evaluation results returned when job completes
5. **Sidecar** - Container that handles service communication (provided by platform)

## Quick Start

### 1. Installation

```bash
# Install from PyPI (when available)
pip install eval-hub-sdk

# Install from source
git clone https://github.com/eval-hub/eval-hub-sdk.git
cd eval-hub-sdk
pip install -e .[dev]
```

### 2. Create Your Adapter

Create a new Python file for your adapter:

```python
# my_framework_adapter.py
from evalhub.adapter import (
    FrameworkAdapter,
    JobSpec,
    JobCallbacks,
    JobResults,
    JobStatus,
    JobPhase,
    JobStatusUpdate,
    EvaluationResult,
)

class MyFrameworkAdapter(FrameworkAdapter):
    def run_benchmark_job(
        self, config: JobSpec, callbacks: JobCallbacks
    ) -> JobResults:
        """Run a benchmark evaluation job."""

        # Report initialization
        callbacks.report_status(JobStatusUpdate(
            status=JobStatus.RUNNING,
            phase=JobPhase.INITIALIZING,
            progress=0.0,
            message="Loading benchmark and model"
        ))

        # Load your evaluation framework and benchmark
        framework = load_your_framework()
        benchmark = framework.load_benchmark(config.benchmark_id)
        model = framework.load_model(config.model)

        # Report evaluation start
        callbacks.report_status(JobStatusUpdate(
            status=JobStatus.RUNNING,
            phase=JobPhase.RUNNING_EVALUATION,
            progress=0.3,
            message=f"Evaluating on {config.num_examples} examples"
        ))

        # Run evaluation (adapter-specific params come from benchmark_config)
        results = framework.evaluate(
            benchmark=benchmark,
            model=model,
            num_examples=config.num_examples,
            num_few_shot=config.benchmark_config.get("num_few_shot", 0)
        )

        # Save and persist artifacts
        output_files = save_results(config.job_id, results)
        artifact = callbacks.create_oci_artifact(OCIArtifactSpec(
            files=output_files,
            job_id=config.job_id,
            benchmark_id=config.benchmark_id,
            model_name=config.model.name
        ))

        # Return results
        return JobResults(
            job_id=config.job_id,
            benchmark_id=config.benchmark_id,
            model_name=config.model.name,
            results=[
                EvaluationResult(
                    metric_name="accuracy",
                    metric_value=results["accuracy"],
                    metric_type="float"
                )
            ],
            num_examples_evaluated=len(results),
            duration_seconds=results["duration"],
            oci_artifact=artifact
        )
```

### 3. OCI Artifact Persistence

The SDK exposes an OCI persistence API via `callbacks.create_oci_artifact(...)`.

Note: in this POC the underlying persister is currently a **placeholder/no-op** implementation (it logs what it would do and returns a dummy digest). This is still useful for adapter development because it keeps the interface stable while storage is implemented.

#### Using DefaultCallbacks

Use `DefaultCallbacks` for both production and development:

```python
from evalhub.adapter import AdapterSettings, DefaultCallbacks, JobSpec

# Load settings and job spec explicitly
settings = AdapterSettings.from_env()
settings.validate_runtime()
job_spec = JobSpec.from_file(settings.resolved_job_spec_path)

# Initialize adapter with settings
adapter = MyFrameworkAdapter(settings=settings)

callbacks = DefaultCallbacks(
    job_id=job_spec.job_id,
    sidecar_url=str(settings.service_url),  # SERVICE_URL
    registry_url=settings.registry_url,      # REGISTRY_URL
    registry_username=settings.registry_username,
    registry_password=settings.registry_password,
    insecure=settings.registry_insecure,     # REGISTRY_INSECURE (true/false)
)

results = adapter.run_benchmark_job(job_spec, callbacks)
```

**Key Points:**
- **Status updates**: Sent to sidecar if `sidecar_url` is provided, otherwise logged locally
- **OCI artifacts**: Always pushed directly by the SDK using `OCIArtifactPersister`

#### Advanced: Direct Persister Usage

The OCI functionality follows the `Persister` protocol. You can use `OCIArtifactPersister` directly or implement your own:

```python
from evalhub.adapter import OCIArtifactPersister, OCIArtifactSpec, Persister
from pathlib import Path

# Use the default implementation
persister: Persister = OCIArtifactPersister(
    registry_url="ghcr.io",
    username="user",
    password="token"
)

result = persister.persist(
    OCIArtifactSpec(
        files=[Path("results.json"), Path("metrics.csv")],
        job_id="job-123",
        benchmark_id="mmlu",
        model_name="llama-2-7b",
        title="MMLU Evaluation Results",
        annotations={"score": "0.85"}
    )
)

print(f"Pushed to: {result.reference}")
print(f"Digest: {result.digest}")
```

**Custom Persister**: Implement your own `Persister` for custom storage backends:

```python
from evalhub.adapter import Persister, OCIArtifactSpec, OCIArtifactResult

class S3Persister:
    """Custom persister that stores artifacts in S3."""

    def persist(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
        # Upload files to S3
        s3_url = self.upload_to_s3(spec.files)
        return OCIArtifactResult(
            digest=compute_digest(spec.files),
            reference=s3_url,
            size_bytes=compute_size(spec.files)
        )
```

**Note**: OCI pushing is not yet implemented in this POC; the persister returns mock results.

### 4. Containerise Your Adapter

Create a Dockerfile for your adapter:

```dockerfile
FROM registry.access.redhat.com/ubi9/python-312

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy adapter code
COPY my_framework_adapter.py .
COPY run_adapter.py .

# Run adapter
CMD ["python", "run_adapter.py"]
```

Create the entrypoint script:

```python
# run_adapter.py
from my_framework_adapter import MyFrameworkAdapter
from evalhub.adapter import AdapterSettings, DefaultCallbacks, JobSpec

# Load settings and job spec explicitly
settings = AdapterSettings.from_env()
settings.validate_runtime()
job_spec = JobSpec.from_file(settings.resolved_job_spec_path)

# Initialize adapter with settings
adapter = MyFrameworkAdapter(settings=settings)

# Create callbacks
callbacks = DefaultCallbacks(
    job_id=job_spec.job_id,
    sidecar_url=str(settings.service_url),
    registry_url=settings.registry_url,
    registry_username=settings.registry_username,
    registry_password=settings.registry_password,
    insecure=settings.registry_insecure,
)

# Run adapter
results = adapter.run_benchmark_job(job_spec, callbacks)

# Report final results to service via sidecar
callbacks.report_results(results)

print(f"Job completed: {results.job_id}")
```

### 4. Deploy to Kubernetes

The eval-hub service will create Kubernetes Jobs for your adapter:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: eval-job-123
spec:
  template:
    spec:
      containers:
      # Your adapter container
      - name: adapter
        image: myregistry/my-adapter:latest
        volumeMounts:
        - name: job-spec
          mountPath: /meta
      # Sidecar container (provided by platform)
      - name: sidecar
        image: evalhub/sidecar:latest
        env:
        - name: EVALHUB_SERVICE_URL
          value: "http://evalhub-service:8080"
      volumes:
      - name: job-spec
        configMap:
          name: job-123-spec
```

For a complete working example, see `evalhub/adapter/examples/simple_adapter.py`.

## Package Organization Guide

The EvalHub SDK is organized into distinct packages based on your use case:

### Which Package Should I Use?

| Use Case | Primary Package | Description |
|----------|----------------|-------------|
| **Building an Adapter** | `evalhub.adapter` | Create a framework adapter for your evaluation framework |
| **Interacting with EvalHub** | `evalhub.client` | REST API client for submitting evaluations |
| **Data Models** | `evalhub.models` | Request/response models for API communication |

### Import Patterns

**Framework Adapter Developer:**
```python
# Building your adapter
from evalhub.adapter import (
    FrameworkAdapter,
    JobSpec,
    JobCallbacks,
    JobResults,
    JobStatus,
    JobPhase,
    JobStatusUpdate,
    EvaluationResult,
    OCIArtifactSpec,
)
```

**EvalHub Service User:**
```python
# Interacting with EvalHub REST API
from evalhub.client import EvalHubClient
from evalhub.models.api import ModelConfig, EvaluationRequest
```

## Complete Example

The SDK includes a complete reference implementation showing all adapter patterns:

**Example Adapter**: `src/evalhub/adapter/examples/simple_adapter.py`

This example demonstrates:
- Loading JobSpec from mounted ConfigMap
- Validating configuration
- Loading benchmark data
- Running evaluation with progress reporting
- Persisting results as OCI artifacts
- Returning structured results

### Using the Example

```python
from evalhub.adapter.examples import ExampleAdapter
from evalhub.adapter import JobSpec

# Load job specification
job_spec = JobSpec(
    job_id="eval-123",
    benchmark_id="mmlu",
    model=ModelConfig(
        url="http://vllm-service:8000",
        name="llama-2-7b"
    ),
    benchmark_config={},
    callback_url="http://localhost:8080",
    num_examples=100
)

# Create adapter and run
adapter = ExampleAdapter()
results = adapter.run_benchmark_job(job_spec, callbacks)
```

## Framework Adapter Interface

Your adapter must implement a single method:

```python
from evalhub.adapter import FrameworkAdapter, JobSpec, JobCallbacks, JobResults

class MyFrameworkAdapter(FrameworkAdapter):
    def run_benchmark_job(
        self, config: JobSpec, callbacks: JobCallbacks
    ) -> JobResults:
        """Run a benchmark evaluation job.

        Args:
            config: Job specification from mounted ConfigMap
            callbacks: Callbacks for status updates and artifact persistence

        Returns:
            JobResults: Evaluation results and metadata

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If evaluation fails
        """
        # Your implementation here
        pass
```

### Key Data Models

**JobSpec** - Configuration loaded from ConfigMap:
```python
class JobSpec(BaseModel):
    # Mandatory fields
    job_id: str                       # Unique job identifier
    benchmark_id: str                 # Benchmark to evaluate
    model: ModelConfig                # Model configuration (url, name)
    benchmark_config: Dict[str, Any]  # Adapter-specific parameters
    callback_url: str                 # Base URL for callbacks (SDK appends /status, /results)

    # Optional fields
    num_examples: Optional[int]       # Number of examples to evaluate
    experiment_name: Optional[str]    # Experiment name
    tags: Dict[str, str]              # Custom tags (default: {})
    timeout_seconds: Optional[int]    # Max execution time (default: 3600)
    retry_attempts: Optional[int]     # Number of retry attempts on failure

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Load JobSpec from a JSON file."""
```

Load a job spec from file:
```python
from evalhub.adapter import JobSpec

# Explicit path (recommended)
spec = JobSpec.from_file("/meta/job.json")

# Or use settings for the path
spec = JobSpec.from_file(settings.resolved_job_spec_path)
```

**JobCallbacks** - Interface for service communication:
```python
class JobCallbacks(ABC):
    @abstractmethod
    def report_status(self, update: JobStatusUpdate) -> None:
        """Report status update to service"""

    @abstractmethod
    def create_oci_artifact(self, spec: OCIArtifactSpec) -> OCIArtifactResult:
        """Create and push OCI artifact"""
```

**JobResults** - Returned when job completes:
```python
class JobResults(BaseModel):
    job_id: str
    benchmark_id: str
    model_name: str
    results: List[EvaluationResult]           # Evaluation metrics
    overall_score: Optional[float]            # Overall score if applicable
    num_examples_evaluated: int               # Number of examples evaluated
    duration_seconds: float                   # Total evaluation time
    evaluation_metadata: Dict[str, Any]       # Framework-specific metadata
    oci_artifact: Optional[OCIArtifactResult] # OCI artifact info if persisted
```

## Deployment

### Container Structure

Your adapter runs as a container in a Kubernetes Job alongside a sidecar:

```dockerfile
FROM registry.access.redhat.com/ubi9/python-312

WORKDIR /app

# Install your framework and dependencies
RUN pip install lm-evaluation-harness==0.4.0 eval-hub-sdk

# Copy adapter implementation
COPY my_adapter.py .
COPY entrypoint.py .

CMD ["python", "entrypoint.py"]
```

### Entrypoint Script

```python
# entrypoint.py
from my_adapter import MyFrameworkAdapter
from evalhub.adapter import AdapterSettings, DefaultCallbacks, JobSpec

# Load settings and job spec explicitly
settings = AdapterSettings.from_env()
settings.validate_runtime()
job_spec = JobSpec.from_file(settings.resolved_job_spec_path)

# Initialize adapter with settings
adapter = MyFrameworkAdapter(settings=settings)

# Create callbacks
callbacks = DefaultCallbacks(
    job_id=job_spec.job_id,
    sidecar_url=str(settings.service_url),
    registry_url=settings.registry_url,
    insecure=settings.registry_insecure,
)

# Run adapter
results = adapter.run_benchmark_job(job_spec, callbacks)

# Report final results
callbacks.report_results(results)

print(f"Job {results.job_id} completed with score: {results.overall_score}")
```

### Kubernetes Job

EvalHub creates Jobs automatically:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: eval-job-123
spec:
  template:
    spec:
      containers:
      - name: adapter
        image: myregistry/my-framework-adapter:latest
        volumeMounts:
        - name: job-spec
          mountPath: /meta
      - name: sidecar
        image: evalhub/sidecar:latest
        env:
        - name: EVALHUB_SERVICE_URL
          value: "http://evalhub-service:8080"
      volumes:
      - name: job-spec
        configMap:
          name: job-123-spec
      restartPolicy: Never
```

## Development

### Setting Up Development Environment
### Development Setup

```bash
# Clone the repository
git clone https://github.com/eval-hub/eval-hub-sdk.git
cd eval-hub-sdk

# Install in development mode with all dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/evalhub --cov-report=html

# Run type checking
mypy src/evalhub

# Run linting
ruff check src/ tests/
ruff format src/ tests/
```

### Testing Your Adapter

```python
from evalhub.adapter import AdapterSettings

def test_settings_parse(monkeypatch):
    monkeypatch.setenv("EVALHUB_MODE", "local")
    monkeypatch.setenv("SERVICE_URL", "http://localhost:8080")
    monkeypatch.setenv("REGISTRY_URL", "localhost:5000")
    s = AdapterSettings.from_env()
    assert str(s.service_url) == "http://localhost:8080/"
```

### Quality Assurance


Run all quality checks:
```bash
# Format code
ruff format .

# Lint and fix issues
ruff check --fix .

# Type check
mypy src/evalhub

# Run full test suite
pytest -v --cov=src/evalhub
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
