# MLflow Integration Guide for Adapters

This guide explains how to integrate MLflow tracking into your EvalHub adapters to ensure artifacts persist, runs are traceable, and the dashboard can embed MLflow UI.

## Overview

When implementing an EvalHub adapter, integrating MLflow provides:

- **Artifact Persistence**: Reports, plots, and results survive pod termination
- **Run Traceability**: Complete lineage from evaluation job to MLflow artifacts
- **Dashboard Integration**: MLflow UI can be embedded in the EvalHub dashboard
- **Experiment Organization**: Multiple runs grouped under experiments

## Quick Start

### 1. Read `mlflow_experiment_id` from JobSpec

The EvalHub server creates an MLflow experiment when the job is submitted and passes the `mlflow_experiment_id` to your adapter via the job spec:

```python
from evalhub.adapter import FrameworkAdapter, JobSpec

class MyAdapter(FrameworkAdapter):
    def run_benchmark_job(self, config: JobSpec, callbacks):
        # Check if MLflow tracking is enabled
        if config.mlflow_experiment_id:
            print(f"MLflow experiment ID: {config.mlflow_experiment_id}")
            # Initialize MLflow tracking...
```

### 2. Create MLflow Client

Use the SDK's built-in `MlflowClient`:

```python
from evalhub.adapter.mlflow import MlflowClient

# Client automatically reads configuration from environment variables:
# - MLFLOW_TRACKING_URI
# - MLFLOW_TRACKING_TOKEN (or MLFLOW_TRACKING_TOKEN_PATH)
# - MLFLOW_WORKSPACE (for multi-tenant deployments)
mlflow_client = MlflowClient()
```

### 3. Start MLflow Run

Create a run within the experiment:

```python
with mlflow_client.start_run(
    experiment_id=config.mlflow_experiment_id,
    run_name=f"{config.benchmark_id}_{config.benchmark_index}",
    tags={
        "job_id": config.id,
        "benchmark_id": config.benchmark_id,
        "model_name": config.model.name,
    }
) as mlflow_run_id:
    # Run evaluation and log to MLflow
    results = run_evaluation(...)

    # Log metrics
    mlflow_client.log_batch(
        mlflow_run_id,
        metrics=[Metric("accuracy", 0.95), Metric("f1", 0.92)]
    )

    # Log parameters
    mlflow_client.log_batch(
        mlflow_run_id,
        params=[Param("model_name", config.model.name)]
    )

    # Upload artifacts
    mlflow_client.upload_artifact_file(
        mlflow_run_id,
        "results/report.json",
        "/tmp/report.json"
    )
```

### 4. Report `mlflow_run_id` in Results

Include the `mlflow_run_id` in your `JobResults`:

```python
from evalhub.adapter import JobResults

return JobResults(
    id=config.id,
    benchmark_id=config.benchmark_id,
    benchmark_index=config.benchmark_index,
    model_name=config.model.name,
    results=evaluation_results,
    mlflow_run_id=mlflow_run_id,  # ✅ Critical!
    # ... other fields
)
```

## Complete Example

See `mlflow_integrated_adapter.py` for a complete working example that demonstrates:

- Initializing MLflow client
- Creating runs with proper tags
- Logging metrics, parameters, and artifacts
- Handling cases where MLflow is not available
- Reporting `mlflow_run_id` in results

## Data Flow

```
1. User submits job → EvalHub server creates MLflow experiment
2. Server stores mlflow_experiment_id in job record
3. Runtime generates JobSpec with mlflow_experiment_id
4. Adapter reads mlflow_experiment_id from JobSpec
5. Adapter creates MLflow run using experiment ID
6. Adapter logs metrics/artifacts to run
7. Adapter reports mlflow_run_id in JobResults
8. Server stores mlflow_run_id in benchmark results
9. GET /jobs/{id} returns mlflow_run_id to client
10. Dashboard embeds MLflow UI using mlflow_run_id
```

## Best Practices

### 1. Always Check for MLflow Availability

```python
mlflow_client = None
mlflow_run_id = None

if config.mlflow_experiment_id:
    try:
        mlflow_client = MlflowClient()
    except Exception as e:
        logger.warning(f"MLflow unavailable: {e}. Continuing without tracking.")
```

### 2. Use Descriptive Run Names

```python
run_name = f"{benchmark_id}_{benchmark_index}_{timestamp}"
```

### 3. Add Comprehensive Tags

```python
tags = {
    "job_id": config.id,
    "benchmark_id": config.benchmark_id,
    "benchmark_index": str(config.benchmark_index),
    "provider_id": config.provider_id,
    "model_name": config.model.name,
    "model_url": config.model.url,
}
```

### 4. Log Configuration as Parameters

```python
mlflow_client.log_batch(
    run_id,
    params=[
        Param("benchmark_id", config.benchmark_id),
        Param("model_name", config.model.name),
        Param("num_examples", str(config.num_examples or "all")),
    ]
)

# Log benchmark-specific parameters
for key, value in config.parameters.items():
    mlflow_client.log_param(run_id, f"benchmark.{key}", str(value))
```

### 5. Upload All Artifacts

```python
# Upload individual files
mlflow_client.upload_artifact_file(run_id, "results/metrics.json", metrics_file)
mlflow_client.upload_artifact_file(run_id, "results/report.html", report_file)

# Upload plots
mlflow_client.upload_artifact_file(run_id, "plots/accuracy.png", plot_file)
```

### 6. Handle Failures Gracefully

```python
try:
    # Run evaluation
    results = evaluate(...)
except Exception as e:
    # Mark MLflow run as failed
    if mlflow_client and mlflow_run_id:
        mlflow_client.set_tag(mlflow_run_id, "status", "failed")
        mlflow_client.set_tag(mlflow_run_id, "error", str(e))
    raise
```

## Environment Variables

The MLflow client automatically configures itself from these environment variables (set by the EvalHub runtime):

| Variable | Description | Example |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `https://mlflow.example.com` |
| `MLFLOW_TRACKING_TOKEN` | Authentication token | `Bearer xyz...` |
| `MLFLOW_TRACKING_TOKEN_PATH` | Path to token file | `/var/run/secrets/mlflow/token` |
| `MLFLOW_WORKSPACE` | Workspace/tenant ID | `my-team` |
| `MLFLOW_TRACKING_SERVER_CERT_PATH` | CA certificate path | `/etc/pki/ca-trust/...` |

You typically don't need to set these manually - the runtime configures them.

## Testing Locally

### 1. Start Local MLflow Server

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

### 2. Set Environment Variables

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_ID=0  # Default experiment
```

### 3. Create Test JobSpec

```json
{
  "id": "test-job-123",
  "provider_id": "my_provider",
  "benchmark_id": "test_benchmark",
  "benchmark_index": 0,
  "model": {
    "url": "http://localhost:8000/v1",
    "name": "test-model"
  },
  "parameters": {},
  "callback_url": "http://localhost:8080",
  "mlflow_experiment_id": "0"
}
```

### 4. Run Adapter

```bash
python examples/simple_adapter/mlflow_integrated_adapter.py
```

### 5. Verify in MLflow UI

Open http://localhost:5000 and verify:
- Run appears in experiment
- Metrics are logged
- Parameters are logged
- Artifacts are uploaded

## Troubleshooting

### MLflow Client Initialization Fails

**Error**: `ValueError: MLflow tracking URI is required`

**Solution**: Ensure `MLFLOW_TRACKING_URI` environment variable is set

### Artifacts Not Appearing

**Symptom**: Metrics logged but no artifacts in MLflow UI

**Cause**: Forgot to call `upload_artifact_file()`

**Solution**:
```python
# After saving results to file
mlflow_client.upload_artifact_file(run_id, "results/report.json", report_path)
```

### `mlflow_run_id` Not in Results

**Symptom**: Dashboard cannot embed MLflow UI

**Cause**: Forgot to include `mlflow_run_id` in `JobResults`

**Solution**:
```python
return JobResults(
    # ... other fields
    mlflow_run_id=mlflow_run_id,  # Must be included!
)
```

### Authentication Failures

**Error**: `401 Unauthorized` when creating runs

**Cause**: Missing or invalid MLflow token

**Solution**: Verify `MLFLOW_TRACKING_TOKEN` or `MLFLOW_TRACKING_TOKEN_PATH` is set correctly

## Migration from Non-MLflow Adapters

If you have an existing adapter without MLflow:

1. **Add MLflow check**:
   ```python
   if config.mlflow_experiment_id:
       # Enable MLflow tracking
   ```

2. **Initialize client**:
   ```python
   mlflow_client = MlflowClient()
   ```

3. **Wrap evaluation in `start_run()`**:
   ```python
   with mlflow_client.start_run(...) as run_id:
       results = your_existing_evaluation_code()
       mlflow_client.log_batch(run_id, metrics=...)
   ```

4. **Upload artifacts**:
   ```python
   for artifact_file in your_output_files:
       mlflow_client.upload_artifact_file(run_id, artifact_path, artifact_file)
   ```

5. **Return `mlflow_run_id`**:
   ```python
   return JobResults(..., mlflow_run_id=run_id)
   ```

## Related Issues

- **RHOAIENG-54869**: mlflow_run_id missing from API response
- **RHOAIENG-54539**: Artifacts not written to MLflow

## References

- [MLflow Tracking Documentation](https://mlflow.org/docs/latest/tracking.html)
- [EvalHub SDK API Reference](../../src/evalhub/adapter/)
- [Example MLflow-Integrated Adapter](./mlflow_integrated_adapter.py)
