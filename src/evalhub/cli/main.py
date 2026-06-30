"""EvalHub CLI entry point and command groups."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import click
import yaml

import evalhub
from evalhub.models import (
    BenchmarkConfig,
    CollectionCreateRequest,
    EvaluationExports,
    EvaluationExportsOCI,
    ExperimentConfig,
    JobStatus,
    JobSubmissionRequest,
    ModelConfig,
    OCIConnectionConfig,
    OCICoordinates,
    ProviderCreateRequest,
    QueueConfig,
    S3TestDataRef,
    TestDataRef,
)

from . import config as cfg
from .client import get_client, handle_api_errors
from .completion import completion
from .formatter import format_option, output
from .mcp_cmd import mcp
from .server_cmd import server


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=evalhub.__version__, prog_name="evalhub")
@click.option(
    "--profile",
    default=None,
    envvar="EVALHUB_PROFILE",
    help="Configuration profile to use (overrides active profile).",
)
@click.option(
    "--base-url",
    default=None,
    envvar="EVALHUB_BASE_URL",
    help="EvalHub server URL (overrides profile config).",
)
@click.option(
    "--token",
    default=None,
    envvar="EVALHUB_TOKEN",
    help="Authentication token (overrides profile config).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    envvar="EVALHUB_VERBOSE",
    help="Enable verbose output (show SDK logs).",
)
@click.pass_context
def main(
    ctx: click.Context,
    profile: str | None,
    base_url: str | None,
    token: str | None,
    verbose: bool,
) -> None:
    """EvalHub CLI - manage evaluations, providers, collections, and configuration.

    \b
    Quick start:
      evalhub config set base_url http://localhost:8080
      evalhub config set token my-api-token
      evalhub config set tenant my-tenant
      evalhub health
      evalhub eval run --config eval.yaml
      evalhub eval status
    """
    if not verbose:
        logging.getLogger("evalhub").setLevel(logging.ERROR)
    ctx.ensure_object(dict)
    ctx.obj["profile"] = profile
    ctx.obj["base_url"] = base_url
    ctx.obj["token"] = token


main.add_command(completion)
main.add_command(mcp)
main.add_command(server)


@main.command()
def version() -> None:
    """Print the EvalHub CLI version.

    \b
    Examples:
      evalhub version
    """
    click.echo(f"evalhub {evalhub.__version__}")


@main.group()
def eval() -> None:
    """Submit and manage evaluation jobs.

    \b
    Use 'eval run' to submit a new evaluation, 'eval status' to track
    progress, 'eval results' to fetch outcomes, and 'eval cancel' to
    abort a running job.

    \b
    Examples:
      evalhub eval run --config eval.yaml
      evalhub eval status
      evalhub eval results eval-123
      evalhub eval cancel eval-123
    """


def _coerce_param_value(value: str) -> Any:
    """Coerce a CLI parameter string to its most appropriate Python type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _load_config_file(path: str) -> dict[str, Any]:
    """Load a YAML or JSON config file for eval run."""
    with open(path) as f:
        content = f.read()
    if path.endswith((".yaml", ".yml")):
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)
    if not isinstance(data, dict):
        raise click.ClickException(
            f"Config file must contain a mapping, got {type(data).__name__}"
        )
    return data


def _build_request_from_flags(
    name: str,
    model_url: str,
    model_name: str,
    provider: str,
    benchmark: tuple[str, ...],
    description: str | None,
    metrics: tuple[str, ...],
    dataset: str | None,
    experiment: ExperimentConfig | None = None,
    exports: EvaluationExports | None = None,
    extra_params: dict[str, Any] | None = None,
    queue: QueueConfig | None = None,
    test_data_ref: TestDataRef | None = None,
) -> JobSubmissionRequest:
    """Build a JobSubmissionRequest from CLI flags."""
    parameters: dict[str, Any] = {}
    if extra_params:
        parameters.update(extra_params)
    if metrics:
        parameters["metrics"] = list(metrics)
    if dataset:
        parameters["dataset"] = dataset
    benchmarks = [
        BenchmarkConfig(
            id=b,
            provider_id=provider,
            parameters=parameters,
            test_data_ref=test_data_ref,
        )
        for b in benchmark
    ]
    return JobSubmissionRequest(
        name=name,
        description=description,
        model=ModelConfig(url=model_url, name=model_name),
        benchmarks=benchmarks,
        experiment=experiment,
        exports=exports,
        queue=queue,
    )


@eval.command("run")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help=(
        "YAML or JSON file with the full job body. When set, all other "
        "job-related flags on this command are ignored; use --wait, "
        "--timeout, --poll-interval, and --format with --config as needed."
    ),
)
@click.option("--name", default=None, help="Job name (required if not using --config).")
@click.option("--model-url", default=None, help="Model endpoint URL.")
@click.option("--model-name", default=None, help="Model name or identifier.")
@click.option("--provider", default=None, help="Evaluation provider ID.")
@click.option("--benchmark", "-b", multiple=True, help="Benchmark ID (repeatable).")
@click.option(
    "--metric", "-m", "metrics", multiple=True, help="Metric name (repeatable)."
)
@click.option("--dataset", default=None, help="Dataset identifier or path.")
@click.option("--description", default=None, help="Job description.")
@click.option(
    "--param",
    "-p",
    "params",
    multiple=True,
    help=(
        "Benchmark parameter as key=value (repeatable). "
        "Example: --param tokenizer=my-tokenizer --param batch_size=3"
    ),
)
@click.option(
    "--experiment",
    "experiment_name",
    default=None,
    help="MLflow experiment name (only when using inline flags, not with --config).",
)
@click.option(
    "--oci-host",
    default=None,
    help="OCI registry host for exports (e.g. quay.io; inline flags only).",
)
@click.option(
    "--oci-repository",
    default=None,
    help="OCI repository path (org/repo; use with --oci-host).",
)
@click.option(
    "--oci-connection",
    default=None,
    help="Kubernetes Secret name for registry auth (optional; use with OCI host/repo).",
)
@click.option(
    "--queue",
    default=None,
    help="Kueue LocalQueue name for workload scheduling (inline flags only).",
)
@click.option(
    "--test-data-s3-bucket",
    default=None,
    help="S3 bucket name for custom test data (inline flags only).",
)
@click.option(
    "--test-data-s3-key",
    default=None,
    help="S3 object key or prefix for custom test data (inline flags only).",
)
@click.option(
    "--test-data-s3-secret",
    default=None,
    help="Kubernetes Secret name with S3 credentials for custom test data (inline flags only).",
)
@click.option(
    "--wait", "wait_for", is_flag=True, default=False, help="Block until job completes."
)
@click.option(
    "--timeout", type=float, default=None, help="Timeout in seconds when using --wait."
)
@click.option(
    "--poll-interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Poll interval in seconds when using --wait.",
)
@format_option()
@click.pass_context
@handle_api_errors
def eval_run(
    ctx: click.Context,
    config_file: str | None,
    name: str | None,
    model_url: str | None,
    model_name: str | None,
    provider: str | None,
    benchmark: tuple[str, ...],
    metrics: tuple[str, ...],
    dataset: str | None,
    description: str | None,
    params: tuple[str, ...],
    experiment_name: str | None,
    oci_host: str | None,
    oci_repository: str | None,
    oci_connection: str | None,
    queue: str | None,
    test_data_s3_bucket: str | None,
    test_data_s3_key: str | None,
    test_data_s3_secret: str | None,
    wait_for: bool,
    timeout: float | None,
    poll_interval: float,
    output_format: str,
) -> None:
    """Submit an evaluation job.

    Use --config to submit from a YAML/JSON file, or specify flags inline.
    Do not mix: with --config, the file is the full job body; all other
    job-related options apply only to the inline-flags path.

    \b
    Examples:
      evalhub eval run --config eval.yaml
      evalhub eval run --config eval.yaml --wait
      evalhub eval run --name my-eval --model-url http://vllm:8000/v1 \\
          --model-name llama3 --provider lm_evaluation_harness -b mmlu -b hellaswag
      evalhub eval run --name my-eval --model-url http://vllm:8000/v1 \\
          --model-name llama3 --provider lm_evaluation_harness -b mmlu \\
          --experiment my-experiment
      evalhub eval run --name my-eval --model-url http://vllm:8000/v1 \\
          --model-name llama3 --provider guidellm -b quick_perf_test \\
          --oci-host quay.io --oci-repository myorg/myrepo --oci-connection my-oci-secret
      evalhub eval run --name my-eval --model-url http://vllm:8000/v1 \\
          --model-name llama3 --provider lm_evaluation_harness -b mmlu \\
          --param tokenizer=my-tokenizer --param batch_size=3
      evalhub eval run --name my-eval --model-url http://vllm:8000/v1 \\
          --model-name llama3 --provider lm_evaluation_harness -b mmlu \\
          --queue my-local-queue
      evalhub eval run --name my-eval --model-url http://vllm:8000/v1 \\
          --model-name llama3 --provider lm_evaluation_harness -b your_benchmark_id \\
          --test-data-s3-bucket evalhub-test --test-data-s3-key dataset/ \\
          --test-data-s3-secret evalhub-s3-credentials
    """
    client = get_client(ctx)

    if config_file:
        data = _load_config_file(config_file)
        request = JobSubmissionRequest(**data)
    else:
        if not all([name, model_url, model_name, provider, benchmark]):
            raise click.ClickException(
                "Either --config or all of --name, --model-url, --model-name, "
                "--provider, and --benchmark are required."
            )
        experiment: ExperimentConfig | None = None
        if experiment_name:
            experiment = ExperimentConfig(name=experiment_name)

        oci_flags = (oci_host, oci_repository, oci_connection)
        if any(oci_flags) and not (oci_host and oci_repository):
            raise click.ClickException(
                "OCI export requires --oci-host and --oci-repository "
                "(--oci-connection is optional)."
            )
        exports: EvaluationExports | None = None
        if oci_host and oci_repository:
            k8s = (
                OCIConnectionConfig(connection=oci_connection)
                if oci_connection
                else None
            )
            exports = EvaluationExports(
                oci=EvaluationExportsOCI(
                    coordinates=OCICoordinates(
                        oci_host=oci_host, oci_repository=oci_repository
                    ),
                    k8s=k8s,
                )
            )
        extra_params: dict[str, Any] = {}
        for p in params:
            if "=" not in p:
                raise click.ClickException(
                    f"Invalid --param format: {p!r}. Expected key=value."
                )
            key, value = p.split("=", 1)
            extra_params[key] = _coerce_param_value(value)
        queue_config: QueueConfig | None = None
        if queue:
            queue_config = QueueConfig(name=queue)
        s3_flags = (test_data_s3_bucket, test_data_s3_key, test_data_s3_secret)
        if any(s3_flags) and not all(s3_flags):
            raise click.ClickException(
                "S3 test data requires --test-data-s3-bucket, --test-data-s3-key, "
                "and --test-data-s3-secret to all be specified."
            )
        test_data_ref: TestDataRef | None = None
        if all(s3_flags):
            test_data_ref = TestDataRef(
                s3=S3TestDataRef(
                    bucket=cast(str, test_data_s3_bucket),
                    key=cast(str, test_data_s3_key),
                    secret_ref=cast(str, test_data_s3_secret),
                )
            )
        request = _build_request_from_flags(
            name=cast(str, name),
            model_url=cast(str, model_url),
            model_name=cast(str, model_name),
            provider=cast(str, provider),
            benchmark=benchmark,
            description=description,
            metrics=metrics,
            dataset=dataset,
            experiment=experiment,
            exports=exports,
            extra_params=extra_params,
            queue=queue_config,
            test_data_ref=test_data_ref,
        )

    job = client.jobs.submit(request)
    structured = output_format in ("json", "yaml")
    click.echo(f"Job submitted: {job.id}", err=structured)

    if wait_for:
        click.echo(f"Waiting for job {job.id} to complete...", err=structured)
        job = client.jobs.wait_for_completion(
            job.id, timeout=timeout, poll_interval=poll_interval
        )
        click.echo(
            f"Job {job.id} finished with state: {job.effective_state.value}",
            err=structured,
        )
        if job.effective_state == JobStatus.FAILED:
            ctx.exit(1)

    if structured:
        output([job.model_dump(mode="json")], output_format=output_format)


@eval.command("status")
@click.argument("job_id", required=False, default=None)
@click.option(
    "--status",
    "status_filter",
    type=click.Choice([s.value for s in JobStatus], case_sensitive=False),
    default=None,
    help="Filter by job status.",
)
@click.option("--limit", type=int, default=None, help="Maximum number of jobs to list.")
@click.option(
    "--provider", "provider_filter", default=None, help="Filter by provider ID."
)
@click.option(
    "--since",
    "since_filter",
    default=None,
    help="Only show jobs created within this window (e.g. '24h', '7d').",
)
@click.option(
    "--watch",
    is_flag=True,
    default=False,
    help="Watch for status changes (single job only).",
)
@click.option(
    "--poll-interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Poll interval in seconds when using --watch.",
)
@format_option()
@click.pass_context
@handle_api_errors
def eval_status(
    ctx: click.Context,
    job_id: str | None,
    status_filter: str | None,
    limit: int | None,
    provider_filter: str | None,
    since_filter: str | None,
    watch: bool,
    poll_interval: float,
    output_format: str,
) -> None:
    """Show job status or list all jobs.

    \b
    Examples:
      evalhub eval status                            # list all jobs
      evalhub eval status eval-123                   # show single job
      evalhub eval status --status running           # filter by status
      evalhub eval status --provider lm_eval --since 24h
      evalhub eval status eval-123 --watch           # watch until complete
    """
    if watch and job_id is None:
        raise click.UsageError("--watch requires a job ID.")

    client = get_client(ctx)

    if job_id is None:
        # List mode
        parsed_status = JobStatus(status_filter) if status_filter else None
        since_dt = _parse_since(since_filter) if since_filter else None
        jobs = client.jobs.list(status=parsed_status, limit=limit)

        # Client-side filters (server API only supports status + limit)
        if provider_filter:
            jobs = [
                j
                for j in jobs
                if j.benchmarks and j.benchmarks[0].provider_id == provider_filter
            ]
        if since_dt:
            jobs = [
                j
                for j in jobs
                if j.resource.created_at and j.resource.created_at >= since_dt
            ]

        rows = [
            {
                "id": j.id,
                "name": j.name,
                "state": j.state.value,
                "provider": j.benchmarks[0].provider_id if j.benchmarks else "",
                "benchmarks": len(j.benchmarks) if j.benchmarks else 0,
                "created": str(j.resource.created_at),
            }
            for j in jobs
        ]
        output(
            rows,
            output_format=output_format,
            columns=["id", "name", "state", "provider", "benchmarks", "created"],
        )
        return

    # Single job mode
    job = client.jobs.get(job_id)

    if watch:
        _watch_job(client, job_id, poll_interval)
        return

    if output_format in ("json", "yaml"):
        output([job.model_dump(mode="json")], output_format=output_format)
        return

    _print_job_detail(job)


def _parse_since(value: str) -> datetime:
    """Parse a duration string like '24h' or '7d' into an absolute UTC datetime."""
    m = re.fullmatch(r"(\d+)([hd])", value.strip())
    if not m:
        raise click.BadParameter(
            f"Invalid --since value {value!r}. Use e.g. '24h' or '7d'."
        )
    amount, unit = int(m.group(1)), m.group(2)
    delta = timedelta(hours=amount) if unit == "h" else timedelta(days=amount)
    return datetime.now(tz=UTC) - delta


def _watch_job(client: Any, job_id: str, poll_interval: float) -> None:
    """Poll a job until it reaches a terminal state."""
    terminal = {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.PARTIALLY_FAILED,
    }
    while True:
        job = client.jobs.get(job_id)
        benchmarks_status = ""
        if job.status and job.status.benchmarks:
            benchmarks = job.status.benchmarks
            done = sum(1 for b in benchmarks if b.state in terminal)
            phase_info = ""
            if len(benchmarks) == 1 and benchmarks[0].phase:
                phase_info = f" phase={benchmarks[0].phase.value}"
            benchmarks_status = f" [{done}/{len(benchmarks)} benchmarks{phase_info}]"
        line = f"{job.id}: {job.effective_state.value}{benchmarks_status}"
        cols = shutil.get_terminal_size().columns
        click.echo(f"\r{line:<{cols}}", nl=False)
        sys.stdout.flush()
        if job.effective_state in terminal:
            click.echo()
            _print_job_detail(job)
            return
        time.sleep(poll_interval)


def _print_job_detail(job: Any) -> None:
    """Print detailed job information in human-readable format."""
    click.echo(f"Job:     {job.id}")
    click.echo(f"Name:    {job.name}")
    click.echo(f"State:   {job.effective_state.value}")
    if job.description:
        click.echo(f"Desc:    {job.description}")
    click.echo(f"Model:   {job.model.name} ({job.model.url})")
    click.echo(f"Created: {job.resource.created_at}")

    if job.status and job.status.benchmarks:
        click.echo(f"\nBenchmarks ({len(job.status.benchmarks)}):")
        for b in job.status.benchmarks:
            line = f"  {b.id} ({b.provider_id}): {b.state.value}"
            if b.phase:
                line += f" [{b.phase.value}]"
            if b.error_message:
                line += f" - {b.error_message.message}"
            click.echo(line)

    if job.status and job.status.message:
        click.echo(f"\nMessage: {job.status.message.message}")


@eval.command("results")
@click.argument("job_id")
@format_option()
@click.pass_context
@handle_api_errors
def eval_results(ctx: click.Context, job_id: str, output_format: str) -> None:
    """Retrieve and display evaluation results.

    \b
    Examples:
      evalhub eval results eval-123
      evalhub eval results eval-123 --format json > results.json
      evalhub eval results eval-123 --format csv
    """
    client = get_client(ctx)
    job = client.jobs.get(job_id)

    if job.effective_state != JobStatus.COMPLETED:
        click.echo(
            f"Warning: job {job_id} is in state '{job.effective_state.value}', "
            "results may be incomplete.",
            err=True,
        )

    if not job.results or not job.results.benchmarks:
        click.echo("No results available.")
        return

    if output_format in ("json", "yaml"):
        data = [b.model_dump(mode="json") for b in job.results.benchmarks]
        output(data, output_format=output_format)
        return

    # Table/CSV: flatten metrics into rows
    rows: list[dict[str, Any]] = []
    for b in job.results.benchmarks:
        for metric_name, metric_value in b.metrics.items():
            rows.append(
                {
                    "benchmark": b.id,
                    "provider": b.provider_id,
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

    if not rows:
        click.echo("No metric results available.")
        return

    output(
        rows,
        output_format=output_format,
        columns=["benchmark", "provider", "metric", "value"],
    )

    if job.results.mlflow_experiment_url:
        click.echo(f"\nMLflow experiment: {job.results.mlflow_experiment_url}")


@eval.command("cancel")
@click.argument("job_id")
@click.option(
    "--hard-delete",
    is_flag=True,
    default=False,
    help="Permanently delete the job instead of cancelling.",
)
@click.confirmation_option(prompt="Are you sure you want to cancel this job?")
@click.pass_context
@handle_api_errors
def eval_cancel(ctx: click.Context, job_id: str, hard_delete: bool) -> None:
    """Cancel a running or queued evaluation job.

    \b
    Examples:
      evalhub eval cancel eval-123
      evalhub eval cancel eval-123 --hard-delete
    """
    client = get_client(ctx)
    client.jobs.cancel(job_id, hard_delete=hard_delete)
    action = "deleted" if hard_delete else "cancelled"
    click.echo(f"Job {job_id} {action}.")


@main.group()
def collections() -> None:
    """Browse and manage benchmark collections.

    \b
    Collections group related benchmarks together for convenient
    evaluation. Use 'collections list' to browse, 'collections describe'
    for details, 'collections run' to evaluate a model against a
    collection, or 'collections create/delete' to manage them.

    \b
    Examples:
      evalhub collections list
      evalhub collections describe rag-safety
      evalhub collections run rag-safety --model-url http://vllm:8000/v1 --model-name llama3
    """


@collections.command("list")
@click.option("--tag", "tag_filter", default=None, help="Filter by tag (client-side).")
@format_option()
@click.pass_context
@handle_api_errors
def collections_list(
    ctx: click.Context, tag_filter: str | None, output_format: str
) -> None:
    """List all available benchmark collections.

    \b
    Examples:
      evalhub collections list
      evalhub collections list --tag safety
      evalhub collections list --format json
    """
    client = get_client(ctx)
    items = client.collections.list()
    if tag_filter:
        items = [c for c in items if tag_filter in c.tags]
    rows = [
        {
            "id": c.resource.id,
            "name": c.name,
            "description": c.description,
            "tags": ", ".join(c.tags),
            "benchmarks": len(c.benchmarks),
        }
        for c in items
    ]
    output(
        rows,
        output_format=output_format,
        columns=["id", "name", "description", "tags", "benchmarks"],
    )


@collections.command("describe")
@click.argument("collection_id")
@format_option()
@click.pass_context
@handle_api_errors
def collections_describe(
    ctx: click.Context, collection_id: str, output_format: str
) -> None:
    """Show detailed information about a collection.

    \b
    Examples:
      evalhub collections describe rag-safety
      evalhub collections describe rag-safety --format json
    """
    client = get_client(ctx)
    collection = client.collections.get(collection_id)

    if output_format in ("json", "yaml"):
        output([collection.model_dump(mode="json")], output_format=output_format)
        return

    click.echo(f"Collection: {collection.name}")
    click.echo(f"ID:          {collection.resource.id}")
    click.echo(f"Description: {collection.description}")
    click.echo(f"Category:    {collection.category}")
    if collection.tags:
        click.echo(f"Tags:        {', '.join(collection.tags)}")
    if collection.pass_criteria:
        click.echo(f"Pass threshold: {collection.pass_criteria.threshold}")
    click.echo(f"\nBenchmarks ({len(collection.benchmarks)}):")
    if collection.benchmarks:
        rows = [
            {
                "id": b.id,
                "provider_id": b.provider_id,
                "weight": b.weight,
            }
            for b in collection.benchmarks
        ]
        output(
            rows, output_format=output_format, columns=["id", "provider_id", "weight"]
        )
    else:
        click.echo("  (none)")


@collections.command("create")
@click.option(
    "--file",
    "spec_file",
    type=click.Path(exists=True),
    required=True,
    help="YAML or JSON file describing the collection.",
)
@format_option()
@click.pass_context
@handle_api_errors
def collections_create(ctx: click.Context, spec_file: str, output_format: str) -> None:
    """Create a new benchmark collection from a spec file.

    \b
    Examples:
      evalhub collections create --file bias-fairness-collection.yaml
      evalhub collections create --file collection.json --format json
    """
    data = _load_config_file(spec_file)
    request = CollectionCreateRequest(**data)
    client = get_client(ctx)
    collection = client.collections.create(request.model_dump(mode="json"))
    click.echo(f"Collection created: {collection.resource.id}")
    if output_format in ("json", "yaml"):
        output([collection.model_dump(mode="json")], output_format=output_format)


@collections.command("delete")
@click.argument("collection_id")
@click.confirmation_option(prompt="Are you sure you want to delete this collection?")
@click.pass_context
@handle_api_errors
def collections_delete(ctx: click.Context, collection_id: str) -> None:
    """Delete a benchmark collection.

    \b
    Examples:
      evalhub collections delete rag-safety
    """
    client = get_client(ctx)
    client.collections.delete(collection_id)
    click.echo(f"Collection {collection_id} deleted.")


@collections.command("run")
@click.argument("collection_id")
@click.option("--model-url", required=True, help="Model endpoint URL.")
@click.option("--model-name", required=True, help="Model name or identifier.")
@click.option("--name", default=None, help="Job name (defaults to collection name).")
@click.option(
    "--queue",
    default=None,
    help="Kueue LocalQueue name for workload scheduling.",
)
@click.option(
    "--wait", "wait_for", is_flag=True, default=False, help="Block until job completes."
)
@click.option(
    "--timeout", type=float, default=None, help="Timeout in seconds when using --wait."
)
@click.option(
    "--poll-interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Poll interval in seconds when using --wait.",
)
@format_option()
@click.pass_context
@handle_api_errors
def collections_run(
    ctx: click.Context,
    collection_id: str,
    model_url: str,
    model_name: str,
    name: str | None,
    queue: str | None,
    wait_for: bool,
    timeout: float | None,
    poll_interval: float,
    output_format: str,
) -> None:
    """Run an evaluation collection against a model.

    Fetches the collection, expands its benchmarks into a job submission,
    and submits it to eval-hub.

    \b
    Examples:
      evalhub collections run rag-safety --model-url http://vllm:8000/v1 --model-name llama3
      evalhub collections run rag-safety --model-url http://vllm:8000/v1 --model-name llama3 --wait
      evalhub collections run rag-safety --model-url http://vllm:8000/v1 --model-name llama3 \\
          --queue my-local-queue
    """
    client = get_client(ctx)
    collection = client.collections.get(collection_id)

    job_name = name or f"{collection.name} ({collection_id})"
    benchmarks = [
        BenchmarkConfig(
            id=b.id,
            provider_id=b.provider_id,
            parameters=b.parameters,
        )
        for b in collection.benchmarks
    ]
    if not benchmarks:
        raise click.ClickException(
            f"Collection '{collection_id}' has no benchmarks to run."
        )

    queue_config: QueueConfig | None = QueueConfig(name=queue) if queue else None
    request = JobSubmissionRequest(
        name=job_name,
        model=ModelConfig(url=model_url, name=model_name),
        benchmarks=benchmarks,
        queue=queue_config,
    )
    job = client.jobs.submit(request)
    structured = output_format in ("json", "yaml")
    click.echo(f"Job submitted: {job.id}", err=structured)

    if wait_for:
        click.echo(f"Waiting for job {job.id} to complete...", err=structured)
        job = client.jobs.wait_for_completion(
            job.id, timeout=timeout, poll_interval=poll_interval
        )
        click.echo(
            f"Job {job.id} finished with state: {job.effective_state.value}",
            err=structured,
        )
        if job.effective_state == JobStatus.FAILED:
            ctx.exit(1)

    if structured:
        output([job.model_dump(mode="json")], output_format=output_format)


@main.group()
def providers() -> None:
    """Manage evaluation providers.

    \b
    Providers are evaluation frameworks (e.g. lm-evaluation-harness,
    ragas, garak) registered with EvalHub. Each provider exposes a
    set of benchmarks that can be used in evaluation jobs.

    \b
    Examples:
      evalhub providers list
      evalhub providers describe lm_evaluation_harness
      evalhub providers create --file my-provider.yaml
      evalhub providers delete my-provider
    """


@providers.command("list")
@format_option()
@click.pass_context
@handle_api_errors
def providers_list(ctx: click.Context, output_format: str) -> None:
    """List all registered evaluation providers.

    \b
    Shows provider ID, name, description, and number of benchmarks.

    \b
    Examples:
      evalhub providers list
      evalhub providers list --format json
      evalhub providers list --format csv
    """
    client = get_client(ctx)
    items = client.providers.list()
    if output_format in ("json", "yaml"):
        output(
            [p.model_dump(mode="json") for p in items],
            output_format=output_format,
        )
        return
    rows = [
        {
            "id": p.resource.id,
            "name": p.name,
            "description": p.description,
            "benchmarks": len(p.benchmarks),
        }
        for p in items
    ]
    output(
        rows,
        output_format=output_format,
        columns=["id", "name", "description", "benchmarks"],
    )


@providers.command("describe")
@click.argument("provider_id")
@format_option()
@click.pass_context
@handle_api_errors
def providers_describe(
    ctx: click.Context, provider_id: str, output_format: str
) -> None:
    """Show detailed information about a provider.

    \b
    Displays provider metadata and its available benchmarks with
    categories and supported metrics.

    \b
    Examples:
      evalhub providers describe lm_evaluation_harness
      evalhub providers describe ragas --format json
      evalhub providers describe garak --format yaml
    """
    client = get_client(ctx)
    provider = client.providers.get(provider_id)

    if output_format in ("json", "yaml"):
        data = provider.model_dump(mode="json")
        output([data], output_format=output_format)
        return

    click.echo(f"Provider: {provider.name}")
    click.echo(f"ID:       {provider.resource.id}")
    click.echo(f"Description: {provider.description}")
    click.echo(f"\nBenchmarks ({len(provider.benchmarks)}):")
    if provider.benchmarks:
        rows = [
            {
                "id": b.id,
                "name": b.name,
                "category": b.category,
                "metrics": ", ".join(b.metrics) if b.metrics else "",
            }
            for b in provider.benchmarks
        ]
        output(
            rows,
            output_format=output_format,
            columns=["id", "name", "category", "metrics"],
        )
    else:
        click.echo("  (none)")


@providers.command("create")
@click.option(
    "--file",
    "spec_file",
    type=click.Path(exists=True),
    required=True,
    help="YAML or JSON file describing the provider.",
)
@format_option()
@click.pass_context
@handle_api_errors
def providers_create(ctx: click.Context, spec_file: str, output_format: str) -> None:
    """Create a new evaluation provider from a spec file.

    \b
    Examples:
      evalhub providers create --file my-provider.yaml
      evalhub providers create --file provider.json --format json
    """
    data = _load_config_file(spec_file)
    request = ProviderCreateRequest(**data)
    client = get_client(ctx)
    provider = client.providers.create(request.model_dump(mode="json"))
    structured = output_format in ("json", "yaml")
    click.echo(f"Provider created: {provider.resource.id}", err=structured)
    if structured:
        output([provider.model_dump(mode="json")], output_format=output_format)


@providers.command("delete")
@click.argument("provider_id")
@click.confirmation_option(prompt="Are you sure you want to delete this provider?")
@click.pass_context
@handle_api_errors
def providers_delete(ctx: click.Context, provider_id: str) -> None:
    """Delete an evaluation provider.

    \b
    Examples:
      evalhub providers delete my-provider
    """
    client = get_client(ctx)
    client.providers.delete(provider_id)
    click.echo(f"Provider {provider_id} deleted.")


@main.command("health")
@click.pass_context
@handle_api_errors
def health(ctx: click.Context) -> None:
    """Check health of the EvalHub service.

    \b
    Sends a health check request and reports service status with
    response time. Exits with code 1 if the service is unhealthy
    or unreachable.

    \b
    Examples:
      evalhub health
      evalhub --base-url https://evalhub.example.com health
    """
    client = get_client(ctx)
    start = time.monotonic()
    try:
        result = client.health()
        elapsed = (time.monotonic() - start) * 1000
        status = result.get("status", "unknown")
        click.echo(f"EvalHub service: {status} ({elapsed:.0f}ms)")
        if status != "healthy":
            ctx.exit(1)
    except Exception:
        elapsed = (time.monotonic() - start) * 1000
        click.echo(f"EvalHub service: unreachable ({elapsed:.0f}ms)")
        ctx.exit(1)


@main.group()
@click.pass_context
def config(ctx: click.Context) -> None:
    """View and update CLI configuration.

    \b
    Configuration is stored in ~/.config/evalhub/config.yaml and
    supports multiple profiles. Use 'config set' to store values,
    'config get' to read them, 'config list' to see the full
    profile, and 'config use' to switch profiles.

    \b
    File-based keys (e.g. server_config_file) store a path to a
    YAML file. Use 'config get <key> --unfold' to view the file
    contents.

    \b
    Examples:
      evalhub config set base_url http://localhost:8080
      evalhub config set server_config_file myserver-config.yaml
      evalhub config get server_config_file --unfold
      evalhub config list
      evalhub config use prod
    """


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Set a configuration value in the active profile.

    \b
    Known keys: base_url, token, tenant, provider, insecure, timeout,
    mcp_transport, mcp_host, mcp_port, server_config_file.

    \b
    File-based keys (server_config_file) accept a path to a YAML file.
    The file is validated and copied into the CLI config directory.

    \b
    mcp_transport values:
      stdio    — default for 'evalhub mcp run'
      http     — Streamable HTTP, default for 'evalhub mcp start'
      http-sse — legacy HTTP+SSE
    When mcp_transport is not set, each command uses its own default.
    mcp_host default: localhost
    mcp_port default: 3001

    \b
    Examples:
      evalhub config set base_url http://localhost:8080
      evalhub config set token my-api-token
      evalhub config set tenant my-tenant
      evalhub config set server_config_file myserver-config.yaml
      evalhub --profile prod config set base_url https://evalhub.example.com
    """
    if not cfg.is_known_key(key):
        click.echo(
            f"Warning: '{key}' is not a recognised config key. "
            f"Known keys: {', '.join(sorted(cfg.KNOWN_KEYS))}",
            err=True,
        )
    profile = ctx.obj.get("profile")
    data = cfg.load_config()
    profile_name = profile or cfg.get_active_profile(data)
    if cfg.is_file_key(key):
        src = Path(value)
        cfg.validate_config_file(src)
        value = cfg.store_file_key(key, src, profile_name)
    cfg.set_value(data, key, value, profile=profile)
    cfg.save_config(data)
    click.echo(f"Set '{key}' in profile '{profile_name}'")


@config.command("unset")
@click.argument("key")
@click.pass_context
def config_unset(ctx: click.Context, key: str) -> None:
    """Remove a configuration key from the active profile.

    \b
    For file-based keys (e.g. server_config_file), also deletes the
    stored config file.

    \b
    Examples:
      evalhub config unset mcp_port
      evalhub config unset server_config_file
      evalhub --profile prod config unset insecure
    """
    profile = ctx.obj.get("profile")
    data = cfg.load_config()
    profile_name = profile or cfg.get_active_profile(data)
    removed = cfg.unset_value(data, key, profile=profile)
    if removed:
        cfg.save_config(data)
        if cfg.is_file_key(key):
            cfg.remove_file_key(key, profile_name)
        click.echo(f"Unset '{key}' from profile '{profile_name}'")
    else:
        raise click.ClickException(f"Key '{key}' not found in profile '{profile_name}'")


@config.command("get")
@click.argument("key")
@click.option(
    "--unmask", is_flag=True, default=False, help="Show the raw value without masking."
)
@click.option(
    "--unfold",
    is_flag=True,
    default=False,
    help="Print the contents of a file-based config key.",
)
@click.pass_context
def config_get(ctx: click.Context, key: str, unmask: bool, unfold: bool) -> None:
    """Get a configuration value from the active profile.

    \b
    Sensitive values (e.g. token) are masked by default.
    Use --unmask to reveal the full value.
    Use --unfold with file-based keys (e.g. server_config_file) to
    print the referenced file's contents.

    \b
    Examples:
      evalhub config get base_url
      evalhub config get token --unmask
      evalhub config get server_config_file
      evalhub config get server_config_file --unfold
    """
    if unmask and unfold:
        raise click.ClickException("--unmask and --unfold are mutually exclusive.")
    profile = ctx.obj.get("profile")
    data = cfg.load_config()
    value = cfg.get_value(data, key, profile=profile)
    if value is None:
        profile_name = profile or cfg.get_active_profile(data)
        raise click.ClickException(f"Key '{key}' not found in profile '{profile_name}'")
    if unfold:
        if not cfg.is_file_key(key):
            raise click.ClickException(
                f"--unfold is only valid for file-based config keys "
                f"({', '.join(sorted(cfg.FILE_KEYS))})"
            )
        p = Path(str(value))
        if not p.is_file():
            raise click.ClickException(f"File not found: {value}")
        click.echo(p.read_text(), nl=False)
    elif key in cfg.SENSITIVE_KEYS and not unmask:
        click.echo(cfg.mask_value(str(value)))
    else:
        click.echo(value)


@config.command("list")
@click.pass_context
def config_list(ctx: click.Context) -> None:
    """List all configuration values in the active profile.

    \b
    Shows all key-value pairs and flags any missing required keys.

    \b
    Examples:
      evalhub config list
      evalhub --profile prod config list
    """
    profile = ctx.obj.get("profile")
    data = cfg.load_config()
    profile_name = profile or cfg.get_active_profile(data)
    prof = cfg.get_profile(data, profile=profile)
    click.echo(f"Profile: {profile_name}")
    if not prof:
        click.echo("  (no configuration values)")
    else:
        for k, v in prof.items():
            display = cfg.mask_value(str(v)) if k in cfg.SENSITIVE_KEYS else v
            click.echo(f"  {k}: {display}")
    missing = cfg.missing_required_keys(data, profile=profile)
    if missing:
        click.echo(f"\n  Missing required keys: {', '.join(missing)}")


@config.command("use")
@click.argument("profile")
def config_use(profile: str) -> None:
    """Switch the active configuration profile.

    \b
    Sets the given profile as the default for all subsequent commands.
    The profile must already exist (create it by setting a value with
    --profile).

    \b
    Examples:
      evalhub config use prod
      evalhub config use staging
    """
    data = cfg.load_config()
    profiles = data.get("profiles", {})
    if profile not in profiles:
        click.echo(
            f"Profile '{profile}' does not exist. Available profiles: "
            f"{', '.join(profiles) or '(none)'}",
            err=True,
        )
        raise SystemExit(1)
    cfg.set_active_profile(data, profile)
    cfg.save_config(data)
    click.echo(f"Active profile set to '{profile}'")
