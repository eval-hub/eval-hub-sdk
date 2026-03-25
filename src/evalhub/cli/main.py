"""EvalHub CLI entry point and command groups."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import click
import yaml

import evalhub
from evalhub.models import (
    BenchmarkConfig,
    CollectionCreateRequest,
    JobStatus,
    JobSubmissionRequest,
    ModelConfig,
)

from . import config as cfg
from .client import get_client, handle_api_errors
from .formatter import format_option, output


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
    """EvalHub CLI - manage evaluations, providers, collections, and configuration."""
    if not verbose:
        logging.getLogger("evalhub").setLevel(logging.ERROR)
    ctx.ensure_object(dict)
    ctx.obj["profile"] = profile
    ctx.obj["base_url"] = base_url
    ctx.obj["token"] = token


@main.command()
def version() -> None:
    """Print version and build info."""
    click.echo(f"evalhub {evalhub.__version__}")


@main.group()
def eval() -> None:
    """Submit and manage evaluation jobs."""


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
) -> JobSubmissionRequest:
    """Build a JobSubmissionRequest from CLI flags."""
    parameters: dict[str, Any] = {}
    if metrics:
        parameters["metrics"] = list(metrics)
    if dataset:
        parameters["dataset"] = dataset
    benchmarks = [
        BenchmarkConfig(id=b, provider_id=provider, parameters=parameters)
        for b in benchmark
    ]
    return JobSubmissionRequest(
        name=name,
        description=description,
        model=ModelConfig(url=model_url, name=model_name),
        benchmarks=benchmarks,
    )


@eval.command("run")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="YAML or JSON config file for the evaluation job.",
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
    wait_for: bool,
    timeout: float | None,
    poll_interval: float,
    output_format: str,
) -> None:
    """Submit an evaluation job.

    Use --config to submit from a YAML/JSON file, or specify flags inline.

    \b
    Examples:
      evalhub eval run --config eval.yaml
      evalhub eval run --config eval.yaml --wait
      evalhub eval run --name my-eval --model-url http://vllm:8000/v1 \\
          --model-name llama3 --provider lm_evaluation_harness -b mmlu -b hellaswag
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
        request = _build_request_from_flags(
            name=cast(str, name),
            model_url=cast(str, model_url),
            model_name=cast(str, model_name),
            provider=cast(str, provider),
            benchmark=benchmark,
            description=description,
            metrics=metrics,
            dataset=dataset,
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
            f"Job {job.id} finished with state: {job.state.value}", err=structured
        )
        if job.state == JobStatus.FAILED:
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
    terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    while True:
        job = client.jobs.get(job_id)
        benchmarks_status = ""
        if job.status and job.status.benchmarks:
            done = sum(1 for b in job.status.benchmarks if b.state in terminal)
            benchmarks_status = f" [{done}/{len(job.status.benchmarks)} benchmarks]"
        click.echo(f"\r{job.id}: {job.state.value}{benchmarks_status}", nl=False)
        sys.stdout.flush()
        if job.state in terminal:
            click.echo()
            _print_job_detail(job)
            return
        time.sleep(poll_interval)


def _print_job_detail(job: Any) -> None:
    """Print detailed job information in human-readable format."""
    click.echo(f"Job:     {job.id}")
    click.echo(f"Name:    {job.name}")
    click.echo(f"State:   {job.state.value}")
    if job.description:
        click.echo(f"Desc:    {job.description}")
    click.echo(f"Model:   {job.model.name} ({job.model.url})")
    click.echo(f"Created: {job.resource.created_at}")

    if job.status and job.status.benchmarks:
        click.echo(f"\nBenchmarks ({len(job.status.benchmarks)}):")
        for b in job.status.benchmarks:
            line = f"  {b.id} ({b.provider_id}): {b.state.value}"
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

    if job.state != JobStatus.COMPLETED:
        click.echo(
            f"Warning: job {job_id} is in state '{job.state.value}', "
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
    """Browse and manage benchmark collections."""


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

    request = JobSubmissionRequest(
        name=job_name,
        model=ModelConfig(url=model_url, name=model_name),
        benchmarks=benchmarks,
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
            f"Job {job.id} finished with state: {job.state.value}", err=structured
        )
        if job.state == JobStatus.FAILED:
            ctx.exit(1)

    if structured:
        output([job.model_dump(mode="json")], output_format=output_format)


@main.group()
def providers() -> None:
    """List and inspect evaluation providers."""


@providers.command("list")
@format_option()
@click.pass_context
@handle_api_errors
def providers_list(ctx: click.Context, output_format: str) -> None:
    """List all registered evaluation providers."""
    client = get_client(ctx)
    items = client.providers.list()
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
    """Show detailed information about a provider."""
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


@main.command("health")
@click.pass_context
@handle_api_errors
def health(ctx: click.Context) -> None:
    """Check health of the EvalHub service."""
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
    """View and update CLI configuration."""


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Set a configuration value in the active profile."""
    if not cfg.is_known_key(key):
        click.echo(
            f"Warning: '{key}' is not a recognised config key. "
            f"Known keys: {', '.join(sorted(cfg.KNOWN_KEYS))}",
            err=True,
        )
    profile = ctx.obj.get("profile")
    data = cfg.load_config()
    cfg.set_value(data, key, value, profile=profile)
    cfg.save_config(data)
    profile_name = profile or cfg.get_active_profile(data)
    click.echo(f"Set '{key}' in profile '{profile_name}'")


@config.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx: click.Context, key: str) -> None:
    """Get a configuration value from the active profile."""
    profile = ctx.obj.get("profile")
    data = cfg.load_config()
    value = cfg.get_value(data, key, profile=profile)
    if value is None:
        profile_name = profile or cfg.get_active_profile(data)
        raise click.ClickException(f"Key '{key}' not found in profile '{profile_name}'")
    click.echo(value)


@config.command("list")
@click.pass_context
def config_list(ctx: click.Context) -> None:
    """List all configuration values in the active profile."""
    profile = ctx.obj.get("profile")
    data = cfg.load_config()
    profile_name = profile or cfg.get_active_profile(data)
    prof = cfg.get_profile(data, profile=profile)
    click.echo(f"Profile: {profile_name}")
    if not prof:
        click.echo("  (no configuration values)")
    else:
        for k, v in prof.items():
            click.echo(f"  {k}: {v}")
    missing = cfg.missing_required_keys(data, profile=profile)
    if missing:
        click.echo(f"\n  Missing required keys: {', '.join(missing)}")


@config.command("use")
@click.argument("profile")
def config_use(profile: str) -> None:
    """Switch the active configuration profile."""
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
