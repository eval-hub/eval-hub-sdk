"""EvalHub CLI entry point and command groups."""

import logging
import time

import click

import evalhub

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


@main.group()
def collections() -> None:
    """Browse and manage benchmark collections."""


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
