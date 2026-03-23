"""EvalHub CLI entry point and command groups."""

import click

import evalhub

from . import config as cfg


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=evalhub.__version__, prog_name="evalhub")
@click.option(
    "--profile",
    default=None,
    envvar="EVALHUB_PROFILE",
    help="Configuration profile to use (overrides active profile).",
)
@click.pass_context
def main(ctx: click.Context, profile: str | None) -> None:
    """EvalHub CLI - manage evaluations, providers, collections, and configuration."""
    ctx.ensure_object(dict)
    ctx.obj["profile"] = profile


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
