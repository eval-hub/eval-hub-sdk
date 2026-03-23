"""CLI client helper — instantiates SyncEvalHubClient from profile config."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import click
import httpx

from evalhub.client import ClientError, SyncEvalHubClient

from . import config as cfg

F = TypeVar("F", bound=Callable[..., Any])


def create_client(
    profile: str | None = None,
    base_url: str | None = None,
    token: str | None = None,
) -> SyncEvalHubClient:
    """Build a SyncEvalHubClient from profile config with optional overrides.

    Priority (highest wins):
    1. Explicit CLI flags (base_url, token)
    2. Profile config values
    3. SyncEvalHubClient defaults
    """
    data = cfg.load_config()
    prof = cfg.get_profile(data, profile)

    resolved_url = base_url or prof.get("base_url", "http://localhost:8080")
    resolved_token = token or prof.get("token")
    tenant = prof.get("tenant")
    insecure = str(prof.get("insecure", "false")).lower() in ("true", "1", "yes")
    timeout = float(prof.get("timeout", 30.0))

    return SyncEvalHubClient(
        base_url=resolved_url,
        auth_token=resolved_token,
        tenant=tenant,
        insecure=insecure,
        timeout=timeout,
    )


def get_client(ctx: click.Context) -> SyncEvalHubClient:
    """Retrieve or create the SyncEvalHubClient from the Click context."""
    if "client" not in ctx.obj:
        ctx.obj["client"] = create_client(
            profile=ctx.obj.get("profile"),
            base_url=ctx.obj.get("base_url"),
            token=ctx.obj.get("token"),
        )
    client: SyncEvalHubClient = ctx.obj["client"]
    return client


def handle_api_errors(f: F) -> F:
    """Decorator that catches API/connection errors and displays user-friendly messages."""

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return f(*args, **kwargs)
        except ClientError as exc:
            raise click.ClickException(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            raise click.ClickException(f"Server error ({status}): {detail}") from exc
        except httpx.RequestError as exc:
            raise click.ClickException(f"Connection error: {exc}") from exc

    return wrapper  # type: ignore[return-value]
