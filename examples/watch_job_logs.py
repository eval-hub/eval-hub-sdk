#!/usr/bin/env python3
"""Watch evaluation job logs on a local or remote EvalHub cluster.

Examples:

  # Remote cluster with explicit flags
  uv run python examples/watch_job_logs.py eval-abc123 \\
    --base-url https://evalhub.apps.my-cluster.example.com \\
    --token "$EVALHUB_TOKEN" \\
    --tenant my-team

  # Use an evalhub CLI profile (evalhub config set/use first)
  uv run python examples/watch_job_logs.py eval-abc123 --profile prod

  # Environment variables instead of flags
  export EVALHUB_BASE_URL=https://evalhub.apps.my-cluster.example.com
  export EVALHUB_TOKEN=...
  export EVALHUB_TENANT=my-team
  uv run python examples/watch_job_logs.py eval-abc123

  # OpenShift service-account token file + cluster CA
  uv run python examples/watch_job_logs.py eval-abc123 \\
    --base-url https://evalhub.evalhub.svc:8080 \\
    --token-file /var/run/secrets/kubernetes.io/serviceaccount/token \\
    --ca-bundle /var/run/secrets/kubernetes.io/serviceaccount/ca.crt \\
    --tenant default
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import cast

import httpx
from evalhub import JobLogOptions, SyncEvalHubClient
from evalhub.client.job_logs import TERMINAL_JOB_STATES
from evalhub.models import JobStatus


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream evaluation job logs until the job completes.",
    )
    parser.add_argument("job_id", help="Evaluation job ID to watch")
    parser.add_argument(
        "--profile",
        help="EvalHub CLI profile name (reads ~/.config/evalhub/config.yaml)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EVALHUB_BASE_URL"),
        help="EvalHub base URL (or set EVALHUB_BASE_URL)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("EVALHUB_TOKEN"),
        help="Bearer token (or set EVALHUB_TOKEN)",
    )
    parser.add_argument(
        "--token-file",
        help="Read bearer token from a file (e.g. Kubernetes service account)",
    )
    parser.add_argument(
        "--tenant",
        default=os.environ.get("EVALHUB_TENANT"),
        help="X-Tenant header value (or set EVALHUB_TENANT)",
    )
    parser.add_argument(
        "--ca-bundle",
        help="Path to CA bundle for TLS verification",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS certificate verification",
    )
    parser.add_argument(
        "--benchmark-index",
        type=int,
        help="Watch logs for a single benchmark index only",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=1000,
        help="Max log lines per poll (default: 1000)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between polls (default: 2.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Stop watching after this many seconds",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include Kubernetes log timestamps when supported",
    )
    return parser


def _resolve_token(args: argparse.Namespace) -> str | None:
    if args.token_file:
        return Path(args.token_file).read_text().strip()
    return cast(str | None, args.token)


def _resolve_base_url(args: argparse.Namespace, profile_name: str, prof: dict) -> str:
    base_url = args.base_url or prof.get("base_url")
    if not base_url:
        raise SystemExit(
            f"Missing base_url for profile {profile_name!r}. "
            "Pass --base-url, set EVALHUB_BASE_URL, or run: "
            "evalhub config set base_url <url>"
        )
    return str(base_url)


def _create_client(args: argparse.Namespace) -> SyncEvalHubClient:
    token = _resolve_token(args)

    if args.profile:
        from evalhub.cli import config as cfg

        data = cfg.load_config()
        prof = cfg.get_profile(data, args.profile)
        return SyncEvalHubClient(
            base_url=_resolve_base_url(args, args.profile, prof),
            auth_token=token or prof.get("token"),
            ca_bundle_path=args.ca_bundle,
            insecure=args.insecure or cfg.parse_bool(prof.get("insecure")),
            tenant=args.tenant if args.tenant is not None else prof.get("tenant"),
            timeout=float(prof.get("timeout", 60.0)),
        )

    if not args.base_url:
        raise SystemExit(
            "Missing --base-url (or EVALHUB_BASE_URL). "
            "Use --profile or pass --base-url explicitly."
        )

    return SyncEvalHubClient(
        base_url=args.base_url,
        auth_token=token,
        ca_bundle_path=args.ca_bundle,
        insecure=args.insecure,
        tenant=args.tenant,
        timeout=60.0,
    )


def _format_http_error(exc: httpx.HTTPError) -> str:
    """Format an HTTP error with server response details when available."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return str(exc)

    response = exc.response
    lines = [
        f"HTTP {response.status_code} {response.reason_phrase} for {response.request.url}"
    ]

    body = response.text.strip()
    if not body:
        return "\n".join(lines)

    try:
        payload = response.json()
    except ValueError:
        lines.append(body)
        return "\n".join(lines)

    if isinstance(payload, dict):
        message = payload.get("message")
        message_code = payload.get("message_code")
        trace = payload.get("trace")
        if message_code:
            lines.append(f"message_code: {message_code}")
        if message:
            lines.append(f"message: {message}")
        if trace:
            lines.append(f"trace: {trace}")
        if not (message or message_code):
            lines.append(body)
    else:
        lines.append(body)

    return "\n".join(lines)


def main() -> int:
    args = _build_parser().parse_args()
    options = JobLogOptions(tail_lines=args.tail_lines, timestamps=args.timestamps)

    print(f"Watching logs for job {args.job_id!r}...", file=sys.stderr)
    if args.base_url or args.profile:
        target = args.base_url or f"profile:{args.profile}"
        print(f"  server: {target}", file=sys.stderr)
    if args.tenant:
        print(f"  tenant: {args.tenant}", file=sys.stderr)

    final_state: JobStatus | None = None
    reached_terminal = False
    with _create_client(args) as client:
        try:
            for update in client.jobs.watch_logs(
                args.job_id,
                benchmark_index=args.benchmark_index,
                options=options,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            ):
                if update.logs:
                    print(update.logs, end="", flush=True)
                final_state = update.job.effective_state
                reached_terminal = final_state in TERMINAL_JOB_STATES
        except TimeoutError:
            print("\nTimed out before job reached a terminal state.", file=sys.stderr)
        except httpx.HTTPError as exc:
            print(f"\nRequest failed:\n{_format_http_error(exc)}", file=sys.stderr)
            return 3
        except KeyboardInterrupt:
            print("\nStopped.", file=sys.stderr)
            return 130

    print(file=sys.stderr)
    if not reached_terminal:
        print(
            f"Watch ended before completion (last state: {final_state}).",
            file=sys.stderr,
        )
        return 2
    print(f"Job finished with state: {final_state}", file=sys.stderr)
    if final_state == JobStatus.FAILED:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
