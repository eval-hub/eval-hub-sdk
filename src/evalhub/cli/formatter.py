"""Shared output formatter for CLI commands.

Supports table, json, yaml, and csv output formats.
Strips ANSI codes when stdout is not a TTY.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import click
import yaml

FORMATS = ("table", "json", "yaml", "csv")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def format_option(default: str = "table") -> Callable:
    """Reusable --format Click option."""
    return click.option(
        "--format",
        "output_format",
        type=click.Choice(FORMATS, case_sensitive=False),
        default=default,
        show_default=True,
        help="Output format.",
    )


def output(
    data: Sequence[dict[str, Any]],
    output_format: str = "table",
    columns: Sequence[str] | None = None,
    file: Any | None = None,
) -> None:
    """Format and print data.

    Args:
        data: List of dicts to render.
        output_format: One of table, json, yaml, csv.
        columns: Column names to include (default: all keys from first row).
        file: Output stream (default: click.get_text_stream("stdout")).
    """
    out = file or click.get_text_stream("stdout")
    is_tty = hasattr(out, "isatty") and out.isatty()

    if output_format == "json":
        text = json.dumps(list(data), indent=2, default=str)
    elif output_format == "yaml":
        text = yaml.safe_dump(list(data), default_flow_style=False, sort_keys=False)
    elif output_format == "csv":
        text = _format_csv(data, columns)
    else:
        text = _format_table(data, columns)

    if not is_tty:
        text = _strip_ansi(text)

    click.echo(text.rstrip(), file=out)


def _format_table(
    data: Sequence[dict[str, Any]],
    columns: Sequence[str] | None = None,
) -> str:
    if not data:
        return "(no data)"

    cols = list(columns) if columns else list(data[0].keys())

    # Calculate column widths
    widths = {c: len(c) for c in cols}
    for row in data:
        for c in cols:
            val = str(row.get(c, ""))
            widths[c] = max(widths[c], len(val))

    # Header
    header = "  ".join(c.upper().ljust(widths[c]) for c in cols)
    separator = "  ".join("-" * widths[c] for c in cols)

    # Rows
    lines = [header, separator]
    for row in data:
        line = "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
        lines.append(line)

    return "\n".join(lines)


def _format_csv(
    data: Sequence[dict[str, Any]],
    columns: Sequence[str] | None = None,
) -> str:
    if not data:
        return ""

    cols = list(columns) if columns else list(data[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        writer.writerow(row)
    return buf.getvalue()
