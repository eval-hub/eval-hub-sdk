"""Shared output formatter for CLI commands.

Supports table, json, yaml, and csv output formats.
Uses Rich for table rendering.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Sequence
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

FORMATS = ("table", "json", "yaml", "csv")


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
    """Format and print data."""
    if output_format == "table":
        _print_table(data, columns, file)
        return

    out = file or click.get_text_stream("stdout")

    if output_format == "json":
        text = json.dumps(list(data), indent=2, default=str)
    elif output_format == "yaml":
        text = yaml.safe_dump(list(data), default_flow_style=False, sort_keys=False)
    elif output_format == "csv":
        text = _format_csv(data, columns)
    else:
        text = ""

    click.echo(text.rstrip(), file=out)


def _print_table(
    data: Sequence[dict[str, Any]],
    columns: Sequence[str] | None = None,
    file: Any | None = None,
) -> None:
    """Render data as a Rich table."""
    console = Console(file=file) if file else Console()

    if not data:
        console.print("(no data)")
        return

    cols = list(columns) if columns else list(data[0].keys())

    table = Table(show_lines=False, padding=(0, 1))
    for c in cols:
        table.add_column(c.upper(), no_wrap=True)

    for row in data:
        table.add_row(*[str(row.get(c, "")) for c in cols])

    console.print(table)


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
