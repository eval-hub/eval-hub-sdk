"""Unit tests for EvalHub CLI output formatter."""

from __future__ import annotations

import io
import json
import re

import pytest
import yaml
from evalhub.cli.formatter import (
    FORMATS,
    _format_csv,
    format_option,
    output,
)

pytestmark = pytest.mark.unit

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

SAMPLE_DATA = [
    {"name": "accuracy", "value": 0.95, "provider": "lm_eval"},
    {"name": "f1_score", "value": 0.88, "provider": "ragas"},
    {"name": "toxicity", "value": 0.02, "provider": "garak"},
]


# --- Table format (Rich) ---


class TestFormatTable:
    def test_renders_headers_and_rows(self) -> None:
        buf = io.StringIO()
        output(SAMPLE_DATA, output_format="table", file=buf)
        text = buf.getvalue()
        assert "NAME" in text
        assert "VALUE" in text
        assert "PROVIDER" in text
        assert "accuracy" in text
        assert "f1_score" in text
        assert "toxicity" in text

    def test_custom_columns(self) -> None:
        buf = io.StringIO()
        output(SAMPLE_DATA, output_format="table", columns=["name", "value"], file=buf)
        text = buf.getvalue()
        assert "PROVIDER" not in text
        assert "NAME" in text
        assert "VALUE" in text

    def test_empty_data(self) -> None:
        buf = io.StringIO()
        output([], output_format="table", file=buf)
        text = buf.getvalue()
        assert "no data" in text

    def test_single_row(self) -> None:
        buf = io.StringIO()
        output([{"key": "val"}], output_format="table", file=buf)
        text = buf.getvalue()
        assert "val" in text
        assert "KEY" in text

    def test_missing_key_in_row(self) -> None:
        data = [{"a": 1, "b": 2}, {"a": 3}]
        buf = io.StringIO()
        output(data, output_format="table", file=buf)
        text = buf.getvalue()
        assert "3" in text


# --- JSON format ---


class TestFormatJson:
    def test_valid_json(self) -> None:
        buf = io.StringIO()
        output(SAMPLE_DATA, output_format="json", file=buf)
        buf.seek(0)
        parsed = json.loads(buf.getvalue())
        assert len(parsed) == 3
        assert parsed[0]["name"] == "accuracy"

    def test_empty_data(self) -> None:
        buf = io.StringIO()
        output([], output_format="json", file=buf)
        buf.seek(0)
        parsed = json.loads(buf.getvalue())
        assert parsed == []


# --- YAML format ---


class TestFormatYaml:
    def test_valid_yaml(self) -> None:
        buf = io.StringIO()
        output(SAMPLE_DATA, output_format="yaml", file=buf)
        buf.seek(0)
        parsed = yaml.safe_load(buf.getvalue())
        assert len(parsed) == 3
        assert parsed[0]["name"] == "accuracy"

    def test_empty_data(self) -> None:
        buf = io.StringIO()
        output([], output_format="yaml", file=buf)
        buf.seek(0)
        parsed = yaml.safe_load(buf.getvalue())
        assert parsed == []


# --- CSV format ---


class TestFormatCsv:
    def test_valid_csv(self) -> None:
        result = _format_csv(SAMPLE_DATA)
        lines = result.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "name,value,provider" in lines[0]
        assert "accuracy,0.95,lm_eval" in lines[1]

    def test_custom_columns(self) -> None:
        result = _format_csv(SAMPLE_DATA, columns=["name", "value"])
        lines = result.strip().split("\n")
        assert "provider" not in lines[0]
        assert "name,value" in lines[0]

    def test_empty_data(self) -> None:
        result = _format_csv([])
        assert result == ""

    def test_missing_key_in_row(self) -> None:
        data = [{"a": 1, "b": 2}, {"a": 3}]
        result = _format_csv(data)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert lines[2] == "3,"


# --- TTY / ANSI behaviour ---


class TestTtyBehaviour:
    def test_no_ansi_when_file_provided(self) -> None:
        """Rich auto-detects non-TTY and strips ANSI."""
        buf = io.StringIO()
        output(SAMPLE_DATA, output_format="table", file=buf)
        text = buf.getvalue()
        assert "\x1b[" not in text

    def test_table_contains_data(self) -> None:
        buf = io.StringIO()
        output(SAMPLE_DATA, output_format="table", file=buf)
        text = buf.getvalue()
        assert "accuracy" in text


# --- format_option decorator ---


class TestFormatOption:
    def test_returns_callable(self) -> None:
        opt = format_option()
        assert callable(opt)

    def test_default_is_table(self) -> None:
        import click
        from click.testing import CliRunner

        @click.command()
        @format_option()
        def cmd(output_format: str) -> None:
            click.echo(output_format)

        runner = CliRunner()
        result = runner.invoke(cmd)
        assert result.output.strip() == "table"

    def test_accepts_json(self) -> None:
        import click
        from click.testing import CliRunner

        @click.command()
        @format_option()
        def cmd(output_format: str) -> None:
            click.echo(output_format)

        runner = CliRunner()
        result = runner.invoke(cmd, ["--format", "json"])
        assert result.output.strip() == "json"


# --- FORMATS constant ---


class TestFormatsConstant:
    def test_contains_all_formats(self) -> None:
        assert "table" in FORMATS
        assert "json" in FORMATS
        assert "yaml" in FORMATS
        assert "csv" in FORMATS
