"""Unit tests for EvalHub CLI output formatter."""

from __future__ import annotations

import io
import json

import yaml
from evalhub.cli.formatter import (
    FORMATS,
    _format_csv,
    _format_table,
    _strip_ansi,
    format_option,
    output,
)

SAMPLE_DATA = [
    {"name": "accuracy", "value": 0.95, "provider": "lm_eval"},
    {"name": "f1_score", "value": 0.88, "provider": "ragas"},
    {"name": "toxicity", "value": 0.02, "provider": "garak"},
]


# --- ANSI stripping ---


class TestStripAnsi:
    def test_strips_colour_codes(self):
        assert _strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_strips_bold(self):
        assert _strip_ansi("\x1b[1mbold\x1b[0m") == "bold"

    def test_noop_on_plain_text(self):
        assert _strip_ansi("hello world") == "hello world"

    def test_strips_multiple_codes(self):
        text = "\x1b[32m\x1b[1mgreen bold\x1b[0m normal"
        assert _strip_ansi(text) == "green bold normal"


# --- Table format ---


class TestFormatTable:
    def test_renders_headers_and_rows(self):
        result = _format_table(SAMPLE_DATA)
        lines = result.split("\n")
        assert len(lines) == 5  # header + separator + 3 rows
        assert "NAME" in lines[0]
        assert "VALUE" in lines[0]
        assert "PROVIDER" in lines[0]
        assert "---" in lines[1]

    def test_column_alignment(self):
        result = _format_table(SAMPLE_DATA)
        lines = result.split("\n")
        # Header and separator should have consistent column positions
        header = lines[0]
        separator = lines[1]
        # Each column separator position in the header should align with dashes
        assert len(header) == len(separator)

    def test_custom_columns(self):
        result = _format_table(SAMPLE_DATA, columns=["name", "value"])
        assert "PROVIDER" not in result
        assert "NAME" in result
        assert "VALUE" in result

    def test_empty_data(self):
        result = _format_table([])
        assert result == "(no data)"

    def test_single_row(self):
        result = _format_table([{"key": "val"}])
        lines = result.split("\n")
        assert len(lines) == 3  # header + separator + 1 row
        assert "val" in lines[2]

    def test_missing_key_in_row(self):
        data = [{"a": 1, "b": 2}, {"a": 3}]
        result = _format_table(data)
        assert "3" in result


# --- JSON format ---


class TestFormatJson:
    def test_valid_json(self):
        buf = io.StringIO()
        buf.isatty = lambda: False
        output(SAMPLE_DATA, output_format="json", file=buf)
        buf.seek(0)
        parsed = json.loads(buf.getvalue())
        assert len(parsed) == 3
        assert parsed[0]["name"] == "accuracy"

    def test_empty_data(self):
        buf = io.StringIO()
        buf.isatty = lambda: False
        output([], output_format="json", file=buf)
        buf.seek(0)
        parsed = json.loads(buf.getvalue())
        assert parsed == []


# --- YAML format ---


class TestFormatYaml:
    def test_valid_yaml(self):
        buf = io.StringIO()
        buf.isatty = lambda: False
        output(SAMPLE_DATA, output_format="yaml", file=buf)
        buf.seek(0)
        parsed = yaml.safe_load(buf.getvalue())
        assert len(parsed) == 3
        assert parsed[0]["name"] == "accuracy"

    def test_empty_data(self):
        buf = io.StringIO()
        buf.isatty = lambda: False
        output([], output_format="yaml", file=buf)
        buf.seek(0)
        # yaml.safe_load returns None for empty list serialised
        # but we pass [] so it should be a list
        parsed = yaml.safe_load(buf.getvalue())
        assert parsed == []


# --- CSV format ---


class TestFormatCsv:
    def test_valid_csv(self):
        result = _format_csv(SAMPLE_DATA)
        lines = result.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "name,value,provider" in lines[0]
        assert "accuracy,0.95,lm_eval" in lines[1]

    def test_custom_columns(self):
        result = _format_csv(SAMPLE_DATA, columns=["name", "value"])
        lines = result.strip().split("\n")
        assert "provider" not in lines[0]
        assert "name,value" in lines[0]

    def test_empty_data(self):
        result = _format_csv([])
        assert result == ""

    def test_missing_key_in_row(self):
        data = [{"a": 1, "b": 2}, {"a": 3}]
        result = _format_csv(data)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        # Second row should have empty value for b
        assert lines[2] == "3,"


# --- TTY / ANSI stripping integration ---


class TestTtyBehaviour:
    def test_ansi_stripped_when_not_tty(self):
        """When output is not a TTY, ANSI codes should be stripped."""
        buf = io.StringIO()
        buf.isatty = lambda: False
        # Table format doesn't add ANSI by itself, but verify the path works
        output(SAMPLE_DATA, output_format="table", file=buf)
        buf.seek(0)
        text = buf.getvalue()
        assert "\x1b[" not in text

    def test_output_works_with_tty(self):
        """When output is a TTY, content should pass through."""
        buf = io.StringIO()
        buf.isatty = lambda: True
        output(SAMPLE_DATA, output_format="table", file=buf)
        buf.seek(0)
        text = buf.getvalue()
        assert "accuracy" in text


# --- format_option decorator ---


class TestFormatOption:
    def test_returns_callable(self):
        opt = format_option()
        assert callable(opt)

    def test_default_is_table(self):
        import click
        from click.testing import CliRunner

        @click.command()
        @format_option()
        def cmd(output_format):
            click.echo(output_format)

        runner = CliRunner()
        result = runner.invoke(cmd)
        assert result.output.strip() == "table"

    def test_accepts_json(self):
        import click
        from click.testing import CliRunner

        @click.command()
        @format_option()
        def cmd(output_format):
            click.echo(output_format)

        runner = CliRunner()
        result = runner.invoke(cmd, ["--format", "json"])
        assert result.output.strip() == "json"


# --- FORMATS constant ---


class TestFormatsConstant:
    def test_contains_all_formats(self):
        assert "table" in FORMATS
        assert "json" in FORMATS
        assert "yaml" in FORMATS
        assert "csv" in FORMATS
