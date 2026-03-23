"""Unit tests for EvalHub CLI providers commands."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from evalhub.cli.main import main
from evalhub.models.api import Benchmark, PassCriteria, PrimaryScore, Provider, Resource


def _make_provider(
    id: str = "lm_eval",
    name: str = "LM Evaluation Harness",
    description: str = "Language model evaluation framework",
    benchmarks: list | None = None,
) -> Provider:
    return Provider(
        resource=Resource(id=id),
        name=name,
        description=description,
        benchmarks=benchmarks or [],
    )


def _make_benchmark(
    id: str = "mmlu",
    name: str = "MMLU",
    description: str = "Massive Multitask Language Understanding",
    category: str = "knowledge",
    metrics: list[str] | None = None,
    num_few_shot: int | None = None,
    dataset_size: int | None = None,
    primary_score: PrimaryScore | None = None,
    pass_criteria: PassCriteria | None = None,
) -> Benchmark:
    return Benchmark(
        id=id,
        name=name,
        description=description,
        category=category,
        metrics=metrics or ["accuracy"],
        num_few_shot=num_few_shot,
        dataset_size=dataset_size,
        primary_score=primary_score,
        pass_criteria=pass_criteria,
    )


@pytest.fixture()
def config_file(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "config.yaml"
    os.environ["EVALHUB_CONFIG"] = str(path)
    yield path
    os.environ.pop("EVALHUB_CONFIG", None)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    client.providers = MagicMock()
    client.health = MagicMock()
    return client


class TestProvidersList:
    def test_list_table_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.providers.list.return_value = [
            _make_provider(
                id="lm_eval", name="LM Eval", benchmarks=[_make_benchmark()]
            ),
            _make_provider(id="ragas", name="RAGAS", description="RAG evaluation"),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "list"])
        assert result.exit_code == 0
        assert "lm_eval" in result.output
        assert "LM Eval" in result.output
        assert "ragas" in result.output
        assert "RAGAS" in result.output

    def test_list_json_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.providers.list.return_value = [
            _make_provider(id="lm_eval", name="LM Eval"),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "list", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "lm_eval"

    def test_list_yaml_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.providers.list.return_value = [
            _make_provider(id="garak", name="Garak"),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "list", "--format", "yaml"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert parsed[0]["id"] == "garak"

    def test_list_csv_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.providers.list.return_value = [
            _make_provider(id="lm_eval", name="LM Eval"),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "list", "--format", "csv"])
        assert result.exit_code == 0
        assert "id,name,description,benchmarks" in result.output
        assert "lm_eval" in result.output

    def test_list_empty(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.providers.list.return_value = []
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "list"])
        assert result.exit_code == 0
        assert "no data" in result.output

    def test_list_shows_benchmark_count(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        benchmarks = [_make_benchmark(id=f"b{i}") for i in range(5)]
        mock_client.providers.list.return_value = [
            _make_provider(id="lm_eval", name="LM Eval", benchmarks=benchmarks),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "list"])
        assert result.exit_code == 0
        assert "5" in result.output


class TestProvidersDescribe:
    def test_describe_table_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        provider = _make_provider(
            id="lm_eval",
            name="LM Evaluation Harness",
            benchmarks=[
                _make_benchmark(id="mmlu", name="MMLU", category="knowledge"),
                _make_benchmark(id="hellaswag", name="HellaSwag", category="reasoning"),
            ],
        )
        mock_client.providers.get.return_value = provider
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "describe", "lm_eval"])
        assert result.exit_code == 0
        assert "LM Evaluation Harness" in result.output
        assert "lm_eval" in result.output
        assert "Benchmarks (2)" in result.output
        assert "mmlu" in result.output
        assert "hellaswag" in result.output

    def test_describe_json_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        provider = _make_provider(id="lm_eval", name="LM Eval")
        mock_client.providers.get.return_value = provider
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["providers", "describe", "lm_eval", "--format", "json"]
            )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed[0]["name"] == "LM Eval"

    def test_describe_yaml_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        provider = _make_provider(id="ragas", name="RAGAS")
        mock_client.providers.get.return_value = provider
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["providers", "describe", "ragas", "--format", "yaml"]
            )
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert parsed[0]["name"] == "RAGAS"

    def test_describe_no_benchmarks(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        provider = _make_provider(id="empty", name="Empty Provider", benchmarks=[])
        mock_client.providers.get.return_value = provider
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "describe", "empty"])
        assert result.exit_code == 0
        assert "Benchmarks (0)" in result.output
        assert "(none)" in result.output

    def test_describe_shows_metrics(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        provider = _make_provider(
            id="lm_eval",
            name="LM Eval",
            benchmarks=[
                _make_benchmark(id="mmlu", metrics=["accuracy", "f1"]),
            ],
        )
        mock_client.providers.get.return_value = provider
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "describe", "lm_eval"])
        assert result.exit_code == 0
        assert "accuracy, f1" in result.output


class TestHealth:
    def test_health_service_healthy(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.health.return_value = {"status": "healthy"}
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["health"])
        assert result.exit_code == 0
        assert "healthy" in result.output
        assert "ms" in result.output

    def test_health_service_unhealthy(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.health.return_value = {"status": "unhealthy"}
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["health"])
        assert result.exit_code == 1
        assert "unhealthy" in result.output

    def test_health_service_unreachable(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.health.side_effect = Exception("Connection refused")
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["health"])
        assert result.exit_code == 1
        assert "unreachable" in result.output


class TestProvidersHelp:
    def test_providers_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["providers", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "describe" in result.output

    def test_providers_list_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["providers", "list", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
