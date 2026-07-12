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
from evalhub.models.api import (
    AgentMetadata,
    Benchmark,
    PassCriteria,
    PrimaryScore,
    Provider,
    Resource,
)

pytestmark = pytest.mark.unit


def _make_provider(
    id: str = "lm_eval",
    name: str = "LM Evaluation Harness",
    description: str = "Language model evaluation framework",
    benchmarks: list | None = None,
    agent: AgentMetadata | None = None,
) -> Provider:
    return Provider(
        resource=Resource(id=id),
        name=name,
        description=description,
        benchmarks=benchmarks or [],
        agent=agent,
    )


def _make_benchmark(
    id: str = "mmlu",
    name: str = "MMLU",
    description: str = "Massive Multitask Language Understanding",
    category: str = "knowledge",
    metrics: list[str] | None = None,
    num_few_shot: int = 0,
    dataset_size: int = 0,
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
        assert parsed[0]["resource"]["id"] == "lm_eval"
        assert "agent" in parsed[0]

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
        assert parsed[0]["resource"]["id"] == "garak"
        assert "agent" in parsed[0]

    def test_list_json_includes_agent_metadata(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        agent = AgentMetadata(
            evaluates=["reasoning", "knowledge"],
            recommended_when=["general LLM evaluation"],
            target_type="llm",
            summary="Broad benchmark suite",
            complements=["garak"],
            hints=["use with mmlu for knowledge tasks"],
            result_interpretation=["higher is better"],
        )
        mock_client.providers.list.return_value = [
            _make_provider(id="lm_eval", name="LM Eval", agent=agent),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["providers", "list", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed[0]["agent"]["target_type"] == "llm"
        assert parsed[0]["agent"]["summary"] == "Broad benchmark suite"
        assert parsed[0]["agent"]["evaluates"] == ["reasoning", "knowledge"]

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


class TestProvidersCreate:
    def test_create_from_yaml(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        spec = {
            "name": "my-provider",
            "title": "MyProvider",
            "description": "A custom provider",
            "tags": ["custom", "byof"],
            "runtime": {"local": {"command": "python main.py"}},
            "benchmarks": [
                {
                    "id": "my-benchmark",
                    "name": "My Benchmark",
                    "description": "A custom benchmark",
                    "category": "general",
                    "metrics": ["accuracy"],
                }
            ],
        }
        spec_file = tmp_path / "provider.yaml"
        spec_file.write_text(yaml.dump(spec))

        created = _make_provider(id="my-provider", name="MyProvider")
        mock_client.providers.create.return_value = created

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["providers", "create", "--file", str(spec_file)]
            )
        assert result.exit_code == 0
        assert "my-provider" in result.output
        mock_client.providers.create.assert_called_once()
        call_data = mock_client.providers.create.call_args[0][0]
        assert call_data["name"] == "my-provider"
        assert call_data["title"] == "MyProvider"
        assert call_data["tags"] == ["custom", "byof"]
        assert call_data["runtime"] == {"local": {"command": "python main.py"}}

    def test_create_from_json(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        spec = {
            "name": "json-provider",
            "title": "JSON Provider",
            "benchmarks": [
                {
                    "id": "bench-1",
                    "name": "Bench 1",
                    "description": "A benchmark",
                    "category": "general",
                }
            ],
        }
        spec_file = tmp_path / "provider.json"
        spec_file.write_text(json.dumps(spec))

        created = _make_provider(id="json-provider", name="JSON Provider")
        mock_client.providers.create.return_value = created

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["providers", "create", "--file", str(spec_file)]
            )
        assert result.exit_code == 0
        assert "json-provider" in result.output

    def test_create_json_output(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        spec = {"name": "my-provider", "title": "MyProvider", "benchmarks": []}
        spec_file = tmp_path / "provider.yaml"
        spec_file.write_text(yaml.dump(spec))

        created = _make_provider(id="my-provider", name="MyProvider")
        mock_client.providers.create.return_value = created

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                ["providers", "create", "--file", str(spec_file), "--format", "json"],
            )
        assert result.exit_code == 0
        assert "my-provider" in result.output
        json_start = result.output.index("[")
        parsed = json.loads(result.output[json_start:])
        assert parsed[0]["name"] == "MyProvider"

    def test_create_missing_file(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                ["providers", "create", "--file", "/nonexistent/path.yaml"],
            )
        assert result.exit_code != 0
        mock_client.providers.create.assert_not_called()

    def test_create_invalid_spec(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        spec: dict = {"description": "Missing required fields", "benchmarks": []}
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(yaml.dump(spec))

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["providers", "create", "--file", str(spec_file)]
            )
        assert result.exit_code != 0
        mock_client.providers.create.assert_not_called()


class TestProvidersDelete:
    def test_delete_confirmed(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.providers.delete.return_value = None
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["providers", "delete", "my-provider"], input="y\n"
            )
        assert result.exit_code == 0
        assert "deleted" in result.output
        mock_client.providers.delete.assert_called_once_with("my-provider")

    def test_delete_aborted(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            runner.invoke(main, ["providers", "delete", "my-provider"], input="n\n")
        mock_client.providers.delete.assert_not_called()

    def test_delete_yes_flag(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.providers.delete.return_value = None
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["providers", "delete", "my-provider", "--yes"]
            )
        assert result.exit_code == 0
        mock_client.providers.delete.assert_called_once_with("my-provider")


class TestProvidersHelp:
    def test_providers_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["providers", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "describe" in result.output
        assert "create" in result.output
        assert "delete" in result.output

    def test_providers_list_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["providers", "list", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
