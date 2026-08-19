"""Unit tests for EvalHub CLI collections commands."""

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
from evalhub.client.job_logs import JobLogUpdate
from evalhub.models.api import (
    BenchmarkReference,
    Collection,
    JobStatus,
    PassCriteria,
    Resource,
)

pytestmark = pytest.mark.unit


def _make_benchmark_ref(
    id: str = "mmlu",
    provider_id: str = "lm_evaluation_harness",
    weight: float = 1.0,
) -> BenchmarkReference:
    return BenchmarkReference(
        id=id,
        provider_id=provider_id,
        weight=weight,
    )


def _make_collection(
    id: str = "rag-safety",
    name: str = "RAG Safety",
    description: str = "Safety evaluation for RAG pipelines",
    category: str = "leaderboard",
    tags: list[str] | None = None,
    benchmarks: list[BenchmarkReference] | None = None,
    pass_criteria: PassCriteria | None = None,
) -> Collection:
    return Collection(
        resource=Resource(id=id),
        name=name,
        description=description,
        category=category,
        tags=tags or [],
        benchmarks=benchmarks or [],
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
    client.collections = MagicMock()
    client.jobs = MagicMock()
    return client


# ---------------------------------------------------------------------------
# collections list
# ---------------------------------------------------------------------------


class TestCollectionsList:
    def test_list_table_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.collections.list.return_value = [
            _make_collection(
                id="col-a",
                name="RAG Safety",
                benchmarks=[_make_benchmark_ref()],
            ),
            _make_collection(id="col-b", name="Finance"),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "list"])
        assert result.exit_code == 0
        assert "RAG Safety" in result.output
        assert "Finance" in result.output

    def test_list_json_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.collections.list.return_value = [
            _make_collection(id="rag-safety", name="RAG Safety"),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "list", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "rag-safety"

    def test_list_yaml_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.collections.list.return_value = [
            _make_collection(id="rag-safety", name="RAG Safety"),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "list", "--format", "yaml"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert parsed[0]["id"] == "rag-safety"

    def test_list_csv_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.collections.list.return_value = [
            _make_collection(id="rag-safety", name="RAG Safety"),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "list", "--format", "csv"])
        assert result.exit_code == 0
        assert "id,name" in result.output
        assert "rag-safety" in result.output

    def test_list_empty(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.collections.list.return_value = []
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "list"])
        assert result.exit_code == 0
        assert "no data" in result.output

    def test_list_shows_benchmark_count(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        benchmarks = [_make_benchmark_ref(id=f"b{i}") for i in range(3)]
        mock_client.collections.list.return_value = [
            _make_collection(id="rag-safety", benchmarks=benchmarks),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "list"])
        assert result.exit_code == 0
        assert "3" in result.output

    def test_list_tag_filter(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.collections.list.return_value = [
            _make_collection(id="safe", name="Safety Col", tags=["safety"]),
            _make_collection(id="fin", name="Finance Col", tags=["finance"]),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "list", "--tag", "safety"])
        assert result.exit_code == 0
        assert "safe" in result.output
        assert "Finance Col" not in result.output

    def test_list_tag_filter_no_match(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.collections.list.return_value = [
            _make_collection(id="rag-safety", tags=["safety"]),
        ]
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "list", "--tag", "nonexistent"]
            )
        assert result.exit_code == 0
        assert "no data" in result.output


# ---------------------------------------------------------------------------
# collections describe
# ---------------------------------------------------------------------------


class TestCollectionsDescribe:
    def test_describe_table_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(
            id="rag-safety",
            name="RAG Safety",
            description="Safety evaluation",
            benchmarks=[
                _make_benchmark_ref("mmlu", "lm_evaluation_harness"),
                _make_benchmark_ref("toxicity", "garak"),
            ],
        )
        mock_client.collections.get.return_value = collection
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "describe", "rag-safety"])
        assert result.exit_code == 0
        assert "RAG Safety" in result.output
        assert "rag-safety" in result.output
        assert "Category:    leaderboard" in result.output
        assert "Benchmarks (2)" in result.output
        assert "mmlu" in result.output
        assert "toxicity" in result.output

    def test_describe_json_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(id="rag-safety", name="RAG Safety")
        mock_client.collections.get.return_value = collection
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "describe", "rag-safety", "--format", "json"]
            )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed[0]["name"] == "RAG Safety"

    def test_describe_yaml_format(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(id="rag-safety", name="RAG Safety")
        mock_client.collections.get.return_value = collection
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "describe", "rag-safety", "--format", "yaml"]
            )
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert parsed[0]["name"] == "RAG Safety"

    def test_describe_no_benchmarks(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(id="empty", name="Empty", benchmarks=[])
        mock_client.collections.get.return_value = collection
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "describe", "empty"])
        assert result.exit_code == 0
        assert "Benchmarks (0)" in result.output
        assert "(none)" in result.output

    def test_describe_shows_tags(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(id="rag-safety", tags=["safety", "rag"])
        mock_client.collections.get.return_value = collection
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "describe", "rag-safety"])
        assert result.exit_code == 0
        assert "safety" in result.output
        assert "rag" in result.output

    def test_describe_shows_pass_criteria(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(
            id="rag-safety",
            pass_criteria=PassCriteria(threshold=0.8),
        )
        mock_client.collections.get.return_value = collection
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(main, ["collections", "describe", "rag-safety"])
        assert result.exit_code == 0
        assert "0.8" in result.output


# ---------------------------------------------------------------------------
# collections create
# ---------------------------------------------------------------------------


class TestCollectionsCreate:
    def test_create_from_yaml(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        spec = {
            "name": "New Collection",
            "description": "Created via CLI",
            "category": "leaderboard",
            "benchmarks": [
                {"benchmark_id": "mmlu", "provider_id": "lm_evaluation_harness"}
            ],
        }
        spec_file = tmp_path / "collection.yaml"
        spec_file.write_text(yaml.dump(spec))

        created = _make_collection(id="new-col-123", name="New Collection")
        mock_client.collections.create.return_value = created

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "create", "--file", str(spec_file)]
            )
        assert result.exit_code == 0
        assert "new-col-123" in result.output
        mock_client.collections.create.assert_called_once()

    def test_create_from_json(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        spec = {
            "name": "JSON Collection",
            "category": "leaderboard",
            "benchmarks": [
                {"benchmark_id": "arc_easy", "provider_id": "lm_evaluation_harness"}
            ],
        }
        spec_file = tmp_path / "collection.json"
        spec_file.write_text(json.dumps(spec))

        created = _make_collection(id="json-col-456", name="JSON Collection")
        mock_client.collections.create.return_value = created

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "create", "--file", str(spec_file)]
            )
        assert result.exit_code == 0
        assert "json-col-456" in result.output

    def test_create_json_output(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        spec = {"name": "My Collection", "category": "leaderboard", "benchmarks": []}
        spec_file = tmp_path / "collection.yaml"
        spec_file.write_text(yaml.dump(spec))

        created = _make_collection(id="my-col", name="My Collection")
        mock_client.collections.create.return_value = created

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                ["collections", "create", "--file", str(spec_file), "--format", "json"],
            )
        assert result.exit_code == 0
        assert "my-col" in result.output
        # The JSON array begins after the "Collection created:" echo line
        json_start = result.output.index("[")
        parsed = json.loads(result.output[json_start:])
        assert parsed[0]["name"] == "My Collection"

    def test_create_missing_file(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                ["collections", "create", "--file", "/nonexistent/path.yaml"],
            )
        assert result.exit_code != 0
        mock_client.collections.create.assert_not_called()

    def test_create_sends_category(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        spec = {
            "name": "Leader Collection",
            "description": "A leaderboard collection",
            "category": "leaderboard",
            "benchmarks": [
                {"benchmark_id": "hellaswag_ar", "provider_id": "lm_evaluation_harness"}
            ],
        }
        spec_file = tmp_path / "collection.yaml"
        spec_file.write_text(yaml.dump(spec))

        created = _make_collection(
            id="leader-col", name="Leader Collection", category="leaderboard"
        )
        mock_client.collections.create.return_value = created

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "create", "--file", str(spec_file)]
            )
        assert result.exit_code == 0
        call_data = mock_client.collections.create.call_args[0][0]
        assert call_data["category"] == "leaderboard"

    def test_create_missing_category(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        spec = {
            "name": "No Category",
            "benchmarks": [],
        }
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(yaml.dump(spec))

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "create", "--file", str(spec_file)]
            )
        assert result.exit_code != 0
        mock_client.collections.create.assert_not_called()

    def test_create_invalid_spec(
        self,
        runner: CliRunner,
        config_file: Path,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Missing required 'name' field
        spec: dict = {"benchmarks": []}
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(yaml.dump(spec))

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "create", "--file", str(spec_file)]
            )
        assert result.exit_code != 0
        mock_client.collections.create.assert_not_called()


# ---------------------------------------------------------------------------
# collections delete
# ---------------------------------------------------------------------------


class TestCollectionsDelete:
    def test_delete_confirmed(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.collections.delete.return_value = None
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "delete", "rag-safety"], input="y\n"
            )
        assert result.exit_code == 0
        assert "deleted" in result.output
        mock_client.collections.delete.assert_called_once_with("rag-safety")

    def test_delete_aborted(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            runner.invoke(main, ["collections", "delete", "rag-safety"], input="n\n")
        mock_client.collections.delete.assert_not_called()

    def test_delete_yes_flag(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        mock_client.collections.delete.return_value = None
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main, ["collections", "delete", "rag-safety", "--yes"]
            )
        assert result.exit_code == 0
        mock_client.collections.delete.assert_called_once_with("rag-safety")


# ---------------------------------------------------------------------------
# collections run
# ---------------------------------------------------------------------------


class TestCollectionsRun:
    def test_run_submits_job(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(
            id="rag-safety",
            name="RAG Safety",
            benchmarks=[
                _make_benchmark_ref("mmlu", "lm_evaluation_harness"),
                _make_benchmark_ref("toxicity", "garak"),
            ],
        )
        mock_client.collections.get.return_value = collection

        job = MagicMock()
        job.id = "job-abc"
        job.state.value = "running"
        mock_client.jobs.submit.return_value = job

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "collections",
                    "run",
                    "rag-safety",
                    "--model-url",
                    "http://vllm:8000/v1",
                    "--model-name",
                    "llama3",
                ],
            )
        assert result.exit_code == 0
        assert "job-abc" in result.output
        mock_client.jobs.submit.assert_called_once()

    def test_run_custom_job_name(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(
            id="rag-safety",
            benchmarks=[_make_benchmark_ref()],
        )
        mock_client.collections.get.return_value = collection

        job = MagicMock()
        job.id = "job-xyz"
        job.state.value = "running"
        mock_client.jobs.submit.return_value = job

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "collections",
                    "run",
                    "rag-safety",
                    "--model-url",
                    "http://vllm:8000/v1",
                    "--model-name",
                    "llama3",
                    "--name",
                    "my-custom-run",
                ],
            )
        assert result.exit_code == 0
        submitted = mock_client.jobs.submit.call_args[0][0]
        assert submitted.name == "my-custom-run"

    def test_run_with_model_auth_secret(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(
            id="rag-safety",
            benchmarks=[_make_benchmark_ref()],
        )
        mock_client.collections.get.return_value = collection

        job = MagicMock()
        job.id = "job-xyz"
        job.state.value = "running"
        mock_client.jobs.submit.return_value = job

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "collections",
                    "run",
                    "rag-safety",
                    "--model-url",
                    "http://vllm:8000/v1",
                    "--model-name",
                    "llama3",
                    "--model-auth-secret",
                    "my-model-credentials",
                ],
            )
        assert result.exit_code == 0
        submitted = mock_client.jobs.submit.call_args[0][0]
        assert submitted.model.auth is not None
        assert submitted.model.auth.secret_ref == "my-model-credentials"

    def test_run_empty_collection_errors(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(id="empty", benchmarks=[])
        mock_client.collections.get.return_value = collection

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "collections",
                    "run",
                    "empty",
                    "--model-url",
                    "http://vllm:8000/v1",
                    "--model-name",
                    "llama3",
                ],
            )
        assert result.exit_code != 0
        mock_client.jobs.submit.assert_not_called()

    def test_run_missing_model_url(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                ["collections", "run", "rag-safety", "--model-name", "llama3"],
            )
        assert result.exit_code != 0
        mock_client.jobs.submit.assert_not_called()

    def test_run_json_output(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(
            id="rag-safety",
            benchmarks=[_make_benchmark_ref()],
        )
        mock_client.collections.get.return_value = collection

        job = MagicMock()
        job.id = "job-abc"
        job.state.value = "running"
        job.model_dump.return_value = {"id": "job-abc", "state": "running"}
        mock_client.jobs.submit.return_value = job

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "collections",
                    "run",
                    "rag-safety",
                    "--model-url",
                    "http://vllm:8000/v1",
                    "--model-name",
                    "llama3",
                    "--format",
                    "json",
                ],
            )
        assert result.exit_code == 0
        assert "job-abc" in result.output

    def test_run_with_queue_flag(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(
            id="rag-safety",
            benchmarks=[_make_benchmark_ref()],
        )
        mock_client.collections.get.return_value = collection

        job = MagicMock()
        job.id = "job-abc"
        job.state.value = "running"
        mock_client.jobs.submit.return_value = job

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "collections",
                    "run",
                    "rag-safety",
                    "--model-url",
                    "http://vllm:8000/v1",
                    "--model-name",
                    "llama3",
                    "--queue",
                    "user-queue",
                ],
            )
        assert result.exit_code == 0
        req = mock_client.jobs.submit.call_args[0][0]
        assert req.queue is not None
        assert req.queue.name == "user-queue"
        assert req.queue.kind is None

    def test_run_without_queue_flag(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(
            id="rag-safety",
            benchmarks=[_make_benchmark_ref()],
        )
        mock_client.collections.get.return_value = collection

        job = MagicMock()
        job.id = "job-abc"
        job.state.value = "running"
        mock_client.jobs.submit.return_value = job

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "collections",
                    "run",
                    "rag-safety",
                    "--model-url",
                    "http://vllm:8000/v1",
                    "--model-name",
                    "llama3",
                ],
            )
        assert result.exit_code == 0
        req = mock_client.jobs.submit.call_args[0][0]
        assert req.queue is None

    def test_run_with_watch(
        self, runner: CliRunner, config_file: Path, mock_client: MagicMock
    ) -> None:
        collection = _make_collection(
            id="rag-safety",
            benchmarks=[_make_benchmark_ref()],
        )
        mock_client.collections.get.return_value = collection

        job = MagicMock()
        job.id = "job-abc"
        mock_client.jobs.submit.return_value = job
        completed = MagicMock()
        completed.effective_state = JobStatus.COMPLETED
        mock_client.jobs.watch_logs.return_value = iter(
            [JobLogUpdate(logs="log line\n", job=completed)]
        )

        with patch("evalhub.cli.main.get_client", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "collections",
                    "run",
                    "rag-safety",
                    "--model-url",
                    "http://vllm:8000/v1",
                    "--model-name",
                    "llama3",
                    "--watch",
                ],
            )
        assert result.exit_code == 0
        assert "log line" in result.output
        mock_client.jobs.watch_logs.assert_called_once()


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


class TestCollectionsHelp:
    def test_collections_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["collections", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "describe" in result.output
        assert "create" in result.output
        assert "delete" in result.output
        assert "run" in result.output

    def test_collections_list_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["collections", "list", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--tag" in result.output

    def test_collections_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["collections", "run", "--help"])
        assert result.exit_code == 0
        assert "--model-url" in result.output
        assert "--model-name" in result.output
        assert "--model-auth-secret" in result.output
        assert "--wait" in result.output
        assert "--watch" in result.output
