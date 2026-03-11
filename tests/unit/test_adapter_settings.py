"""Unit tests for AdapterSettings and config.get_job_spec_path()."""

import json
from pathlib import Path

import pytest
from evalhub.adapter.config import get_job_spec_path
from evalhub.adapter.settings import AdapterSettings


class TestAdapterSettingsModeNormalization:
    """Tests for case-insensitive EVALHUB_MODE handling."""

    @pytest.mark.parametrize(
        "env_value, expected",
        [
            ("k8s", "k8s"),
            ("K8s", "k8s"),
            ("local", "local"),
            ("Local", "local"),
            ("LOCAL", "local"),
        ],
    )
    def test_mode_is_normalized_to_lowercase(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_value: str,
        expected: str,
    ) -> None:
        """EVALHUB_MODE should accept any casing and normalize to lowercase."""
        monkeypatch.setenv("EVALHUB_MODE", env_value)
        settings = AdapterSettings.from_env()
        assert settings.mode == expected

    def test_mode_defaults_to_local_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When EVALHUB_MODE is not set, mode defaults to 'local'."""
        monkeypatch.delenv("EVALHUB_MODE", raising=False)
        settings = AdapterSettings.from_env()
        assert settings.mode == "local"

    def test_invalid_mode_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognized mode value should raise a validation error."""
        monkeypatch.setenv("EVALHUB_MODE", "unknown")
        with pytest.raises(Exception, match="Input should be"):
            AdapterSettings.from_env()


class TestResolvedJobSpecPath:
    """Tests for AdapterSettings.resolved_job_spec_path.
    This to verify and align with specified behaviour in docstring.
    """

    def test_k8s_mode_resolves_to_absolute_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "k8s")
        monkeypatch.delenv("EVALHUB_JOB_SPEC_PATH", raising=False)
        settings = AdapterSettings.from_env()
        assert settings.resolved_job_spec_path == Path("/meta/job.json")

    def test_local_mode_resolves_to_relative_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        monkeypatch.delenv("EVALHUB_JOB_SPEC_PATH", raising=False)
        settings = AdapterSettings.from_env()
        assert settings.resolved_job_spec_path == Path("meta/job.json")

    def test_explicit_job_spec_path_overrides_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        custom = tmp_path / "custom.json"
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(custom))
        monkeypatch.setenv("EVALHUB_MODE", "k8s")
        settings = AdapterSettings.from_env()
        assert settings.resolved_job_spec_path == custom


class TestGetJobSpecPath:
    """Tests for config.get_job_spec_path() delegating to AdapterSettings."""

    def test_returns_path_when_file_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spec_file = tmp_path / "job.json"
        spec_file.write_text(json.dumps({"id": "test"}))
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(spec_file))
        assert get_job_spec_path() == spec_file

    def test_raises_when_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does_not_exist.json"
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(missing))
        with pytest.raises(FileNotFoundError, match="Job spec file not found"):
            get_job_spec_path()
