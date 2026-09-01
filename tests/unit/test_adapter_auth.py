"""Unit tests for adapter model auth resolution."""

from pathlib import Path
from typing import Any

import pytest
from evalhub.adapter.auth import (
    ModelCredentials,
    _resolve_auth_dir,
    read_model_auth_key,
    resolve_model_credentials,
)
from evalhub.adapter.models.job import JobSpec
from evalhub.models.api import ModelAuth, ModelConfig

pytestmark = pytest.mark.unit

_MINIMAL_JOB_SPEC_FIELDS: dict[str, Any] = {
    "id": "job-1",
    "provider_id": "prov-1",
    "benchmark_id": "bench-1",
    "benchmark_index": 0,
    "parameters": {},
    "callback_url": "http://localhost:8080",
}


def _file_url(path: Path) -> str:
    return path.as_uri()


def _make_job_spec(secret_ref: str) -> JobSpec:
    return JobSpec(
        model=ModelConfig(
            url="http://model:8000/v1",
            name="test-model",
            auth=ModelAuth(secret_ref=secret_ref),
        ),
        **_MINIMAL_JOB_SPEC_FIELDS,
    )


def _make_job_spec_no_auth() -> JobSpec:
    return JobSpec(
        model=ModelConfig(
            url="http://model:8000/v1",
            name="test-model",
        ),
        **_MINIMAL_JOB_SPEC_FIELDS,
    )


def _write_job_json(tmp_path: Path, spec: JobSpec) -> Path:
    """Write a job.json file and return the path."""
    job_json = tmp_path / "job.json"
    job_json.write_text(spec.model_dump_json(), encoding="utf-8")
    return job_json


# ---------------------------------------------------------------------------
# _resolve_auth_dir
# ---------------------------------------------------------------------------


class TestResolveAuthDir:
    def test_no_job_spec_file_returns_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(tmp_path / "missing.json"))
        assert _resolve_auth_dir() == Path("/var/run/secrets/model")

    def test_k8s_mode_returns_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "k8s")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert _resolve_auth_dir() == Path("/var/run/secrets/model")

    def test_local_mode_uses_secret_ref(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert _resolve_auth_dir() == auth_dir

    def test_local_mode_no_auth_returns_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        spec = _make_job_spec_no_auth()
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert _resolve_auth_dir() == Path("/var/run/secrets/model")

    def test_local_mode_nonexistent_dir_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        spec = _make_job_spec(_file_url(tmp_path / "does-not-exist"))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        with pytest.raises(ValueError, match="existing directory"):
            _resolve_auth_dir()

    def test_local_mode_plain_path_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        spec = _make_job_spec(str(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        with pytest.raises(ValueError, match="must be a file:/// URL"):
            _resolve_auth_dir()

    def test_local_mode_malformed_file_url_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        spec = _make_job_spec(f"file://{auth_dir.name}/{auth_dir}")
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        with pytest.raises(ValueError, match="file:///path, not file://path"):
            _resolve_auth_dir()

    def test_local_mode_symlink_dir_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        real_dir = tmp_path / "real-auth"
        real_dir.mkdir()
        link = tmp_path / "link-auth"
        link.symlink_to(real_dir)
        spec = _make_job_spec(_file_url(link))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        with pytest.raises(ValueError, match="must not be a symlink"):
            _resolve_auth_dir()

    def test_local_mode_accepts_projected_volume_layout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        version_dir = auth_dir / "..2026_08_31_00_00_00"
        version_dir.mkdir()
        (auth_dir / "..data").symlink_to(version_dir.name)
        for key_name, value in {
            "api-key": "api-key-value",
            "hf-token": "hf-token-value",
            "ca_cert": "certificate",
        }.items():
            (version_dir / key_name).write_text(value)
            (auth_dir / key_name).symlink_to(Path("..data") / key_name)

        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert _resolve_auth_dir() == auth_dir


# ---------------------------------------------------------------------------
# read_model_auth_key
# ---------------------------------------------------------------------------


class TestReadModelAuthKey:
    def test_reads_key_from_local_auth_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        (auth_dir / "api-key").write_text("my-secret-key\n")
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert read_model_auth_key("api-key") == "my-secret-key"

    def test_returns_none_when_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert read_model_auth_key("api-key") is None

    def test_returns_none_for_empty_key_name(self) -> None:
        assert read_model_auth_key("") is None
        assert read_model_auth_key("   ") is None

    def test_returns_none_for_empty_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        (auth_dir / "api-key").write_text("")
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert read_model_auth_key("api-key") is None

    def test_strips_whitespace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        (auth_dir / "api-key").write_text("  key-with-whitespace  \n")
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert read_model_auth_key("api-key") == "key-with-whitespace"

    @pytest.mark.parametrize(
        "key_name",
        [
            "/etc/passwd",
            "../secret",
            "../../etc/shadow",
            "subdir/file",
            "a\\b",
            ".",
            "..",
        ],
    )
    def test_rejects_path_traversal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        key_name: str,
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert read_model_auth_key(key_name) is None

    def test_rejects_symlinked_key_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        real_key = tmp_path / "real-api-key"
        real_key.write_text("secret-via-symlink")
        (auth_dir / "api-key").symlink_to(real_key)
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert read_model_auth_key("api-key") is None
        assert (
            "Ignoring non-projected symlink for model auth file api-key" in caplog.text
        )

    def test_reads_keys_from_projected_volume_symlinks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        version_dir = auth_dir / "..2026_08_31_00_00_00"
        version_dir.mkdir()
        (auth_dir / "..data").symlink_to(version_dir.name)
        for key_name, value in {
            "api-key": "api-key-value",
            "hf-token": "hf-token-value",
            "ca_cert": "certificate",
        }.items():
            (version_dir / key_name).write_text(value)
            (auth_dir / key_name).symlink_to(Path("..data") / key_name)

        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        assert read_model_auth_key("api-key") == "api-key-value"
        assert read_model_auth_key("hf-token") == "hf-token-value"

    def test_backward_compat_no_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(tmp_path / "missing.json"))
        result = read_model_auth_key("api-key")
        assert result is None


# ---------------------------------------------------------------------------
# resolve_model_credentials
# ---------------------------------------------------------------------------


class TestResolveModelCredentials:
    def test_resolves_all_keys_local_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        (auth_dir / "api-key").write_text("real-api-key")
        (auth_dir / "hf-token").write_text("hf-abc123")
        (auth_dir / "ca_cert").write_text("-----BEGIN CERTIFICATE-----")
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        creds = resolve_model_credentials()

        assert creds.api_key == "real-api-key"
        assert creds.hf_token == "hf-abc123"
        assert creds.ca_cert_path == auth_dir / "ca_cert"

    def test_missing_optional_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        (auth_dir / "api-key").write_text("only-api-key")
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        creds = resolve_model_credentials()

        assert creds.api_key == "only-api-key"
        assert creds.hf_token is None
        assert creds.ca_cert_path is None

    def test_no_ca_cert_path_when_not_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        (auth_dir / "ca_cert").mkdir()
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        creds = resolve_model_credentials()
        assert creds.ca_cert_path is None

    def test_rejects_symlinked_ca_cert(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        real_cert = tmp_path / "real_ca_cert"
        real_cert.write_text("-----BEGIN CERTIFICATE-----")
        (auth_dir / "ca_cert").symlink_to(real_cert)
        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        creds = resolve_model_credentials()
        assert creds.ca_cert_path is None

    def test_resolves_projected_volume_symlinks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_MODE", "local")
        auth_dir = tmp_path / "model-auth"
        auth_dir.mkdir()
        version_dir = auth_dir / "..2026_08_31_00_00_00"
        version_dir.mkdir()
        (auth_dir / "..data").symlink_to(version_dir.name)
        for key_name, value in {
            "api-key": "api-key-value",
            "hf-token": "hf-token-value",
            "ca_cert": "certificate",
        }.items():
            (version_dir / key_name).write_text(value)
            (auth_dir / key_name).symlink_to(Path("..data") / key_name)

        spec = _make_job_spec(_file_url(auth_dir))
        job_json = _write_job_json(tmp_path, spec)
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_json))

        creds = resolve_model_credentials()

        assert creds.api_key == "api-key-value"
        assert creds.hf_token == "hf-token-value"
        assert creds.ca_cert_path == auth_dir / "ca_cert"

    def test_backward_compat_no_job_spec(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(tmp_path / "missing.json"))
        creds = resolve_model_credentials()
        assert isinstance(creds, ModelCredentials)
        assert creds.api_key is None
        assert creds.hf_token is None
        assert creds.ca_cert_path is None


# ---------------------------------------------------------------------------
# ModelCredentials
# ---------------------------------------------------------------------------


class TestModelCredentials:
    def test_defaults(self) -> None:
        creds = ModelCredentials()
        assert creds.api_key is None
        assert creds.hf_token is None
        assert creds.ca_cert_path is None

    def test_secrets_not_in_repr(self) -> None:
        creds = ModelCredentials(api_key="secret", hf_token="token")
        r = repr(creds)
        assert "secret" not in r
        assert "token" not in r

    def test_ca_cert_path_in_repr(self) -> None:
        creds = ModelCredentials(ca_cert_path=Path("/some/ca_cert"))
        assert "ca_cert" in repr(creds)
