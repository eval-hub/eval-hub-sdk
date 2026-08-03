"""Unit tests for adapter auth: read_model_auth_key and resolve_model_credentials."""

from pathlib import Path

import pytest
from evalhub.adapter.auth import (
    ModelCredentials,
    read_model_auth_key,
    resolve_model_credentials,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def model_auth_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect _MODEL_AUTH_DIR to a temp directory."""
    auth_dir = tmp_path / "model"
    auth_dir.mkdir()
    monkeypatch.setattr("evalhub.adapter.auth._MODEL_AUTH_DIR", auth_dir)
    return auth_dir


# ---------------------------------------------------------------------------
# read_model_auth_key
# ---------------------------------------------------------------------------


class TestReadModelAuthKey:
    """Tests for read_model_auth_key()."""

    def test_reads_value_from_file(self, model_auth_dir: Path) -> None:
        """Returns the trimmed content of the key file."""
        (model_auth_dir / "api-key").write_text("my-secret-key\n")
        assert read_model_auth_key("api-key") == "my-secret-key"

    def test_returns_none_when_file_missing(self, model_auth_dir: Path) -> None:
        """Returns None when the key file does not exist."""
        assert read_model_auth_key("api-key") is None

    def test_returns_none_for_empty_file(self, model_auth_dir: Path) -> None:
        """Returns None when the key file is empty or whitespace-only."""
        (model_auth_dir / "api-key").write_text("   \n")
        assert read_model_auth_key("api-key") is None

    def test_returns_none_for_empty_key_name(self, model_auth_dir: Path) -> None:
        """Returns None when the key name is empty or whitespace."""
        assert read_model_auth_key("") is None
        assert read_model_auth_key("   ") is None

    def test_strips_key_name(self, model_auth_dir: Path) -> None:
        """Strips whitespace from the key name before lookup."""
        (model_auth_dir / "hf-token").write_text("tok123")
        assert read_model_auth_key("  hf-token  ") == "tok123"

    def test_returns_none_on_read_error(
        self, model_auth_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns None and logs a warning on read errors."""
        key_file = model_auth_dir / "api-key"
        key_file.write_text("value")
        monkeypatch.setattr(
            "pathlib.Path.read_text",
            lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk error")),
        )
        assert read_model_auth_key("api-key") is None


# ---------------------------------------------------------------------------
# ModelCredentials
# ---------------------------------------------------------------------------


class TestModelCredentials:
    """Tests for the ModelCredentials dataclass."""

    def test_defaults_to_none(self) -> None:
        """All fields default to None."""
        creds = ModelCredentials()
        assert creds.api_key is None
        assert creds.ca_cert_path is None
        assert creds.hf_token is None

    def test_fields_set(self) -> None:
        """Fields can be set via constructor."""
        creds = ModelCredentials(
            api_key="key", ca_cert_path="/certs/ca.pem", hf_token="hf-123"
        )
        assert creds.api_key == "key"
        assert creds.ca_cert_path == "/certs/ca.pem"
        assert creds.hf_token == "hf-123"

    def test_api_key_and_hf_token_hidden_in_repr(self) -> None:
        """api_key and hf_token are excluded from repr for security."""
        creds = ModelCredentials(api_key="secret", hf_token="also-secret")
        r = repr(creds)
        assert "secret" not in r
        assert "also-secret" not in r


# ---------------------------------------------------------------------------
# resolve_model_credentials
# ---------------------------------------------------------------------------


class TestResolveModelCredentials:
    """Tests for resolve_model_credentials()."""

    @pytest.fixture(autouse=True)
    def _clear_local_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Remove all EVALHUB_LOCAL_MODEL_* env vars for test isolation."""
        monkeypatch.delenv("EVALHUB_LOCAL_MODEL_API_KEY", raising=False)
        monkeypatch.delenv("EVALHUB_LOCAL_MODEL_CA_CERT_PATH", raising=False)
        monkeypatch.delenv("EVALHUB_LOCAL_MODEL_HF_TOKEN", raising=False)

    def test_reads_from_files_k8s_style(
        self,
        model_auth_dir: Path,
    ) -> None:
        """Reads api-key, ca_cert, and hf-token from mounted secret files."""
        (model_auth_dir / "api-key").write_text("api-key:ref")
        (model_auth_dir / "ca_cert").write_text("-----BEGIN CERTIFICATE-----\n...")
        (model_auth_dir / "hf-token").write_text("hf-abc")

        creds = resolve_model_credentials()

        assert creds.api_key == "api-key:ref"
        assert creds.ca_cert_path == str(model_auth_dir / "ca_cert")
        assert creds.hf_token == "hf-abc"

    def test_falls_back_to_env_vars(
        self, model_auth_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uses EVALHUB_LOCAL_MODEL_* env vars when files are absent."""
        monkeypatch.setenv("EVALHUB_LOCAL_MODEL_API_KEY", "real-api-key")
        monkeypatch.setenv("EVALHUB_LOCAL_MODEL_CA_CERT_PATH", "/local/ca.pem")
        monkeypatch.setenv("EVALHUB_LOCAL_MODEL_HF_TOKEN", "hf-local")

        creds = resolve_model_credentials()

        assert creds.api_key == "real-api-key"
        assert creds.ca_cert_path == "/local/ca.pem"
        assert creds.hf_token == "hf-local"

    def test_file_takes_precedence_over_env(
        self, model_auth_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """K8s-mounted files take precedence over env vars."""
        (model_auth_dir / "api-key").write_text("from-file")
        (model_auth_dir / "ca_cert").write_text("cert-content")
        (model_auth_dir / "hf-token").write_text("hf-from-file")
        monkeypatch.setenv("EVALHUB_LOCAL_MODEL_API_KEY", "from-env")
        monkeypatch.setenv("EVALHUB_LOCAL_MODEL_CA_CERT_PATH", "/env/ca.pem")
        monkeypatch.setenv("EVALHUB_LOCAL_MODEL_HF_TOKEN", "hf-from-env")

        creds = resolve_model_credentials()

        assert creds.api_key == "from-file"
        assert creds.ca_cert_path == str(model_auth_dir / "ca_cert")
        assert creds.hf_token == "hf-from-file"

    def test_returns_none_when_nothing_configured(
        self,
        model_auth_dir: Path,
    ) -> None:
        """Returns all-None credentials when no files or env vars exist."""
        creds = resolve_model_credentials()

        assert creds.api_key is None
        assert creds.ca_cert_path is None
        assert creds.hf_token is None

    def test_partial_configuration(
        self, model_auth_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Supports partial configuration (e.g. api_key from env, no cert)."""
        monkeypatch.setenv("EVALHUB_LOCAL_MODEL_API_KEY", "only-key")

        creds = resolve_model_credentials()

        assert creds.api_key == "only-key"
        assert creds.ca_cert_path is None
        assert creds.hf_token is None
