"""Helpers for resolving model auth from environment."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
_MODEL_AUTH_DIR = Path("/var/run/secrets/model")


def read_model_auth_key(key_name: str) -> str | None:
    """Read a specific key from the mounted model auth secret."""
    cleaned = key_name.strip()
    if not cleaned:
        return None
    path = _MODEL_AUTH_DIR / cleaned
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Failed to read model auth key %s", cleaned, exc_info=exc)
        return None
    return value or None


@dataclass
class ModelCredentials:
    """Resolved model credentials from environment."""

    api_key: str | None = field(default=None, repr=False)
    ca_cert_path: str | None = None
    _service_account_token: str | None = field(default=None, repr=False)

    @property
    def auth_headers(self) -> dict[str, str]:
        """Auth headers derived from the ServiceAccount token only.

        The API key is intentionally excluded and must be consumed separately
        by the adapter.
        """
        if self._service_account_token:
            return {"Authorization": f"Bearer {self._service_account_token}"}
        return {}


def resolve_model_credentials() -> ModelCredentials:
    """Resolve model authentication from the pod environment.

    Reads credentials from the mounted model auth secret path.
    """
    creds = ModelCredentials()

    api_key = read_model_auth_key("api-key")
    if api_key:
        creds.api_key = api_key

    if not creds.api_key:
        sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        path = Path(sa_token_path)
        if path.is_file():
            try:
                sa_token = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                logger.warning(
                    "Failed to read service account token file", exc_info=exc
                )
            else:
                if sa_token:
                    creds._service_account_token = sa_token

    ca_cert = read_model_auth_key("ca_cert")
    if ca_cert:
        creds.ca_cert_path = str(_MODEL_AUTH_DIR / "ca_cert")

    return creds
