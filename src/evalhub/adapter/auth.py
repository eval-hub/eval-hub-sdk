"""Helpers for resolving model auth from environment."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelCredentials:
    """Resolved model credentials from environment."""

    api_key: str | None = field(default=None, repr=False)
    ca_cert_path: str | None = None
    _service_account_token: str | None = field(default=None, repr=False)

    @property
    def auth_headers(self) -> dict[str, str]:
        if self._service_account_token:
            return {"Authorization": f"Bearer {self._service_account_token}"}
        return {}


def resolve_model_credentials() -> ModelCredentials:
    """Resolve model authentication from the pod environment.

    Reads MODEL_AUTH_API_KEY_PATH and MODEL_AUTH_CA_CERT_PATH env vars set by EvalHub.
    """
    creds = ModelCredentials()

    api_key_path = os.environ.get("MODEL_AUTH_API_KEY_PATH")
    if api_key_path:
        path = Path(api_key_path)
        if path.is_file():
            try:
                api_key = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                logger.warning("Failed to read model API key file", exc_info=exc)
            else:
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

    ca_path = os.environ.get("MODEL_AUTH_CA_CERT_PATH")
    if ca_path:
        path = Path(ca_path)
        if path.is_file():
            try:
                ca_cert = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                logger.warning("Failed to read model CA cert file", exc_info=exc)
            else:
                if ca_cert:
                    creds.ca_cert_path = ca_path

    return creds
