"""Helpers for resolving model auth from environment.

In Kubernetes the auth files live under a well-known mount
(``/var/run/secrets/model``).  In local mode, when ``job_spec.model.auth``
is present, the directory is derived from its ``secret_ref`` field which
must be a ``file:///`` URL pointing to a directory that may contain
``api-key``, ``hf-token``, and ``ca_cert`` files.

The job spec is discovered automatically from ``EVALHUB_JOB_SPEC_PATH``
(or the mode-dependent default) — callers do not need to pass it in.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .config import EvalHubMode, get_job_spec_path
from .models.job import JobSpec
from .settings import AdapterSettings

logger = logging.getLogger(__name__)
_MODEL_AUTH_DIR = Path("/var/run/secrets/model")


def _parse_file_url(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.path == "":
        raise ValueError(f"secret_ref must be a file:/// URL in local mode, got: {url}")
    if parsed.netloc:
        raise ValueError(
            f"Malformed file URL (use file:///path, not file://path): {url}"
        )
    return Path(parsed.path)


def _resolve_auth_dir() -> Path:
    try:
        job_spec = JobSpec.from_file(get_job_spec_path())
    except (FileNotFoundError, ValueError) as exc:
        logger.debug("Job spec not available: %s", exc)
        return _MODEL_AUTH_DIR

    settings = AdapterSettings.from_env()
    if settings.mode == EvalHubMode.LOCAL and job_spec.model.auth is not None:
        ref = _parse_file_url(job_spec.model.auth.secret_ref)
        if ref.is_symlink() and not _is_projected_data_dir(ref):
            raise ValueError(f"secret_ref must not be a symlink: {ref}")
        if not ref.is_dir():
            raise ValueError(
                f"secret_ref does not point to an existing directory: {ref}"
            )
        return ref
    return _MODEL_AUTH_DIR


def _is_projected_data_dir(path: Path) -> bool:
    """Return whether *path* is Kubernetes' hidden ``..data`` directory link."""
    if path.name != "..data" or not path.is_symlink():
        return False
    try:
        resolved = path.resolve(strict=True)
        parent = path.parent.resolve(strict=True)
    except OSError:
        return False
    return resolved.is_dir() and resolved.parent == parent


def _is_projected_file(path: Path, auth_dir: Path) -> bool:
    """Return whether *path* is a safe Kubernetes projected-volume file link.

    Kubernetes' atomic writer exposes each projected key as ``key ->
    ..data/key``.  Validate both the link shape and its resolved location so
    that arbitrary symlinks cannot make local-mode auth read outside the
    configured directory.
    """
    if not path.is_symlink():
        return False
    try:
        link_target = path.readlink()
        auth_root = auth_dir.resolve(strict=True)
        data_dir = (auth_dir / "..data").resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError:
        return False

    return (
        link_target.parent == Path("..data")
        and link_target.name == path.name
        and data_dir.is_dir()
        and data_dir.parent == auth_root
        and resolved.is_file()
        and auth_root in resolved.parents
    )


def _is_regular_file(path: Path, auth_dir: Path) -> bool:
    if not path.is_file():
        return False
    return not path.is_symlink() or _is_projected_file(path, auth_dir)


def _read_key_from_dir(auth_dir: Path, key_name: str) -> str | None:
    if not key_name or "/" in key_name or "\\" in key_name or key_name in (".", ".."):
        return None
    path = auth_dir / key_name
    if not _is_regular_file(path, auth_dir):
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read model auth key %s", key_name, exc_info=exc)
        return None
    return value or None


def read_model_auth_key(key_name: str) -> str | None:
    """Read a specific key from the model auth secret directory.

    The auth directory is resolved automatically from the job spec
    (via ``EVALHUB_JOB_SPEC_PATH``) in local mode, or from the default
    Kubernetes mount path.
    """
    cleaned = key_name.strip()
    if not cleaned:
        return None
    return _read_key_from_dir(_resolve_auth_dir(), cleaned)


@dataclass
class ModelCredentials:
    """Resolved model credentials.

    In Kubernetes ``api_key`` holds a ref token (e.g. ``"api-key:ref"``)
    resolved by the sidecar proxy.  In local mode ``api_key`` is the
    actual credential — the adapter sets the ``Authorization`` header
    directly.

    ``hf_token`` is the Hugging Face token from the ``hf-token`` file.
    ``ca_cert_path`` points to the ``ca_cert`` file when present.
    """

    api_key: str | None = field(default=None, repr=False)
    hf_token: str | None = field(default=None, repr=False)
    ca_cert_path: Path | None = field(default=None)


def resolve_model_credentials() -> ModelCredentials:
    """Resolve model authentication from the auth secret directory.

    Reads ``api-key``, ``hf-token``, and checks for ``ca_cert``.
    The auth directory is resolved automatically from the job spec.
    """
    auth_dir = _resolve_auth_dir()
    ca_cert = auth_dir / "ca_cert"
    return ModelCredentials(
        api_key=_read_key_from_dir(auth_dir, "api-key"),
        hf_token=_read_key_from_dir(auth_dir, "hf-token"),
        ca_cert_path=ca_cert if _is_regular_file(ca_cert, auth_dir) else None,
    )
