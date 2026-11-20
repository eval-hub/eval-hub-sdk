"""On-demand NeMo Gym ATIF fixtures fetched from a pinned upstream commit.

The NeMo Gym ATIF v1.7 documents used by the contract tests are not vendored into
this repository. They are downloaded once per test session from a fixed commit of
``NVIDIA-NeMo/Gym``, verified against a pinned SHA-256, and cached in a git-ignored
directory next to this module.

Two knobs make the fetch predictable:

- ``EVALHUB_GYM_FIXTURE_CACHE`` overrides the cache directory (for example to a
  prepared directory on an air-gapped machine).
- A transport failure skips the dependent tests with the URL in the reason. A
  digest mismatch for a pinned commit fails instead: upstream bytes for a fixed
  commit must not change, so a mismatch means the cached copy or the pin is wrong.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

import pytest

NEMO_GYM_REPOSITORY: Final = "NVIDIA-NeMo/Gym"
NEMO_GYM_COMMIT: Final = "76116d5d78c8f613cf2038231c66cc02d7ae4a5f"

# ATIF v1.7 document produced by NeMo Gym's relay qualification run, carrying
# producer provenance in the declared ``extra`` object.
NEMO_GYM_ATIF_V1_7_PATH: Final = (
    "tests/unit_tests/trajectory/fixtures/"
    "relay_atif_v1_7_responses_tool_trajectory.json"
)
NEMO_GYM_ATIF_V1_7_SHA256: Final = (
    "8d9c3c2d21c4ef0a8d9488eac560b1fdfaa532712b8a4fc628023a94d0bab825"
)

_DOWNLOAD_TIMEOUT_SECONDS: Final = 30
_DOWNLOAD_ATTEMPTS: Final = 2


def _raw_url(source_path: str) -> str:
    """Return the raw GitHub URL of ``source_path`` at the pinned commit."""
    return f"https://raw.githubusercontent.com/{NEMO_GYM_REPOSITORY}/{NEMO_GYM_COMMIT}/{source_path}"


def _cache_dir() -> Path:
    """Return the directory holding downloaded fixtures."""
    override = os.environ.get("EVALHUB_GYM_FIXTURE_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(__file__).parent / ".cache" / f"nemogym-{NEMO_GYM_COMMIT[:12]}"


def fetch_pinned_fixture(source_path: str, expected_sha256: str) -> bytes:
    """Return the bytes of a NeMo Gym fixture at the pinned commit.

    Args:
        source_path: Repository-relative path of the fixture in ``NVIDIA-NeMo/Gym``.
        expected_sha256: Hex SHA-256 the downloaded bytes must match.

    Returns:
        The verified fixture bytes.

    Raises:
        pytest.fail.Exception: The downloaded bytes do not match ``expected_sha256``.
    """
    cached = _cache_dir() / Path(source_path).name
    if cached.is_file():
        cached_bytes = cached.read_bytes()
        if hashlib.sha256(cached_bytes).hexdigest() == expected_sha256:
            return cached_bytes

    url = _raw_url(source_path)
    last_error: Exception | None = None
    for _ in range(_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp:
                payload: bytes = resp.read()
        except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - network
            last_error = exc
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_sha256:
            pytest.fail(
                f"NeMo Gym fixture {source_path} at {NEMO_GYM_COMMIT} has SHA-256 "
                f"{digest}, expected {expected_sha256}",
                pytrace=False,
            )
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(payload)
        return payload

    pytest.skip(  # pragma: no cover - offline environments
        f"cannot download pinned NeMo Gym fixture {url}: {last_error}"
    )


@pytest.fixture(scope="session")
def nemogym_atif_v1_7_document() -> dict[str, Any]:
    """Parsed ATIF v1.7 document from the pinned NeMo Gym commit."""
    raw = fetch_pinned_fixture(NEMO_GYM_ATIF_V1_7_PATH, NEMO_GYM_ATIF_V1_7_SHA256)
    document: dict[str, Any] = json.loads(raw)
    return document
