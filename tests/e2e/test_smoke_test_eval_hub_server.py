import multiprocessing
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
from evalhub import SyncEvalHubClient
from httpx import HTTPStatusError

from .conftest import _ensure_server_binary, _run_server


@pytest.fixture
def evalhub_server() -> Generator[str, None, None]:
    """
    Start the eval-hub server in a separate process and wait for it to be ready.

    Yields:
        str: The base URL of the running server (e.g., "http://localhost:8080")
    """
    # Ensure the binary is available (copy from local eval-hub repo if needed)
    if not _ensure_server_binary():
        pytest.skip(
            "eval-hub-server binary not available. "
            "Build it locally or install from a release with binaries."
        )

    # Create temporary directory for server files
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"

        # Create minimal config for testing
        # Only service (port + files) and database (in-memory SQLite) are required
        config_content = f"""service:
  port: 8080
  ready_file: "{tmpdir}/repo-ready"
  termination_file: "{tmpdir}/termination-log"
database:
  driver: sqlite
  url: file::memory:?mode=memory&cache=shared
"""
        config_file.write_text(config_content)

        # Start server in a separate process
        server_process = multiprocessing.Process(
            target=_run_server, args=(str(config_dir.parent),)
        )
        server_process.start()

        # Wait for server to be ready
        base_url = "http://localhost:8080"
        max_retries = 5
        base_delay = 0.5

        for i in range(max_retries):
            try:
                # Use health endpoint to check if server is ready
                response = httpx.get(f"{base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    break
            except (httpx.ConnectError, httpx.TimeoutException):
                if i == max_retries - 1:
                    server_process.terminate()
                    server_process.join()
                    raise RuntimeError("Server failed to start within expected time")
                # Exponential backoff: 0.5s, 1s, 2s, 4s
                time.sleep(base_delay * (2**i))

        yield base_url

        # Cleanup: terminate the server process
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
            server_process.join()


@pytest.mark.e2e
def test_evaluations_providers_endpoint(evalhub_server: str) -> None:
    """Test that the evaluations providers endpoint is accessible."""
    with SyncEvalHubClient(base_url=evalhub_server) as client:
        providers = client.providers.list()
        assert isinstance(providers, list)


@pytest.mark.e2e
def test_collections_endpoint(evalhub_server: str) -> None:
    """Test that the collections endpoint returns 501 Not Implemented."""
    with SyncEvalHubClient(base_url=evalhub_server) as client:
        with pytest.raises(HTTPStatusError) as exc_info:
            client.collections.list()
        assert exc_info.value.response.status_code == 501


@pytest.mark.e2e
def test_jobs_endpoint(evalhub_server: str) -> None:
    """Test that the jobs endpoint is accessible."""
    with SyncEvalHubClient(base_url=evalhub_server) as client:
        jobs = client.jobs.list()
        assert isinstance(jobs, list)


@pytest.mark.e2e
def test_health_endpoint(evalhub_server: str) -> None:
    """Test that the health endpoint is accessible."""
    with SyncEvalHubClient(base_url=evalhub_server) as client:
        health = client.health()
        assert health is not None
