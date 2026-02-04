import multiprocessing
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
from evalhub import SyncEvalHubClient
from httpx import HTTPStatusError


def _run_server(config_parent_dir: str) -> None:
    os.chdir(config_parent_dir)
    from evalhub_server.main import main

    main()


def _ensure_server_binary() -> bool:
    """
    TODO: this should be REMOVED when eval-hub-server is moved to a pypi release
    TODO: this is temporary until eval-hub-server is release'd on Pypi because we need the binary(ies)
    """
    try:
        from evalhub_server import get_binary_path

        # Check if binary already exists
        try:
            binary_path = get_binary_path()
            return Path(binary_path).exists()
        except FileNotFoundError:
            pass

        # Try to copy from local eval-hub repo
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "darwin":
            binary_name = (
                f"eval-hub-darwin-{'arm64' if machine == 'arm64' else 'amd64'}"
            )
        elif system == "linux":
            binary_name = f"eval-hub-linux-{'arm64' if 'aarch64' in machine or 'arm64' in machine else 'amd64'}"
        else:
            return False

        # Look for eval-hub repo (assume it's a sibling directory)
        eval_hub_repo = Path(__file__).parent.parent.parent.parent / "eval-hub"
        binary_source = eval_hub_repo / "bin" / binary_name

        if binary_source.exists():
            # Copy to evalhub_server package
            import evalhub_server

            pkg_dir = Path(evalhub_server.__file__).parent
            binaries_dir = pkg_dir / "binaries"
            binaries_dir.mkdir(exist_ok=True)

            binary_dest = binaries_dir / binary_name
            shutil.copy2(binary_source, binary_dest)
            binary_dest.chmod(0o755)
            return True

        return False
    except Exception:
        return False


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
