"""Shared fixtures and utilities for E2E tests."""

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


def _run_server(config_parent_dir: str, log_file: str | None = None) -> None:
    os.chdir(config_parent_dir)
    if log_file:
        import sys
        sys.stdout = open(log_file, 'w')
        sys.stderr = sys.stdout
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
def evalhub_server_with_real_config() -> Generator[str, None, None]:
    """
    Start eval-hub server with real config from tests/e2e/config directory.

    This fixture uses the real configuration from the local config directory as-is,
    including all provider definitions and settings from the eval-hub repository.

    Yields:
        str: The base URL of the running server (e.g., "http://localhost:8080")

    Raises:
        pytest.skip: If server binary or config directory is not available
    """
    # Ensure binary is available
    if not _ensure_server_binary():
        pytest.skip(
            "eval-hub-server binary not available. "
            "Build it locally or install from a release with binaries."
        )

    # Check that config directory exists
    config_source_dir = Path(__file__).parent / "config"
    if not config_source_dir.exists() or not config_source_dir.is_dir():
        pytest.skip(
            "tests/e2e/config directory not found. "
            "Please create it and copy config files from eval-hub repository."
        )

    config_file = config_source_dir / "config.yaml"
    if not config_file.exists():
        pytest.skip(
            "config.yaml not found in tests/e2e/config directory. "
            "Please ensure the config directory is properly set up."
        )

    # Create temporary directory for server files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy entire config directory to temp location (including providers subdirectory)
        config_dir = Path(tmpdir) / "config"
        shutil.copytree(config_source_dir, config_dir)

        # Debug: print directory structure
        print(f"\n\n===== SERVER DIRECTORY STRUCTURE =====")
        print(f"Working dir will be: {tmpdir}")
        for item in sorted(Path(tmpdir).rglob("*")):
            rel = item.relative_to(tmpdir)
            print(f"  {rel}{'/' if item.is_dir() else ''}")
        print("=" * 50)

        # Create log file for server output
        log_file = str(Path(tmpdir) / "server.log")

        # Start server in a separate process
        server_process = multiprocessing.Process(
            target=_run_server, args=(str(config_dir.parent), log_file)
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

        # Debug: Print server logs
        if Path(log_file).exists():
            print(f"\n\n===== SERVER LOGS =====")
            with open(log_file) as f:
                logs = f.read()
                # Only print first 3000 chars to avoid flooding output
                if len(logs) > 3000:
                    print(logs[:3000] + f"\n... ({len(logs) - 3000} more chars)")
                else:
                    print(logs)
            print("=" * 50)

        yield base_url

        # Cleanup: terminate the server process
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
            server_process.join()
