"""Shared fixtures and utilities for E2E tests."""

import logging
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest

logger = logging.getLogger(__name__)


def _kill_process_on_port(port: int) -> bool:
    """
    Kill any process using the specified port.

    Returns:
        bool: True if a process was killed, False if no process was found
    """
    try:
        if platform.system() == "Windows":
            # Windows: use netstat and taskkill
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = parts[-1]
                    subprocess.run(["taskkill", "/PID", pid, "/F"], timeout=5)
                    return True
        else:
            # Unix-like systems: use lsof
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5
            )
            pids = result.stdout.strip().split()
            if pids:
                for pid in pids:
                    subprocess.run(["kill", "-9", pid], timeout=5)
                return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
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
    binary_path = shutil.which("eval-hub-server")
    if not binary_path:
        pytest.skip(
            "eval-hub-server binary not available. "
            "Install it with: pip install 'eval-hub-sdk[server]'"
        )
    assert binary_path is not None  # narrow type for mypy

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

    # Create temporary directory for server files (preserved after run for debugging of server logfiles, etc)
    tmpdir = tempfile.mkdtemp(prefix="evalhub-e2e-")
    server_process = None
    try:
        logger.debug(f"\nTemp directory for this run: {tmpdir}")
        # Copy entire config directory to temp location (including providers subdirectory)
        config_dir = Path(tmpdir) / "config"
        shutil.copytree(config_source_dir, config_dir)

        # Debug: print directory structure
        dir_listing = "\n".join(
            f"  {item.relative_to(tmpdir)}{'/' if item.is_dir() else ''}"
            for item in sorted(Path(tmpdir).rglob("*"))
        )
        logger.debug(
            "Server directory structure (working dir: %s):\n%s", tmpdir, dir_listing
        )

        # Create log file for server output
        log_file = Path(tmpdir) / "server.log"

        # Kill any process already using port 8080
        port = 8080
        if _kill_process_on_port(port):
            logger.warning(
                "Killed existing process on port %d (normal if a previous test run didn't clean up properly)",
                port,
            )
            # Give the OS a moment to release the port
            time.sleep(0.5)

        with open(log_file, "w") as log_f:
            server_process = subprocess.Popen(
                [binary_path, "--local"],
                cwd=str(config_dir.parent),
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )

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
                    server_process.wait()
                    raise RuntimeError("Server failed to start within expected time")
                # Exponential backoff: 0.5s, 1s, 2s, 4s
                time.sleep(base_delay * (2**i))

        # Debug: Print server logs
        if log_file.exists():
            with open(log_file) as f:
                logs = f.read()
            if len(logs) > 3000:
                logs = logs[:3000] + f"\n... ({len(logs) - 3000} more chars)"
            logger.debug("Server log file: %s\n%s", log_file.resolve(), logs)

        yield base_url
    finally:
        # Cleanup: terminate the server subprocess
        if server_process is not None:
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
