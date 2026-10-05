"""Shared fixtures and utilities for E2E tests."""

import importlib.util
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def mlflow_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[str, None, None]:
    """Start an isolated MLflow artifact server using the test interpreter."""
    if importlib.util.find_spec("mlflow") is None:
        pytest.skip("MLflow is not installed; run uv sync to install dev dependencies")

    server_dir = tmp_path_factory.mktemp("mlflow-server")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    # Do not inherit deployment credentials, workspace, or server configuration.
    server_env = {
        key: value for key, value in os.environ.items() if not key.startswith("MLFLOW_")
    }
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--backend-store-uri",
        f"sqlite:///{server_dir / 'mlflow.db'}",
        "--serve-artifacts",
        "--default-artifact-root",
        "mlflow-artifacts:/",
        "--artifacts-destination",
        str(server_dir / "artifacts"),
    ]
    log_file = server_dir / "server.log"
    with log_file.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=server_dir,
            env=server_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        try:
            deadline = time.monotonic() + 60
            with httpx.Client(timeout=1.0, trust_env=False) as health_client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        pytest.fail(
                            f"MLflow exited during startup:\n{log_file.read_text()}"
                        )
                    try:
                        if health_client.get(f"{base_url}/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.2)
                else:
                    pytest.fail(f"MLflow startup timed out:\n{log_file.read_text()}")
            yield base_url
        finally:
            # The MLflow CLI launches a child server; stop the entire process group.
            if os.name == "nt":
                process.terminate()
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def pytest_configure(config: Any) -> None:
    """Configure logging for E2E tests."""
    if config.getoption("--e2e-debug", default=False):
        e2e_logger = logging.getLogger("tests.e2e.conftest")
        e2e_logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
        e2e_logger.addHandler(handler)


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
    binary_path = shutil.which(
        "eval-hub-server",
        path=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
    )
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
