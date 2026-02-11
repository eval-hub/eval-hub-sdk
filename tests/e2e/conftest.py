"""Shared fixtures and utilities for E2E tests."""

import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest


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


def _run_server(working_dir: str) -> None:
    """
    Run the eval-hub server binary in the specified working directory.

    This function is intended to be used as a target for multiprocessing.Process.

    Args:
        working_dir: Directory containing the config subdirectory
    """
    from evalhub_server import get_binary_path

    binary_path = get_binary_path()
    subprocess.run([binary_path], cwd=working_dir, check=False)


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
        print("\n\n===== SERVER DIRECTORY STRUCTURE =====")
        print(f"Working dir will be: {tmpdir}")
        for item in sorted(Path(tmpdir).rglob("*")):
            rel = item.relative_to(tmpdir)
            print(f"  {rel}{'/' if item.is_dir() else ''}")
        print("=" * 50)

        # Create log file for server output
        log_file = Path(tmpdir) / "server.log"

        # Kill any process already using port 8080
        port = 8080
        if _kill_process_on_port(port):
            print(f"\n⚠️  WARNING: Killed existing process on port {port}")
            print(
                "    (This is normal if a previous test run didn't clean up properly)\n"
            )
            # Give the OS a moment to release the port
            time.sleep(0.5)

        # Get the server binary path and start it directly as a subprocess
        from evalhub_server import get_binary_path

        binary_path = get_binary_path()

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
            print("\n\n===== SERVER LOGS =====")
            with open(log_file) as f:
                logs = f.read()
                # Only print first 3000 chars to avoid flooding output
                if len(logs) > 3000:
                    print(logs[:3000] + f"\n... ({len(logs) - 3000} more chars)")
                else:
                    print(logs)
            print("=" * 50)

        yield base_url

        # Cleanup: terminate the server subprocess
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
