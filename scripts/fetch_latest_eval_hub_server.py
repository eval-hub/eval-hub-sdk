#!/usr/bin/env python3
"""Fetch and install the latest eval-hub-server wheel from GitHub Actions.

This script:
1. Detects the current platform (OS + architecture)
2. Fetches the latest successful workflow run from GitHub Actions
3. Downloads the appropriate wheel artifact for the platform
4. Installs it using the UV package manager

Environment variables:
    GITHUB_TOKEN: Optional GitHub personal access token for API authentication.
                  Without this, API requests are rate-limited.
                  ALTERNATIVELY use project root's/.github_token file (read-only for Actions)

Usage:
    python scripts/fetch_latest_eval_hub_server.py
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Check for PyGithub availability early
try:
    from github import Github, GithubException
    from github.WorkflowRun import WorkflowRun
except ImportError:
    print(
        "Error: PyGithub is not installed in the current environment.",
        file=sys.stderr,
    )
    print("Please run: uv sync", file=sys.stderr)
    print(
        "This will install all dev dependencies including PyGithub. Then run source .venv/bin/activate",
        file=sys.stderr,
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class PlatformNotSupportedError(Exception):
    """Raised when the current platform is not supported."""

    pass


class ArtifactNotFoundError(Exception):
    """Raised when the required artifact is not found in the workflow run."""

    pass


def detect_platform() -> tuple[str, str]:
    """Detect the current operating system and architecture.

    Returns:
        tuple[str, str]: A tuple of (platform_name, architecture) where:
            - platform_name is one of: 'macosx', 'manylinux', 'win'
            - architecture is one of: 'arm64', 'x86_64', 'aarch64', 'amd64'

    Raises:
        PlatformNotSupportedError: If the platform/architecture combination
                                   is not supported.
    """
    system = platform.system()
    machine = platform.machine()

    logger.debug(f"Detected system: {system}, machine: {machine}")

    # Map system to platform name
    if system == "Darwin":
        platform_name = "macosx"
        if machine == "arm64":
            return platform_name, "arm64"
        elif machine == "x86_64":
            return platform_name, "x86_64"
        else:
            raise PlatformNotSupportedError(
                f"Unsupported macOS architecture: {machine}"
            )
    elif system == "Linux":
        platform_name = "manylinux"
        if machine in ["aarch64", "arm64"]:
            return platform_name, "aarch64"
        elif machine == "x86_64":
            return platform_name, "x86_64"
        else:
            raise PlatformNotSupportedError(
                f"Unsupported Linux architecture: {machine}"
            )
    elif system == "Windows":
        platform_name = "win"
        # Windows typically reports AMD64
        return platform_name, "amd64"
    else:
        raise PlatformNotSupportedError(f"Unsupported operating system: {system}")


def get_github_token() -> str | None:
    """Resolve GitHub authentication token.

    Checks the following sources in order:
    1. GITHUB_TOKEN environment variable
    2. .github_token file in repository root
    3. ~/.github_token file

    Returns:
        str | None: The GitHub token if found, None otherwise.
    """
    # Check environment variable first
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        logger.debug("Using GITHUB_TOKEN from environment variable")
        return token

    # Check repository root .github_token file
    repo_token_file = Path(__file__).parent.parent / ".github_token"
    if repo_token_file.exists():
        logger.debug(f"Using GitHub token from {repo_token_file}")
        return repo_token_file.read_text().strip()

    # Check ~/.github_token file
    home_token_file = Path.home() / ".github_token"
    if home_token_file.exists():
        logger.debug(f"Using GitHub token from {home_token_file}")
        return home_token_file.read_text().strip()

    logger.warning(
        "No GitHub token found. API requests will be rate-limited to 60/hour."
    )
    logger.warning(
        "Set GITHUB_TOKEN environment variable or create .github_token file "
        "in repo root or ~/.github_token for 5000 requests/hour."
    )
    return None


def get_artifact_name(platform_name: str, architecture: str) -> str:
    """Build the artifact name from platform and architecture.

    Args:
        platform_name: The platform name (e.g., 'macosx', 'manylinux', 'win')
        architecture: The architecture (e.g., 'arm64', 'x86_64', 'amd64')

    Returns:
        str: The artifact name (e.g., 'wheel-macosx-arm64')
    """
    return f"wheel-{platform_name}-{architecture}"


def fetch_latest_workflow_run(gh_client: Github) -> WorkflowRun:
    """Fetch the latest successful workflow run.

    Args:
        gh_client: Authenticated GitHub client instance

    Returns:
        WorkflowRun: The most recent successful workflow run

    Raises:
        ArtifactNotFoundError: If no successful workflow run is found
        GithubException: If there's an error accessing the GitHub API
    """
    logger.info("Fetching latest workflow run from eval-hub/eval-hub repository...")

    try:
        repo = gh_client.get_repo("eval-hub/eval-hub")
        workflow = repo.get_workflow("publish-python-server.yml")

        # Get workflow runs, filtered by success status
        runs = workflow.get_runs(status="success", branch="main")

        # Get the first (most recent) run
        try:
            latest_run = runs[0]
            logger.info(
                f"Found workflow run #{latest_run.run_number} from {latest_run.created_at}"
            )
            return latest_run
        except IndexError:
            raise ArtifactNotFoundError(
                "No successful workflow runs found for publish-python-server.yml"
            )

    except GithubException as e:
        if e.status == 404:
            raise ArtifactNotFoundError(
                "Repository or workflow not found. "
                "Make sure you have access to eval-hub/eval-hub."
            ) from e
        raise


def download_artifact(
    token: str | None, run: WorkflowRun, artifact_name: str, download_dir: Path
) -> Path:
    """Download and extract the artifact zip file.

    Args:
        token: GitHub authentication token (required for artifact downloads)
        run: The workflow run containing the artifact
        artifact_name: Name of the artifact to download
        download_dir: Directory to download and extract the artifact

    Returns:
        Path: Path to the extracted .whl file

    Raises:
        ArtifactNotFoundError: If the artifact is not found in the workflow run
    """
    import httpx

    logger.info(f"Looking for artifact: {artifact_name}")

    # Find the artifact
    artifacts = run.get_artifacts()
    target_artifact = None

    for artifact in artifacts:
        if artifact.name == artifact_name:
            target_artifact = artifact
            break

    if target_artifact is None:
        # List available artifacts to help debugging
        available = [a.name for a in artifacts]
        raise ArtifactNotFoundError(
            f"Artifact '{artifact_name}' not found in workflow run.\n"
            f"Available artifacts: {', '.join(available)}"
        )

    logger.info(
        f"Downloading artifact (size: {target_artifact.size_in_bytes} bytes)..."
    )

    # Download the artifact
    zip_path = download_dir / f"{artifact_name}.zip"

    # GitHub's artifact download requires authentication
    if token is None:
        raise ValueError(
            "GitHub token is required for downloading artifacts. "
            "Please set GITHUB_TOKEN env or .github_token file."
        )

    headers = {"Authorization": f"token {token}"}

    # Use httpx to download the artifact
    with httpx.Client(follow_redirects=True) as client:
        response = client.get(target_artifact.archive_download_url, headers=headers)
        response.raise_for_status()

        # Save the zip file
        zip_path.write_bytes(response.content)

    logger.info(f"Downloaded to {zip_path}")

    # Extract the zip file
    logger.info("Extracting artifact...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    # Find the .whl file
    whl_files = list(download_dir.glob("*.whl"))
    if not whl_files:
        raise ArtifactNotFoundError(f"No .whl file found in artifact {artifact_name}")

    whl_path = whl_files[0]
    logger.info(f"Found wheel: {whl_path.name}")
    return whl_path


def install_wheel(wheel_path: Path) -> None:
    """Install the wheel using UV package manager.

    Args:
        wheel_path: Path to the .whl file to install

    Raises:
        FileNotFoundError: If UV is not found in PATH
        subprocess.CalledProcessError: If UV installation fails
    """
    # Check if UV is installed
    uv_path = shutil.which("uv")
    if uv_path is None:
        raise FileNotFoundError(
            "UV package manager not found in PATH. "
            "Please install UV: https://github.com/astral-sh/uv"
        )

    logger.info(f"Installing wheel using UV: {wheel_path.name}")

    # Run UV pip install
    try:
        result = subprocess.run(
            ["uv", "pip", "install", "--force-reinstall", str(wheel_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug(result.stdout)
        logger.info("Installation completed successfully!")

    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed with exit code {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise


def main() -> int:
    try:
        platform_name, architecture = detect_platform()
        logger.info(f"Detected platform: {platform_name}-{architecture}")

        artifact_name = get_artifact_name(platform_name, architecture)
        logger.info(f"Target artifact: {artifact_name}")

        token = get_github_token()
        gh_client = Github(token)

        latest_run = fetch_latest_workflow_run(gh_client)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wheel_path = download_artifact(token, latest_run, artifact_name, temp_path)

            install_wheel(wheel_path)

        logger.info("eval-hub-server has been successfully installed!")
        return 0

    except PlatformNotSupportedError as e:
        logger.error(f"Platform not supported: {e}")
        return 1

    except ArtifactNotFoundError as e:
        logger.error(f"Artifact not found: {e}")
        return 1

    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        if e.status == 403:
            logger.error(
                "This may be due to rate limiting. "
                "Please set GITHUB_TOKEN environment variable."
            )
        return 1

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    except subprocess.CalledProcessError:
        logger.error("Installation failed. Please check the error messages above.")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
