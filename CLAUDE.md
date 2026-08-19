# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EvalHub SDK is a Python SDK providing three components: **common models** (Pydantic v2 data models, always available), a **REST API client** (async/sync), and a **framework adapter SDK** for building custom evaluation adapters using a "Bring Your Own Framework" (BYOF) pattern. It also ships a Click-based CLI that manages the standalone Go-based MCP binary (`evalhub-mcp`).

## Commands

```bash
# Install with all dev dependencies
uv sync --all-extras

# Run all unit tests with coverage
make test

# Run a single test file
uv run pytest tests/unit/test_models.py -v

# Run a single test by name
uv run pytest tests/unit/test_models.py -k "test_job_status" -v

# Lint and format
make ruff

# Type checking
make mypy

# Both lint + type check
make tidy

# Pre-commit hooks (ruff, mypy, trailing whitespace, conventional commits)
make pre-commit

# E2E tests (requires kind cluster + eval-hub-server)
make test-e2e
```

## Architecture

### Package structure (`src/evalhub/`)

- **`models/`** — Pydantic v2 request/response models (JobStatus, EvaluationStatus, BenchmarkConfig, etc.). Always importable with zero optional deps.
- **`adapter/`** — SDK for framework developers. `FrameworkAdapter` is the abstract base class; implementers override `run_benchmark_job(config, callbacks) -> JobResults`. Includes `DefaultCallbacks`, `AdapterSettings` (env-based config), and `OCIArtifactPersister`.
- **`client/`** — `AsyncEvalHubClient` and `SyncEvalHubClient` with nested resource classes (providers, benchmarks, collections, jobs). Job resources support `submit`, `get`, `list`, `cancel`, `wait_for_completion`, `get_logs`, and `watch_logs`. Log helpers live in `client/job_logs.py` (`JobLogOptions`, `JobLogUpdate`). Requires `httpx`.
- **`cli/`** — Click command groups (`eval`, `provider`, `benchmark`, `collection`, `config`, `health`, `mcp`). Entry point: `evalhub.cli.bootstrap:main`. The `mcp` subcommand manages the standalone Go binary (`evalhub-mcp`).

### Adapter job lifecycle (Kubernetes)

1. Service creates a Kubernetes Job: adapter container + sidecar
2. ConfigMap mounts `JobSpec` at `/meta/job.json`
3. Adapter loads `JobSpec` on init, calls `run_benchmark_job()`
4. Status updates go via callbacks → localhost sidecar → EvalHub service
5. Results persisted as OCI artifacts; pod terminates

### Optional dependency extras

Components are gated behind pip extras: `core`, `adapter`, `client`, `cli`, `dev`, `all`. The `mcp` extra is a placeholder for the `evalhub-mcp` Go binary PyPI package. The top-level `__init__.py` uses conditional imports — client classes only appear when `httpx` is installed.

## Code Style

- **Formatter/linter**: ruff (v0.1.6, pinned). Line length 88. Rules: E, W, F, I, N, UP. E501 ignored.
- **Type checking**: mypy in strict mode (Python 3.11 target).
- **Commits**: conventional commits enforced by commitizen/commitlint.
- **Python**: >=3.11. CI tests on 3.12, 3.13, 3.14.

## Test Markers

Tests use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.adapter`, `@pytest.mark.e2e`. The default `make test` runs unit tests only.
