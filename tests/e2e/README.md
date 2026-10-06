E2E tests exercise EvalHub server, adapter, and client behavior against real
services.

Run the E2E suite, including the MLflow artifact round-trip test:

```bash
make test-e2e
```

MLflow 3.10.* is pinned in the dev dependency group in `pyproject.toml` and is
installed by `uv sync` or synchronized by `uv run pytest`. MLflow remains a dev
dependency; the default ODH backend does not require it at runtime.

The `mlflow_server` fixture starts MLflow's default server using the same Python
interpreter as pytest. It selects an unused port on `127.0.0.1`, uses a temporary
SQLite database and artifact directory, waits for `/health`, and stops the
server process group afterward, including on test failure. Existing servers at
port 5000 are left running. Deployment `MLFLOW_*` settings are excluded from the
local server and client configuration.

File uploads send `Content-Length` from the opened file's size and retain
bounded 64 KiB reads. This also works with MLflow 3.10's default Uvicorn/WSGI
server, without selecting Gunicorn.

The tests upload a file spanning multiple 64 KiB chunks and an empty file, then
download both and verify their contents. They also call
`DefaultCallbacks.mlflow.save()` with both ODH and upstream backends against the
real server. The upstream case uses the installed official MLflow library,
without mocking its APIs. Both cases verify mixed bytes/file artifacts, matching
and renamed destinations, metrics, params, tags, a single finished run, and its
returned ID. Source files remain intact.

These tests run as part of the existing CI E2E job. The MLflow tests themselves
need no kind cluster or OCI registry.
Direct pytest invocation skips the test if
MLflow is not installed.

Other tests require the `eval-hub-server` binary and a local OCI registry at
`localhost:5001`; start the registry with `make start-oci-registry`.
The EvalHub fixture uses configuration from `tests/e2e/config/` and starts the
server at `localhost:8080`.
