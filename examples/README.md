# How to run the examples

## Watching logs

**File:** [watch_job_logs.py](watch_job_logs.py)

The script streams new log lines to stdout and prints the final job state to
stderr when the job reaches a terminal state.

### Exit codes

| Code | Meaning |
| ---- | ------- |
| `0` | Job reached a terminal state (completed, cancelled, partially_failed) |
| `1` | Job finished `failed` |
| `2` | Watch ended before completion (timeout or non-terminal last state) |
| `3` | HTTP/request error |
| `130` | Interrupted with Ctrl+C |

### 1. Configure connection (optional, reusable)

**Default profile** (used when `--profile` is omitted):

```shell
evalhub config set base_url https://evalhub.apps.my-cluster.example.com
evalhub config set token "$(oc whoami -t)"
evalhub config set tenant my-team
```

**Named profile** (use with `--profile prod`):

```shell
evalhub config set base_url https://evalhub.apps.my-cluster.example.com --profile prod
evalhub config set token "$(oc whoami -t)" --profile prod
evalhub config set tenant my-team --profile prod
```

### 2. Get a job ID

Submit a job or list existing jobs:

```shell
uv run evalhub eval run --config eval.yaml
# → Job submitted: 86c904cc-cdcf-452d-b9d8-754a7d6391ec

uv run evalhub eval status --status running
```

### 3. Watch logs

**Default profile** (reads `~/.config/evalhub/config.yaml`):

```shell
cd eval-hub-sdk
uv run python examples/watch_job_logs.py 86c904cc-cdcf-452d-b9d8-754a7d6391ec
```

**Named profile:**

```shell
uv run python examples/watch_job_logs.py 86c904cc-cdcf-452d-b9d8-754a7d6391ec --profile prod
```

**Explicit remote settings** (no profile file required):

```shell
export EVALHUB_BASE_URL=https://evalhub.apps.my-cluster.example.com
export EVALHUB_TOKEN="$(oc whoami -t)"
export EVALHUB_TENANT=my-team

uv run python examples/watch_job_logs.py 86c904cc-cdcf-452d-b9d8-754a7d6391ec
```

Or pass flags directly:

```shell
uv run python examples/watch_job_logs.py 86c904cc-cdcf-452d-b9d8-754a7d6391ec \
  --base-url https://evalhub.apps.my-cluster.example.com \
  --token "$(oc whoami -t)" \
  --tenant my-team
```

**Inside a cluster** (service account token):

```shell
uv run python examples/watch_job_logs.py 86c904cc-cdcf-452d-b9d8-754a7d6391ec \
  --base-url https://evalhub.evalhub.svc:8080 \
  --token-file /var/run/secrets/kubernetes.io/serviceaccount/token \
  --ca-bundle /var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
  --tenant default
```

### Useful flags

```shell
uv run python examples/watch_job_logs.py --help
```

Common options:

- `--benchmark-index 0` — watch a single benchmark only
- `--tail-lines 500` — max lines returned per poll (default: 1000)
- `--poll-interval 3` — seconds between polls (default: 2.0)
- `--timeout 3600` — stop watching after one hour
- `--timestamps` — include Kubernetes log timestamps when supported
