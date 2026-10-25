"""Regression tests: TLS trust for a correctly-signed EvalHub endpoint.

The SDK's CA auto-detection used to prefer the Kubernetes ServiceAccount CA
(``/var/run/secrets/kubernetes.io/serviceaccount/ca.crt``) over the system
trust store. That file exists in *every* pod and holds the kube-apiserver CA,
which can never validate a publicly-trusted endpoint such as a Let's
Encrypt-signed Route. Because it always exists, the ``verify=True`` (system
trust store) branch was unreachable inside a pod, so ``evalhub health`` failed
with::

    [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    unable to get issuer certificate

These tests pin the contract:

* auto-detection must not treat the ServiceAccount CA as a general-purpose
  HTTPS trust anchor, and
* an explicitly configured ``ca_bundle_path`` still wins over it, and
* the CLI, running in a pod-shaped environment, must verify a certificate
  issued by a CA the system trust store trusts.

The end-to-end tests run the real ``evalhub`` console script in a subprocess
against a loopback TLS server. The pod is simulated on two axes:

* ``SSL_CERT_FILE`` points at the test CA to stand in for a system trust store
  that trusts the endpoint (OpenSSL reads it only at process start, hence the
  subprocess), and
* the CLI runs in a private mount namespace (``unshare -rm``) where the
  ServiceAccount CA really exists at its canonical path, holding a CA that
  does *not* issue the server certificate — exactly what a pod sees.

The mount namespace is what makes the pod case faithful: the SDK hands the
path to OpenSSL, so the file has to be genuinely readable, not merely
reported as existing. The negative control (no planted CA) proves the TLS
fixture itself is sound, so a failure in the pod-shaped case is attributable
to CA selection rather than certificate generation, server wiring or config
handling.
"""

from __future__ import annotations

import functools
import os
import shutil
import ssl
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: CA files the SDK's auto-detection probes. On a real pod the first is usually
#: absent and the second always present.
SERVICEACCOUNT_CA = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
OPENSHIFT_SERVICE_CA = Path("/etc/pki/ca-trust/source/anchors/service-ca.crt")

#: Long enough for the client's 3 retries plus backoff on a verify failure.
CLI_TIMEOUT_SECONDS = 120

#: Simulate the pod: hide the OpenShift service-ca anchor (a pod either has it
#: injected or not, and this test is about the ServiceAccount CA), then project
#: a real CA bundle at the ServiceAccount CA path, as the kubelet does.
POD_MOUNT_SHIM = """\
set -e
mount --make-rprivate / 2>/dev/null || true
if [ -d /etc/pki/ca-trust/source/anchors ]; then
    mount -t tmpfs tmpfs /etc/pki/ca-trust/source/anchors
fi
mount -t tmpfs tmpfs /run
mkdir -p "$(dirname "$SA_CA_TARGET")"
cp "$SA_CA_SOURCE" "$SA_CA_TARGET"
exec "$@"
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_exists(monkeypatch: pytest.MonkeyPatch, present: set[Path]) -> None:
    """Report exactly *present* of the auto-detected CA paths as existing.

    Both probed paths are pinned rather than only the one under test, so the
    result does not depend on what the host running the suite happens to have
    in ``/etc/pki`` or ``/var/run``.
    """
    probed = {str(SERVICEACCOUNT_CA), str(OPENSHIFT_SERVICE_CA)}
    real_exists = Path.exists

    def exists(self: Path, *args: object, **kwargs: object) -> bool:
        if str(self) in probed:
            return self in present
        return bool(real_exists(self, *args, **kwargs))

    monkeypatch.setattr(Path, "exists", exists)


def _openssl(*args: str) -> None:
    result = subprocess.run(["openssl", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"openssl {args[0]} failed ({result.returncode}): {result.stderr.strip()}"
        )


@pytest.fixture(scope="module")
def tls_pki(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """CA plus a server certificate for ``localhost`` signed by it.

    Also builds an unrelated CA, used as the CA bundle projected into the pod:
    a real bundle that simply cannot validate this endpoint.
    """
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI is required to generate the test PKI")

    workdir = tmp_path_factory.mktemp("cli-tls-pki")
    ca_key, ca_crt = workdir / "ca.key", workdir / "ca.crt"
    srv_key, srv_crt = workdir / "server.key", workdir / "server.crt"
    srv_csr, srv_ext = workdir / "server.csr", workdir / "server.ext"
    other_ca_key, other_ca_crt = workdir / "other-ca.key", workdir / "other-ca.crt"

    for key, crt, subject in (
        (ca_key, ca_crt, "/CN=evalhub-sdk-cli-tls-test-ca"),
        (other_ca_key, other_ca_crt, "/CN=kubernetes-serviceaccount-ca"),
    ):
        _openssl(
            "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes", "-days", "2",
            "-keyout", str(key), "-out", str(crt),
            "-subj", subject,
            "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            "-addext", "subjectKeyIdentifier=hash",
        )  # fmt: skip
    _openssl(
        "req", "-new", "-newkey", "rsa:2048", "-sha256", "-nodes",
        "-keyout", str(srv_key), "-out", str(srv_csr), "-subj", "/CN=localhost",
    )  # fmt: skip
    srv_ext.write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        "subjectAltName=DNS:localhost,IP:127.0.0.1\n"
    )
    _openssl(
        "x509", "-req", "-in", str(srv_csr), "-sha256", "-days", "2",
        "-CA", str(ca_crt), "-CAkey", str(ca_key), "-CAcreateserial",
        "-out", str(srv_crt), "-extfile", str(srv_ext),
    )  # fmt: skip

    return {"ca": ca_crt, "cert": srv_crt, "key": srv_key, "wrong_ca": other_ca_crt}


class _HealthHandler(BaseHTTPRequestHandler):
    """Answers every GET with the EvalHub health payload."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = b'{"status": "healthy"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Keep the TLS handshake noise out of pytest output."""


@pytest.fixture()
def evalhub_tls_server(tls_pki: dict[str, Path]) -> Iterator[str]:
    """Serve a health endpoint over TLS with the test server certificate."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(tls_pki["cert"]), str(tls_pki["key"]))

    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://localhost:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture()
def cli_config(tmp_path: Path, evalhub_tls_server: str) -> Path:
    """A profile pointing at the TLS test server (no ``insecure`` key)."""
    config = tmp_path / "evalhub" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "active_profile: default\n"
        "profiles:\n"
        "  default:\n"
        f"    base_url: {evalhub_tls_server}\n"
        "    token: test-token\n"
        "    tenant: test-tenant\n"
    )
    return config


def _evalhub_command() -> list[str]:
    """Invoke the installed console script, or the entry point as a fallback."""
    script = Path(sys.executable).with_name("evalhub")
    if script.is_file():
        return [str(script)]
    return [sys.executable, "-c", "from evalhub.cli.bootstrap import main; main()"]


@functools.cache
def _unshare_supported() -> bool:
    """True when unprivileged user+mount namespaces are usable here."""
    if sys.platform != "linux" or shutil.which("unshare") is None:
        return False
    probe = subprocess.run(
        ["unshare", "-rm", "--propagation", "private", "true"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def _run_cli_health(
    config: Path, tls_pki: dict[str, Path], *, in_pod: bool
) -> subprocess.CompletedProcess[str]:
    """Run ``evalhub health`` with the test CA as the trusted root.

    ``in_pod=True`` additionally projects an unrelated CA bundle at the
    Kubernetes ServiceAccount CA path inside a private mount namespace.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {"EVALHUB_CONFIG", "SSL_CERT_FILE", "SA_CA_SOURCE", "SA_CA_TARGET"}
    }
    env["EVALHUB_CONFIG"] = str(config)
    env["SSL_CERT_FILE"] = str(tls_pki["ca"])

    command = [*_evalhub_command(), "health"]
    if in_pod:
        if not _unshare_supported():
            pytest.skip(
                "unprivileged mount namespaces (unshare -rm) are required to "
                "project a real ServiceAccount CA; the CA-selection unit test "
                "still covers the regression"
            )
        env["SA_CA_SOURCE"] = str(tls_pki["wrong_ca"])
        env["SA_CA_TARGET"] = str(SERVICEACCOUNT_CA)
        command = [
            "unshare", "-rm", "--propagation", "private",
            "sh", "-c", POD_MOUNT_SHIM, "sh", *command,
        ]  # fmt: skip

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        timeout=CLI_TIMEOUT_SECONDS,
        check=False,
    )


# ---------------------------------------------------------------------------
# CA selection
# ---------------------------------------------------------------------------


def test_auto_detection_ignores_serviceaccount_ca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pod's projected ServiceAccount CA must not become the trust anchor.

    Returning it makes ``verify=True`` unreachable, so every correctly-signed
    endpoint fails — the reported bug.
    """
    from evalhub.client.base import _resolve_ca_bundle

    _fake_exists(monkeypatch, {SERVICEACCOUNT_CA})

    resolved = _resolve_ca_bundle(None)

    assert resolved is None, (
        "CA auto-detection selected "
        f"{resolved}, which is the Kubernetes API server CA projected into "
        "every pod. It cannot validate a publicly-trusted endpoint and it "
        "shadows the system trust store; auto-detection "
        "must fall through to the system trust store instead."
    )


def test_explicit_ca_bundle_path_is_honoured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit ``ca_bundle_path`` still wins over auto-detection."""
    from evalhub.client.base import _resolve_ca_bundle

    explicit = tmp_path / "custom-ca.crt"
    explicit.write_text("explicit-ca")
    _fake_exists(monkeypatch, {SERVICEACCOUNT_CA})

    assert _resolve_ca_bundle(explicit) == explicit


# ---------------------------------------------------------------------------
# CLI against a correctly-signed endpoint
# ---------------------------------------------------------------------------


def test_cli_health_trusts_publicly_signed_endpoint_outside_pod(
    cli_config: Path,
    tls_pki: dict[str, Path],
) -> None:
    """Control: without a projected ServiceAccount CA the CLI verifies fine.

    This proves the fixture is sound — certificate generation, TLS server,
    profile handling and ``SSL_CERT_FILE`` all work — so the pod-shaped test
    below isolates CA selection as the cause of any failure.
    """
    result = _run_cli_health(cli_config, tls_pki, in_pod=False)

    assert result.returncode == 0, (
        "evalhub health failed without any pod CA present, so the TLS test "
        f"fixture itself is broken.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "healthy" in result.stdout


def test_cli_health_trusts_publicly_signed_endpoint_inside_pod(
    cli_config: Path,
    tls_pki: dict[str, Path],
) -> None:
    """The reported bug: a pod-shaped env must reach the system trust store."""
    result = _run_cli_health(cli_config, tls_pki, in_pod=True)
    output = result.stdout + result.stderr

    assert result.returncode == 0, (
        "evalhub health could not verify a certificate issued by the CA the "
        "system trust store trusts, while the pod's ServiceAccount CA was "
        "present. The SDK used "
        f"{SERVICEACCOUNT_CA} as its trust anchor instead of the system "
        "store.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "CERTIFICATE_VERIFY_FAILED" not in output
    assert "healthy" in result.stdout
