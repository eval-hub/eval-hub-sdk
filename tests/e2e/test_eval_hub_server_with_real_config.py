"""E2E tests for eval-hub server using real production config."""

import pytest
from evalhub import SyncEvalHubClient
from httpx import HTTPStatusError


@pytest.mark.e2e
def test_server_starts_with_real_config(evalhub_server_with_real_config: str) -> None:
    """Verify server can start successfully with real production config."""
    # If fixture yields successfully, server started
    assert evalhub_server_with_real_config == "http://localhost:8080"


@pytest.mark.e2e
def test_providers_endpoint_with_real_config(
    evalhub_server_with_real_config: str,
) -> None:
    """Verify that the providers endpoint is accessible with real config."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        providers = client.providers.list()
        print(f"\n\n===== PROVIDERS COUNT: {len(providers)} =====")
        for p in providers:
            print(f"  - {p.provider_id}: {p.provider_name}")
        print("=" * 50)
        assert isinstance(providers, list)
        assert len(providers) > 0, f"Expected providers to be loaded from config, but got {len(providers)}"


@pytest.mark.e2e
def test_health_endpoint_with_real_config(evalhub_server_with_real_config: str) -> None:
    """Verify health endpoint works with real config."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        health = client.health()
        assert health is not None


@pytest.mark.e2e
def test_api_endpoints_with_real_config(evalhub_server_with_real_config: str) -> None:
    """Verify basic API endpoints work with real config."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        # Test multiple endpoints to verify real config doesn't break anything
        providers = client.providers.list()
        assert isinstance(providers, list)

        jobs = client.jobs.list()
        assert isinstance(jobs, list)

        # Collections still returns 501
        with pytest.raises(HTTPStatusError) as exc_info:
            client.collections.list()
        assert exc_info.value.response.status_code == 501
