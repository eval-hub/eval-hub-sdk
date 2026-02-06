"""E2E tests for eval-hub server using real production config."""

from pathlib import Path

import pytest
import yaml
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
    """Verify that providers and benchmarks match the YAML configuration files."""
    # Load provider YAML files
    config_dir = Path(__file__).parent / "config" / "providers"
    provider_yamls = {}
    for yaml_file in config_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            provider_yamls[data["provider_id"]] = data

    # Get providers from server
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        providers = client.providers.list()
        print(f"\n\n===== PROVIDERS COUNT: {len(providers)} =====")
        for p in providers:
            print(f"  - {p.id}: {p.label}")
        print("=" * 50)

        assert isinstance(providers, list)
        assert (
            len(providers) > 0
        ), f"Expected providers to be loaded from config, but got {len(providers)}"

        # Verify each provider matches the YAML configuration
        for provider in providers:
            assert (
                provider.id in provider_yamls
            ), f"Provider {provider.id} not found in YAML configs"
            yaml_data = provider_yamls[provider.id]

            # Check provider fields
            assert provider.label == yaml_data["provider_name"], (
                f"Provider {provider.id}: label mismatch. "
                f"Expected '{yaml_data['provider_name']}', got '{provider.label}'"
            )

            # Get benchmarks from provider object
            provider_benchmarks = provider.benchmarks
            yaml_benchmarks = yaml_data.get("benchmarks", [])

            print(f"\n  Provider {provider.id}:")
            print(f"    Provider has {len(provider_benchmarks)} benchmarks")
            print(f"    YAML defines {len(yaml_benchmarks)} benchmarks")

            # Verify benchmark count matches exactly
            assert len(provider_benchmarks) == len(yaml_benchmarks), (
                f"Provider {provider.id}: benchmark count mismatch. "
                f"Expected {len(yaml_benchmarks)}, got {len(provider_benchmarks)}"
            )

            # Verify that all benchmarks defined in YAML are present in server response
            yaml_benchmark_ids = {b["benchmark_id"] for b in yaml_benchmarks}
            provider_benchmark_ids = {b.benchmark_id for b in provider_benchmarks}

            missing_benchmarks = yaml_benchmark_ids - provider_benchmark_ids
            assert not missing_benchmarks, (
                f"Provider {provider.id}: benchmarks defined in YAML are missing from server: "
                f"{missing_benchmarks}"
            )

            print(
                f"    ✓ Benchmark count matches YAML: {len(provider_benchmarks)} benchmarks"
            )
            print("    ✓ All YAML-defined benchmarks found in server response")

            # Verify the first benchmark in detail
            if yaml_benchmarks:
                yaml_first = yaml_benchmarks[0]
                # Find the corresponding benchmark from server
                server_first = next(
                    (
                        b
                        for b in provider_benchmarks
                        if b.benchmark_id == yaml_first["benchmark_id"]
                    ),
                    None,
                )

                assert server_first is not None, (
                    f"Provider {provider.id}: first benchmark '{yaml_first['benchmark_id']}' "
                    "not found in server response"
                )

                # Verify all fields of the first benchmark
                assert server_first.benchmark_id == yaml_first["benchmark_id"]
                assert server_first.name == yaml_first["name"]
                assert server_first.description == yaml_first["description"]
                assert server_first.category == yaml_first["category"]
                assert server_first.metrics == yaml_first["metrics"]
                assert server_first.default_few_shot == yaml_first["num_few_shot"]
                yaml_dataset_size = yaml_first.get(
                    "dataset_size"
                )  # Handle dataset_size: null in YAML becomes 0 or None in server
                if yaml_dataset_size is None:
                    assert server_first.dataset_size in (None, 0), (
                        f"Expected dataset_size to be None or 0 for null YAML value, "
                        f"got {server_first.dataset_size}"
                    )
                else:
                    assert server_first.dataset_size == yaml_dataset_size
                assert server_first.tags == yaml_first.get("tags", [])

                print(
                    f"  ✓ First benchmark '{server_first.benchmark_id}' content verified against YAML"
                )


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
