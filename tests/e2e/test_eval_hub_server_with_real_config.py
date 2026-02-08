"""E2E tests for eval-hub server using real production config."""

from pathlib import Path

import pytest
import yaml
from evalhub import SyncEvalHubClient


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
    # Gather test data from provider YAML files
    config_dir = Path(__file__).parent / "config" / "providers"
    provider_yamls = {}
    for yaml_file in config_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            provider_id = data["id"]
            assert (
                provider_id not in provider_yamls
            ), f"Duplicate provider.id '{provider_id}' in {yaml_file}"
            provider_yamls[provider_id] = data

    # Get providers from actual server
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        providers = client.providers.list()
        print(f"\n\n===== PROVIDERS COUNT: {len(providers)} =====")
        for p in providers:
            print(f"  - {p.id}: {p.name}")
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
            assert provider.name == yaml_data["name"], (
                f"Provider {provider.id}: label mismatch. "
                f"Expected '{yaml_data['name']}', got '{provider.name}'"
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
            yaml_benchmark_ids = {b["id"] for b in yaml_benchmarks}
            provider_benchmark_ids = {b.id for b in provider_benchmarks}

            missing_benchmarks = yaml_benchmark_ids - provider_benchmark_ids
            assert not missing_benchmarks, (
                f"Provider {provider.id}: benchmarks defined in YAML are missing from server: "
                f"{missing_benchmarks}"
            )

            print(
                f"    ✓ Benchmark count matches YAML: {len(provider_benchmarks)} benchmarks"
            )
            print("    ✓ All YAML-defined benchmarks found in server response")

            # Verify all benchmarks in detail
            assert (
                yaml_benchmarks
            ), f"Provider {provider.id}: expected benchmarks in YAML config"

            for yaml_benchmark in yaml_benchmarks:
                # Find the corresponding benchmark from server
                server_benchmark = next(
                    (b for b in provider_benchmarks if b.id == yaml_benchmark["id"]),
                    None,
                )

                assert server_benchmark is not None, (
                    f"Provider {provider.id}: benchmark '{yaml_benchmark['id']}' "
                    "not found in server response"
                )

                # Verify all fields of the benchmark
                assert server_benchmark.id == yaml_benchmark["id"]
                assert server_benchmark.name == yaml_benchmark["name"]
                assert server_benchmark.description == yaml_benchmark["description"]
                assert server_benchmark.category == yaml_benchmark["category"]
                assert server_benchmark.metrics == yaml_benchmark["metrics"]
                yaml_num_few_shot = yaml_benchmark.get(
                    "num_few_shot", 0
                )  # Handle num_few_shot: null or missing in YAML, becomes 0 in server
                assert server_benchmark.num_few_shot == yaml_num_few_shot
                yaml_dataset_size = yaml_benchmark.get(
                    "dataset_size"
                )  # Handle dataset_size: null in YAML becomes 0 or None in server
                if yaml_dataset_size is None:
                    assert server_benchmark.dataset_size in (None, 0), (
                        f"Expected dataset_size to be None or 0 for null YAML value, "
                        f"got {server_benchmark.dataset_size}"
                    )
                else:
                    assert server_benchmark.dataset_size == yaml_dataset_size
                assert server_benchmark.tags == yaml_benchmark.get("tags", [])

                print(
                    f"    ✓ Benchmark '{server_benchmark.id}' content verified against YAML"
                )


@pytest.mark.e2e
def test_health_endpoint_with_real_config(evalhub_server_with_real_config: str) -> None:
    """Verify health endpoint works with real config."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        health = client.health()
        assert health is not None
