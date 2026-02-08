"""Example usage of the EvalHub client SDK.

The SDK provides separate client classes for async and sync operations,
with a nested resource structure.
"""

import asyncio

from evalhub import AsyncEvalHubClient, ModelConfig, SyncEvalHubClient
from evalhub.models.api import BenchmarkConfig, JobSubmissionRequest

# Example 1: Synchronous client usage with nested resources
print("=" * 60)
print("Example 1: Synchronous Client Usage (Nested Resources)")
print("=" * 60)

# Create synchronous client (defaults to http://localhost:8080)
with SyncEvalHubClient() as client:  # type: SyncEvalHubClient
    # Check health
    try:
        health = client.health()
        print(f"✓ EvalHub is healthy: {health}")
    except Exception as e:
        print(f"✗ Failed to connect to local EvalHub: {e}")

    # List available providers using nested resource
    try:
        providers = client.providers.list()
        print(f"\n✓ Found {len(providers)} providers")
        for provider in providers[:3]:  # Show first 3
            print(f"  - {provider.id}: {provider.name}")
    except Exception as e:
        print(f"✗ Failed to list providers: {e}")

    # List available benchmarks using nested resource
    try:
        benchmarks = client.benchmarks.list(category="math")
        print(f"\n✓ Found {len(benchmarks)} math benchmarks")
        for benchmark in benchmarks[:3]:  # Show first 3
            print(f"  - {benchmark.id}: {benchmark.name}")
    except Exception as e:
        print(f"✗ Failed to list benchmarks: {e}")


# Example 2: Connect to remote EvalHub instance
print("\n" + "=" * 60)
print("Example 2: Remote EvalHub Connection")
print("=" * 60)

# Create client for remote instance with authentication
with SyncEvalHubClient(
    base_url="https://evalhub.example.com",
    auth_token="your-api-token-here",
    timeout=60.0,
) as remote_client:
    print("✓ Remote client created (connection would be tested on first API call)")


# Example 3: Submit an evaluation job (synchronous with nested resources)
print("\n" + "=" * 60)
print("Example 3: Submit Evaluation Job (Synchronous)")
print("=" * 60)

with SyncEvalHubClient() as eval_client:  # type: SyncEvalHubClient
    # Create evaluation request
    # Using a vLLM endpoint deployed on OpenShift
    model = ModelConfig(
        url="http://vllm-service.my-namespace.svc.cluster.local:8000",
        name="meta-llama/Llama-2-7b-chat-hf",
    )

    benchmark_config: BenchmarkConfig = BenchmarkConfig(
        id="gsm8k",
        provider_id="lm_evaluation_harness",
        parameters={"num_fewshot": 5},
    )

    request = JobSubmissionRequest(
        model=model,
        benchmarks=[benchmark_config],
        timeout_minutes=30,
    )

    try:
        # Submit job using nested resource
        job = eval_client.jobs.submit(request)
        print(f"✓ Job submitted: {job.id}")
        print(f"  Status: {job.status}")

        # Check status using nested resource
        updated_job = eval_client.jobs.get(job.id)
        print(f"✓ Job status updated: {updated_job.status}")

        # Wait for completion (polling)
        # final_job = eval_client.jobs.wait_for_completion(job.id, timeout=300)
        # if final_job.status == "completed":
        #     results = eval_client.jobs.results(job.id)
        #     print(f"✓ Results: {len(results.results)} metrics")

    except NotImplementedError:
        print("✗ Job submission not yet implemented (skeleton only)")
    except Exception as e:
        print(f"✗ Failed to submit job: {e}")


# Example 4: Using async client (recommended for I/O-bound workloads)
print("\n" + "=" * 60)
print("Example 4: Async Usage")
print("=" * 60)


async def async_example() -> None:
    """Demonstrate async client usage with nested resources.

    NOTE: Same method names as sync - just await them!
    """
    async with AsyncEvalHubClient() as client:  # type: AsyncEvalHubClient
        try:
            # Async health check
            health = await client.health()
            print(f"✓ Async health check: {health}")

            # Async provider list using nested resource
            providers = await client.providers.list()
            print(f"✓ Found {len(providers)} providers (async)")

            # Example: Submit evaluation job (commented out to avoid actual job creation)
            # Uncomment below to submit a real evaluation job:
            # request = EvaluationRequest(
            #     benchmark_id="mmlu",
            #     model=ModelConfig(
            #         url="http://vllm-service.my-namespace.svc.cluster.local:8000/v1",
            #         name="meta-llama/Llama-2-7b-chat-hf",
            #     ),
            # )
            # job = await client.jobs.submit(request)
            # print(f"✓ Async job submitted: {job.id}")
            #
            # # You can also wait for completion asynchronously
            # final_job = await client.jobs.wait_for_completion(job.id, timeout=300)
            # if final_job.status == "completed":
            #     results = await client.jobs.results(job.id)
            #     print(f"✓ Results: {len(results.results)} metrics")

        except NotImplementedError:
            print("✗ Some async operations not yet implemented (skeleton only)")
        except Exception as e:
            print(f"✗ Async operation failed: {e}")


# Run async example
try:
    asyncio.run(async_example())
except Exception as e:
    print(f"✗ Failed to run async example: {e}")


# Example 5: Client class comparison
print("\n" + "=" * 60)
print("Example 5: Client Class Comparison")
print("=" * 60)

print(
    """
Sync vs Async - Same nested structure!

Synchronous (SyncEvalHubClient):
    with SyncEvalHubClient() as client:
        providers = client.providers.list()       # No await needed
        benchmarks = client.benchmarks.list()     # No await needed
        job = client.jobs.submit(request)         # No await needed
        status = client.jobs.get(job_id)          # Results embedded

Asynchronous (AsyncEvalHubClient):
    async with AsyncEvalHubClient() as client:
        providers = await client.providers.list()       # Await needed
        benchmarks = await client.benchmarks.list()     # Await needed
        job = await client.jobs.submit(request)         # Await needed
        status = await client.jobs.get(job_id)          # Results embedded
"""
)


print("\n" + "=" * 60)
print("All examples completed!")
print("=" * 60)
