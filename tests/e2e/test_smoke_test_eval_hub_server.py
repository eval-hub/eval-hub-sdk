import pytest
from evalhub import (
    BenchmarkConfig,
    CollectionRef,
    JobSubmissionRequest,
    ModelConfig,
    SyncEvalHubClient,
)


@pytest.mark.e2e
def test_evaluations_providers_endpoint(evalhub_server_with_real_config: str) -> None:
    """Test that the evaluations providers endpoint is accessible."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        providers = client.providers.list()
        assert isinstance(providers, list)


@pytest.mark.e2e
def test_collections_endpoint(evalhub_server_with_real_config: str) -> None:
    """Test that the collections endpoint is accessible."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        collections = client.collections.list()
        assert isinstance(collections, list)


@pytest.mark.e2e
def test_jobs_endpoint(evalhub_server_with_real_config: str) -> None:
    """Test that the jobs endpoint is accessible and can submit jobs."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        # Test listing jobs
        jobs = client.jobs.list()
        assert isinstance(jobs, list)

        # Test submitting a job
        model = ModelConfig(url="http://test-model-server:8000", name="test-model")

        benchmark = BenchmarkConfig(
            id="toxicity", provider_id="garak", parameters={"myparam": 5}
        )

        job_request = JobSubmissionRequest(
            name="toxicity-eval",
            description="Toxicity evaluation run",
            tags=["safety", "toxicity"],
            model=model,
            benchmarks=[benchmark],
        )

        # Submit job and verify we get a response without errors
        job = client.jobs.submit(job_request)

        # Verify the response has expected fields
        assert job is not None
        assert hasattr(job, "id")
        assert hasattr(job, "resource")
        assert job.id is not None


@pytest.mark.e2e
def test_jobs_endpoint_with_collection(evalhub_server_with_real_config: str) -> None:
    """Test submitting a job using a collection reference."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        # Get a collection to reference
        collections = client.collections.list()
        if not collections:
            pytest.skip("No collections available on server")

        collection = collections[0]
        model = ModelConfig(url="http://test-model-server:8000", name="test-model")

        job_request = JobSubmissionRequest(
            name="collection-eval",
            description="Evaluation via collection",
            tags=["collection-test"],
            model=model,
            collection=CollectionRef(id=collection.resource.id),
        )

        # Submit job and verify we get a response without errors
        job = client.jobs.submit(job_request)

        # Verify the response has expected fields
        assert job is not None
        assert hasattr(job, "id")
        assert hasattr(job, "resource")
        assert job.id is not None


@pytest.mark.e2e
def test_health_endpoint(evalhub_server_with_real_config: str) -> None:
    """Test that the health endpoint is accessible."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        health = client.health()
        assert health is not None
