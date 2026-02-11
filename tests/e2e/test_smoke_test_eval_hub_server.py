import pytest
from evalhub import (
    BenchmarkConfig,
    JobSubmissionRequest,
    ModelConfig,
    SyncEvalHubClient,
)
from httpx import HTTPStatusError


@pytest.mark.e2e
def test_evaluations_providers_endpoint(evalhub_server_with_real_config: str) -> None:
    """Test that the evaluations providers endpoint is accessible."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        providers = client.providers.list()
        assert isinstance(providers, list)


@pytest.mark.e2e
def test_collections_endpoint(evalhub_server_with_real_config: str) -> None:
    """Test that the collections endpoint returns 501 Not Implemented."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        with pytest.raises(HTTPStatusError) as exc_info:
            client.collections.list()
        assert exc_info.value.response.status_code == 501


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
            id="test_benchmark", provider_id="test_provider", parameters={"myparam": 5}
        )

        job_request = JobSubmissionRequest(model=model, benchmarks=[benchmark])

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
