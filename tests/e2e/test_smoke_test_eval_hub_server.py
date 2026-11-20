import pytest
from evalhub import (
    AsyncEvalHubClient,
    BenchmarkConfig,
    CollectionRef,
    JobSubmissionRequest,
    ModelConfig,
    SyncEvalHubClient,
)
from evalhub.models import StandalonePostProcessingRequest


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


@pytest.fixture
def post_processing_request() -> StandalonePostProcessingRequest:
    """Use external data so submission does not require a completed evaluation."""
    return StandalonePostProcessingRequest.model_validate(
        {
            "name": "accuracy-ci-smoke",
            "operations": {
                "confidence_interval": {
                    "results_data_ref": {"pvc": {"claim_name": "results-data"}},
                    "primary_score": {"metric": "accuracy"},
                    "calibration_data_ref": [
                        {
                            "pvc": {"claim_name": "calibration-data"},
                            "data_config": {
                                "format": "jsonl",
                                "columns": {
                                    "label": "expected",
                                    "prediction": "actual",
                                },
                            },
                        }
                    ],
                    "significance_level": 0.05,
                }
            },
        }
    )


@pytest.mark.skip(reason="Requires eval-hub-server >=1.0.4; pending PyPI release")
@pytest.mark.e2e
def test_post_processing_endpoint(
    evalhub_server_with_real_config: str,
    post_processing_request: StandalonePostProcessingRequest,
) -> None:
    """Test submission without requiring a working post-processing adapter."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        # Execution may fail asynchronously when the local adapter is unavailable.
        # Like the jobs smoke tests, only verify acceptance, not completion.
        job = client.post_processing.submit(post_processing_request)

        assert job.resource.id
        assert job.name == post_processing_request.name
        assert job.operations == post_processing_request.operations


@pytest.mark.skip(reason="Requires eval-hub-server >=1.0.4; pending PyPI release")
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_async_post_processing_endpoint(
    evalhub_server_with_real_config: str,
    post_processing_request: StandalonePostProcessingRequest,
) -> None:
    """Test async submission without waiting for adapter execution."""
    async with AsyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        job = await client.post_processing.submit(post_processing_request)

        assert job.resource.id
        assert job.name == post_processing_request.name
        assert job.operations == post_processing_request.operations


@pytest.mark.e2e
def test_health_endpoint(evalhub_server_with_real_config: str) -> None:
    """Test that the health endpoint is accessible."""
    with SyncEvalHubClient(base_url=evalhub_server_with_real_config) as client:
        health = client.health()
        assert health is not None
