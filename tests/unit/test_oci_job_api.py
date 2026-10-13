"""OCI job contracts, including future bundle/signing output shapes."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import evalhub
import pytest
from evalhub import models
from evalhub.client import AsyncEvalHubClient, SyncEvalHubClient
from evalhub.client.job_logs import is_terminal_job
from evalhub.models import (
    EvaluationJob,
    EvaluationJobResults,
    EvaluationOCIResults,
    JobStatus,
    JobSubmissionRequest,
    OCIArtifactReference,
    OCIProcessingStatus,
    OCISigningConfig,
    OCISigningResult,
)
from pydantic import ValidationError

pytestmark = pytest.mark.unit

MESSAGE = {
    "message": "Server OCI outcome",
    "message_code": "SERVER_CODE",
    "message_origin": "server",
}
ARTIFACT = {
    "oci_digest": "sha256:actual-manifest",
    "oci_reference": "quay.io/org/results:evaluation-card-job-id@sha256:actual-manifest",
}


def request_data(signing: Any = None) -> dict[str, Any]:
    return {
        "name": "oci-test",
        "model": {"name": "test"},
        "benchmarks": [{"id": "test", "provider_id": "test"}],
        "exports": {
            "oci": {
                "coordinates": {
                    "oci_host": "quay.io",
                    "oci_repository": "org/results",
                    "oci_tag": "user-tag",
                },
                "signing": signing,
            }
        },
    }


def job_data(state: str = "completed") -> dict[str, Any]:
    return {
        **request_data({"type": "key-pair", "secret_ref": "signing-key"}),
        "resource": {"id": "job-id", "created_at": "2026-10-10T00:00:00Z"},
        "status": {"state": "completed"},
        "results": {
            "oci": {
                "status": {"state": state, "message": MESSAGE},
                "evaluation_card": ARTIFACT,
            }
        },
    }


@pytest.mark.parametrize("secret", [None, "signing-key", " retained value "])
def test_signing_request_preserves_values(secret: str | None) -> None:
    signing = OCISigningConfig(type="key-pair", secret_ref=secret)
    assert signing.secret_ref == secret
    request = JobSubmissionRequest.model_validate(request_data(signing.model_dump()))
    body = request.model_dump(exclude_none=True, context={"for_submission": True})
    assert body["exports"]["oci"]["signing"] == signing.model_dump()
    assert ("secret_ref" in signing.model_dump()) == (secret is not None)


@pytest.mark.parametrize(
    "signing",
    [
        {},
        {"type": None},
        {"type": ""},
        {"type": "keyless"},
        {"secret_ref": "key"},
        {"type": "key-pair", "secret_ref": ""},
        {"type": "key-pair", "secret_ref": " \t\n"},
    ],
)
def test_invalid_signing_request(signing: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        OCISigningConfig.model_validate(signing)
    with pytest.raises(ValidationError):
        JobSubmissionRequest.model_validate(request_data(signing))


def test_absent_fields_and_empty_oci_container() -> None:
    data = request_data()
    for present in [True, False]:
        if not present:
            data["exports"]["oci"].pop("signing")
        request = JobSubmissionRequest.model_validate(data)
        assert request.exports is not None and request.exports.oci is not None
        assert request.exports.oci.signing is None
        assert "signing" not in request.model_dump()["exports"]["oci"]
    values: list[dict[str, Any] | None] = [None, {}]
    for value in values:
        results = EvaluationJobResults.model_validate({"oci": value})
        if value is None:
            assert results.oci is None
            assert "oci" not in results.model_dump()
        else:
            assert results.model_dump()["oci"] == {}
        # Unrelated optional fields keep their existing serialization policy.
        assert results.model_dump()["mlflow_experiment_url"] is None
    assert (
        EvaluationOCIResults.model_validate(
            {
                "status": None,
                "evaluation_card": None,
                "evaluation_bundle": None,
                "signing": None,
            }
        ).model_dump()
        == {}
    )


@pytest.mark.parametrize(
    "state",
    [
        "pending",
        "generating_evaluation_card",
        "generating_evaluation_bundle",
        "signing",
        "failed",
        "completed",
        "future_server_state",
    ],
)
def test_all_oci_states_remain_independent_of_evaluation(state: str) -> None:
    job = EvaluationJob.model_validate(job_data(state))
    assert job.results is not None and job.results.oci is not None
    assert job.results.oci.status is not None
    assert job.results.oci.status.state == state
    assert job.state == job.effective_state == JobStatus.COMPLETED
    assert is_terminal_job(job)
    assert job.model_dump(mode="json")["results"]["oci"]["evaluation_card"] == ARTIFACT
    assert job.results.oci.signing is None
    assert job.results.oci.evaluation_bundle is None


@pytest.mark.parametrize(
    "status",
    [
        {},
        {"state": None, "message": MESSAGE},
        {"state": 1, "message": MESSAGE},
        {"state": True, "message": MESSAGE},
        {"state": [], "message": MESSAGE},
        {"state": "completed"},
        {"state": "completed", "message": None},
        {"state": "completed", "message": {}},
        {"state": "completed", "message": {"message": "x"}},
        {"state": "completed", "message": {"message_code": "x"}},
        {"state": "completed", "message": {"message": None, "message_code": "x"}},
        {"state": "completed", "message": {"message": "x", "message_code": None}},
    ],
)
def test_invalid_status(status: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        OCIProcessingStatus.model_validate(status)


@pytest.mark.parametrize("field", ["oci_digest", "oci_reference"])
@pytest.mark.parametrize("value", [None, "", " \n\t"])
def test_artifact_requires_nonblank_identity(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        OCIArtifactReference.model_validate({**ARTIFACT, field: value})
    data = dict(ARTIFACT)
    data.pop(field)
    with pytest.raises(ValidationError):
        OCIArtifactReference.model_validate(data)


@pytest.mark.parametrize("field", ["evaluation_card", "evaluation_bundle"])
def test_independently_optional_artifacts(field: str) -> None:
    artifact = {
        "oci_digest": " opaque digest ",
        "oci_reference": " unchanged reference ",
    }
    result = EvaluationOCIResults.model_validate({field: artifact})
    assert result.model_dump() == {field: artifact}


def test_status_only_and_future_outputs() -> None:
    status = {"state": "completed", "message": {"message": "x", "message_code": "x"}}
    result = EvaluationOCIResults.model_validate({"status": status})
    assert result.evaluation_bundle is None and result.signing is None
    result = EvaluationOCIResults.model_validate(
        {
            "evaluation_card": ARTIFACT,
            "evaluation_bundle": ARTIFACT,
            "signing": {"type": "key-pair"},
        }
    )
    assert result.model_dump() == {
        "evaluation_card": ARTIFACT,
        "evaluation_bundle": ARTIFACT,
        "signing": {"type": "key-pair"},
    }
    for signing in [{}, {"type": None}, {"type": "keyless"}, {"type": ""}]:
        with pytest.raises(ValidationError):
            OCISigningResult.model_validate(signing)


def test_public_exports() -> None:
    for model in [
        EvaluationOCIResults,
        OCIArtifactReference,
        OCIProcessingStatus,
        OCISigningConfig,
        OCISigningResult,
    ]:
        assert getattr(evalhub, model.__name__) is model
        assert getattr(models, model.__name__) is model
        assert model.__name__ in evalhub.__all__
        assert model.__name__ in models.__all__


@pytest.mark.parametrize(
    "signing", [None, {"type": "key-pair"}, {"type": "key-pair", "secret_ref": " key "}]
)
def test_sync_job_operations_preserve_oci(signing: Any) -> None:
    response = Mock()
    response.json.return_value = job_data("failed")
    with SyncEvalHubClient() as client:
        with patch.object(client, "_request_post", return_value=response) as post:
            submitted = client.jobs.submit(
                JobSubmissionRequest.model_validate(request_data(signing))
            )
            body = post.call_args.kwargs["json"]["exports"]["oci"]
            if signing is None:
                assert "signing" not in body
            else:
                assert body["signing"] == signing
        with patch.object(client, "_request_get", return_value=response) as get:
            fetched = client.jobs.get("job-id")
            waited = client.jobs.wait_for_completion("job-id", poll_interval=0)
            assert get.call_count == 2
            response.json.return_value = {
                "items": [job_data("failed")],
                "total_count": 1,
            }
            listed = client.jobs.list()
    for job in [submitted, fetched, waited, *listed]:
        assert (
            job.model_dump(mode="json")["results"]["oci"]
            == job_data("failed")["results"]["oci"]
        )
        assert job.model_dump()["exports"]["oci"]["signing"] == {
            "type": "key-pair",
            "secret_ref": "signing-key",
        }


@pytest.mark.parametrize(
    "signing", [None, {"type": "key-pair"}, {"type": "key-pair", "secret_ref": " key "}]
)
async def test_async_job_operations_preserve_oci(signing: Any) -> None:
    response = Mock()
    response.json.return_value = job_data("generating_evaluation_card")
    async with AsyncEvalHubClient() as client:
        with patch.object(
            client, "_request_post", new_callable=AsyncMock, return_value=response
        ) as post:
            submitted = await client.jobs.submit(
                JobSubmissionRequest.model_validate(request_data(signing))
            )
            body = post.call_args.kwargs["json"]["exports"]["oci"]
            if signing is None:
                assert "signing" not in body
            else:
                assert body["signing"] == signing
        with patch.object(
            client, "_request_get", new_callable=AsyncMock, return_value=response
        ) as get:
            fetched = await client.jobs.get("job-id")
            waited = await client.jobs.wait_for_completion("job-id", poll_interval=0)
            assert get.call_count == 2
            response.json.return_value = {
                "items": [job_data("generating_evaluation_card")],
                "total_count": 1,
            }
            listed = await client.jobs.list()
    for job in [submitted, fetched, waited, *listed]:
        assert (
            json.loads(job.model_dump_json())["results"]["oci"]
            == job_data("generating_evaluation_card")["results"]["oci"]
        )
