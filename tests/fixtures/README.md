# ATIF test fixtures

## ATIF v1.7 contract fixtures

`atif_v1_7_contract.json` exercises the SDK's ATIF v1.7 model and the declared
`Trajectory.extra` extension point. Its trajectory body is adapted from
`eval-hub-contrib/adapters/atif/tests/fixtures/trajectory_v1_7.json` at
`eval-hub-contrib` commit `175dafd245152c4b7aa0b91ef939cae40c5c3537` (Apache-2.0).
The `producer_metadata` object was added here as synthetic contract-test data;
it is not a NeMo Gym artifact or a claim about NeMo Gym's output schema.

`atif_v1_7_undeclared_provenance.json` is the same synthetic trajectory with a
`producer_sha256` key at the ATIF document root. It exists to test the SDK's
unknown-field policy, not to represent a verified Gym field or hash contract.

## Contract exercised

- The SDK accepts the standard ATIF v1.7 fields, including `trajectory_id`.
- `Trajectory.extra` is a declared custom-metadata object. JSON placed inside
  it survives `Trajectory.model_validate()` followed by `to_json_dict()`; the
  SDK does not interpret, validate, or authenticate those values.
- The model configuration is `extra="forbid"`. An unrecognized key at the
  trajectory root (or another model level) fails validation. The SDK does not
  globally ignore or allow undeclared fields.
- If a producer's provenance is represented as undeclared root-level fields,
  the current model rejects the whole trajectory. A consumer that needs to
  retain such fields can capture them from the raw JSON before validation and
  place them in a documented carrier such as `Trajectory.extra`, or a future
  SDK change can add an explicitly typed field. Do not loosen the model's
  unknown-field policy globally without a schema/version compatibility review.

## Results metadata transport

`JobResults.evaluation_metadata` and `JobResults.additional_info` are separate
fields. `DefaultCallbacks.report_results()` does not copy arbitrary
`evaluation_metadata` into `BenchmarkStatusEvent.additional_info`. It sends
`results.additional_info` (or the value returned by an explicitly configured
`generate_additional_info` callback) as `additional_info`; it separately merges
only `evaluation_metadata["artifacts"]` into the event's `artifacts` map. The
ATIF-specific MLflow parameter extraction is another separate path. Tests in
`tests/unit/test_callbacks_events.py` demonstrate that metadata is not
implicitly copied and that explicitly supplied `additional_info` is serialized.

## Genuine NeMo Gym exporter sample

`nemogym_maas/0-0.json` and its `manifest.jsonl` were generated from a real
single-rollout NeMo Gym evaluation, not copied from a checked-in sample. The
source was the bundled `resources_servers/mcqa/data/example.jsonl` task, run
with `mcqa_simple_agent` and the MaaS-configured `openai_model` server. The model
was selected through `MAAS_MODEL`; the endpoint and key were supplied through
`MAAS_URL` and `MAAS_API_KEY` environment references. Their values are
intentionally not recorded. The run enabled `observability_enabled` and
`model_call_capture_dir`, which are required for this path to emit `ng_trajectory`.

Provenance: `NVIDIA-NeMo/Gym` commit
`76116d5d78c8f613cf2038231c66cc02d7ae4a5f`, installed with `uv sync --frozen
--no-dev` on Python 3.13.14; run and export completed 2026-11-15. The exporter
wrote `ATIF-v1.7` with Gym provenance under the declared `extra.nemo_gym` field.
Before check-in, provider-specific model identifiers were replaced with
`maas-model-redacted`; session, trajectory, and opaque request identifiers were
replaced with stable fixture markers. The accompanying manifest digest was
recomputed over the sanitised fixture, so it checks fixture integrity, not the
original export's bytes. The unmodified export remains local. The test validates the checked-in
fixture with the SDK `Trajectory` model and confirms provenance survives
serialization.

This demonstrates compatibility for this specific Gym revision, exporter
profile, MCQA example and MaaS-backed run. It does not prove that every Gym
benchmark or producer revision emits the same data, nor does it establish a
canonicalization rule beyond the bytes Gym hashed in this manifest. The separate
synthetic root-field fixture still demonstrates the SDK's strict rejection of
undeclared root fields; the genuine export does not add such a field because
Gym places its provenance in `extra`.
