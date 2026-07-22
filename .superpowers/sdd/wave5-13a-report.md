# Wave 5 Task 13A — Storage and migrations report

## Delivered

- Added `evaluation_evidence_packets` and `evaluation_slot_resolutions` with campaign and attempt foreign keys, schema/condition/run identity, historical defaults, and idempotency keys.
- Added typed evidence/slot record schemas and a typed `AgenticV9TracePayload` on the existing agent trace with historical `None` default.
- Added repository round trips for evidence packets and slot resolutions, with a 256 KiB JSON payload guard.
- Extended existing context packs and claims with optional attempt, condition, and schema-version metadata while preserving v8 defaults.
- Kept provider usage authoritative in the pre-existing `evaluation_usage_events` table; no parallel usage storage was added.

## Migration compatibility

No schema migration conflicts were observed. The additive migration path repairs partial legacy evidence/slot tables by adding the identity, schema, and payload columns with empty/default values. New databases receive foreign keys directly from the table definitions; SQLite cannot retroactively add foreign-key constraints to pre-existing legacy tables, so those retain their existing table-level constraint set while receiving the additive columns and indexes.

## Verification

- `python -m pytest tests/test_evaluation_observability_schema.py tests/test_evaluation_observability_repository.py tests/test_agentic_v9_schemas.py -q` — 44 passed.
- `python -m ruff check evaluation/db.py evaluation/trace_schemas.py evaluation/observability_storage.py tests/test_evaluation_observability_schema.py tests/test_evaluation_observability_repository.py` — passed.
- `git diff --check` — passed.

The test run emits pre-existing Pydantic deprecation warnings from the `storage3` dependency.
