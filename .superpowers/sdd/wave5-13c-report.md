# Wave 5 Task 13C — Secure idempotent v9 evidence persistence

## Delivered

- Added attempt-scoped materialization storage with an atomic transaction, duplicate-safe evidence/slot writes, a retained cancelled trace, and no evidence, slot, or claim promotion for cancelled attempts.
- Enforced campaign ownership for evidence, slot, and attempt-scoped claim persistence; the run observability route remains user-owned and has an explicit cross-user denial regression test.
- Sanitized default observability output to bounded plain-text excerpts, removed raw prompt/payload fields, redacted credential patterns in exports, rejected oversized v9 JSON payloads, and removed source display names from persisted v9 scopes.
- Added regressions for prompt-injection-shaped evidence, retry idempotency, cancellation, direct ownership bypass attempts, and cross-user API access.

## Verification

- `python -m pytest tests/test_evaluation_v9_attempt_persistence.py tests/test_evaluation_observability_schema.py tests/test_evaluation_observability_repository.py tests/test_agentic_v9_schemas.py tests/test_evaluation_api.py::test_run_observability_endpoint_denies_a_different_user -q -p no:cacheprovider` — 51 passed.
- `python -m ruff check evaluation/analytics.py evaluation/db.py evaluation/observability_storage.py evaluation/router.py evaluation/trace_schemas.py tests/test_evaluation_v9_attempt_persistence.py tests/test_evaluation_api.py` — passed.
- `git diff --check` — passed.

## Environment note

The full `tests/test_evaluation_api.py` file has one unrelated existing failure because this worktree does not contain `bergen/golden_dataset.json`; all other 16 tests passed, and the new focused cross-user test passed independently.
