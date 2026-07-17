# Task 5 — Execution Attempt Accounting Report

## Status

Completed. Task 5 adds durable execution-attempt accounting only; no phase-site,
RAGAS, analytics, frontend, or ledger-schema changes were made.

## Delivered

- Added `evaluation/accounting_runtime.py` with `EvaluationAccountingSink`,
  `ExecutionAccountingSession`, and `start_execution_scope()`.
- Callback events are normalized, strictly priced from one worker-lifecycle
  snapshot, idempotently stored by callback event ID, and retain raw usage.
- `DatasetExecutionWorker` creates a UUID run ID and one accounting scope before
  provider execution. Successful result token projections come only from the
  scope ledger summary.
- Successful attempts promote their sole target then finalize the scope as
  `completed`; failed and cancelled attempts finalize respectively without an
  official target. Callback persistence failures keep the successful lifecycle
  and yield `token_accounting_status: partial`.
- `EvaluationJobWorker` interrupts running scopes immediately before each
  durable-attempt recovery call in `start`, `run_until_idle`, and `stop`.

## TDD and Verification

- RED: the new runtime/import test failed with
  `ModuleNotFoundError: evaluation.accounting_runtime` before implementation.
- GREEN: `20 passed`:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_runtime.py tests/test_evaluation_execution_worker.py tests/test_evaluation_job_worker.py -q
  ```

- Ruff format/check passed for all Task 5 files.
- `git diff --check` passed.

## Hang Investigation

The initial runtime test did not isolate `evaluation.db`, so it used the shared
`data/evaluation.db` and stalled during SQLite initialization while earlier
Python processes held the database. The first sandbox test also reported
pytest-cache `WinError 5`. The runtime fixture now uses a unique temporary
database and seeds the required campaign foreign-key parent. The focused test
command was rerun with approved escalation and completed normally.

## Residual Risk

The focused suite still emits pre-existing `storage3` Pydantic deprecation
warnings. No full repository suite was run because this task's requested
verification scope is execution/runtime/job-worker accounting.

## Review Fixes — Atomic Promotion and Strict Token Projection

- `EvaluationJobStore.complete_execution_attempt()` now accepts optional,
  keyword-only `accounting_scope_id`. When supplied, it validates the running
  execution scope and target against the active claim, promotes the target, and
  completes the scope within the existing `BEGIN IMMEDIATE` transaction. Any
  identity, lifecycle, or row-count mismatch rolls back the result, attempt,
  work item, and accounting writes together.
- Execution workers now use that atomic path and have no success-path
  post-commit accounting writes.
- Scope summaries treat callback `status='failed'` as partial even if it has
  measured tokens. The v2 result projection has `total_tokens: null` for
  unavailable or partial aggregates; zero-valued category fields are not
  synthesized for an unobserved scope. The legacy SQL column remains its
  backward-compatible `0` sentinel, which the v2 read projection converts back
  to `None` only when the persisted token payload explicitly says null.
- Added regression tests for atomic promotion/rollback, provider failure and
  cancellation without official results, missing usage, persistence failure,
  and failed callbacks with measured usage.

Verification for the review fixes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_store.py tests/test_evaluation_accounting_runtime.py tests/test_evaluation_execution_worker.py tests/test_evaluation_job_worker.py tests/test_evaluation_job_store.py -q
```

Result: `52 passed` (with the same pre-existing `storage3` deprecation
warnings).

## Review Fixes — Post-Commit Closure

- `complete_execution_attempt()` now returns a `model_copy()` of its already
  persisted result with the resolved official result ID and source attempt ID.
  It performs no await after the transaction commit, so cancellation or a
  repository read error cannot affect the completed accounting scope.
- The legacy RAGAS metrics response maps a v2 `token_usage.total_tokens: null`
  to `0` only while constructing its non-null `CampaignMetricRow`; strict
  accounting projections remain null elsewhere.

## Review Fixes — Rerun Return Metadata

- `complete_execution_attempt()` reads the existing official result's `id` and
  `created_at` within its transaction. An upsert conflict now returns that
  persisted creation timestamp together with the resolved ID and current source
  attempt, while a new row returns the timestamp just persisted for that row.
- No post-commit repository read was reintroduced. The rerun regression test
  verifies the returned model matches the stored result identity and creation
  time while retaining the current attempt and output fields.
