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
