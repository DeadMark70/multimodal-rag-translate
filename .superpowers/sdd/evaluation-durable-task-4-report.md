# Task 4: Single-Process Worker and Recovery

## Delivered

- Added `EvaluationJobWorker`, a process-local worker with explicit `start`,
  `stop`, `notify`, and `run_once` lifecycle APIs plus a singleton accessor.
- Startup recovers interrupted attempts before dispatch begins. Shutdown stops
  claims, cancels handlers and heartbeats, serializes claim/recovery work to
  prevent a late shutdown claim, then records unfinished attempts as interrupted.
- Dispatch is event-driven and retry-aware, with injectable clock/sleep
  functions, independent limits of four execution handlers and two RAGAS
  handlers, and 15-second heartbeat cadence.
- Extended claimed work snapshots with immutable routing metadata (`work_type`
  and `logical_key`) and added ledger lookup for the next retry deadline.
- FastAPI lifespan initializes the database first, starts the durable worker,
  then stops it in `finally`.
- Added lifecycle coverage for recovery, wakeup, shutdown, concurrency, and
  lifespan worker ownership.

## Verification

```text
python -m pytest tests/test_evaluation_job_worker.py tests/test_rag_startup.py tests/test_evaluation_job_schemas.py tests/test_evaluation_job_store.py -q -p no:cacheprovider
31 passed

python -m ruff check evaluation/job_worker.py evaluation/job_schemas.py evaluation/job_store.py core/app_factory.py tests/test_evaluation_job_worker.py tests/test_rag_startup.py
All checks passed

git diff --check
No output
```

The pytest run retains existing configuration and third-party Pydantic
deprecation warnings; it has no test failures.
