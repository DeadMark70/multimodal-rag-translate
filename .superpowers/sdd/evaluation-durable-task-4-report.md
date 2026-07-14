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
- FastAPI lifespan initializes the database first, preserves legacy
  `CampaignEngine.recover_inflight_campaigns()` while campaign execution is
  still process-local, then starts/stops the durable worker only once Task 5/6
  handlers have configured it.
- Added lifecycle coverage for recovery, wakeup, shutdown, concurrency, and
  lifespan worker ownership.

## Follow-up Safety Fixes

- The process singleton now refuses to start without a handler. Task 5/6 can
  install the real execution and RAGAS handlers through the typed
  `configure_evaluation_job_worker` path; this task deliberately does not add
  placeholder adapters.
- Claims are now made independently for each work type, preserving the four
  execution and two RAGAS capacity limits even for a mixed ready queue.
- `EvaluationJobStore` exposes a typed post-commit notification hook. Task 5
  producers will provide `worker.notify` when they begin creating durable jobs.

## Verification

```text
python -m pytest tests/test_evaluation_job_worker.py tests/test_rag_startup.py tests/test_evaluation_job_store.py -q -p no:cacheprovider
28 passed

python -m ruff check core/app_factory.py evaluation/job_worker.py evaluation/job_store.py tests/test_evaluation_job_worker.py tests/test_rag_startup.py tests/test_evaluation_job_store.py
All checks passed

git diff --check
No output
```

The pytest run retains existing configuration and third-party Pydantic
deprecation warnings; it has no test failures.
