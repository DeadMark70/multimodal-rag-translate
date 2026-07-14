# Task 5: Durable Dataset Execution and Campaign Facade

## Delivered

- Added `DatasetExecutionWorker`, which executes only the immutable work-item
  snapshot and promotes a completed `CampaignResult` through the ledger's
  atomic `complete_execution_attempt` path.
- Provider and validation failures are classified and retained as failed
  append-only attempts. They do not create a failed official result.
- Post-promotion observability and trace persistence are best-effort; an
  observability failure cannot undo the already-committed official result.
- Changed `CampaignEngine.create_and_start` to create stable execution work
  items, persist the full case/config/ablation/budget snapshot, notify the
  durable worker, and return immediately. The process-local campaign task map
  and restart reconstruction from result rows have been removed.
- Added durable cancellation of pending/running ledger items and
  `completed_with_errors` campaign state derivation from dataset job-item
  outcomes.
- Reset worker wake/stop events for each lifecycle start so a restarted
  process-local worker does not retain an event loop binding.

## Verification

```text
python -m pytest tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py -k "durable or failed_unit" -q -p no:cacheprovider
2 passed

python -m ruff check evaluation/execution_worker.py evaluation/campaign_engine.py evaluation/campaign_schemas.py evaluation/db.py evaluation/job_store.py evaluation/job_worker.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py --no-cache
All checks passed

git diff --check
No output
```

The requested broader pre-Task-5 suite currently has 11 failures in
`tests/test_campaign_engine.py`. These tests assert the removed process-local
campaign execution/recovery and immediate RAGAS lifecycle. They require the
Task 6 durable RAGAS migration (and durable-job seeded recovery fixtures), not
an execution-worker rollback. The focused Task 5 checks pass.
