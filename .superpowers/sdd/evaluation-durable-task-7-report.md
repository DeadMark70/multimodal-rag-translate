# Task 7: Rerun, Job, Attempt, and Compatibility APIs

## Delivered

- Added the authenticated rerun, campaign-job, job detail, job cancellation,
  and work-item attempt-history routes.
- Added engine-level rerun selection for failed-only, selected, and all scopes;
  execution, RAGAS-only, and combined stages; selected metric validation; and
  stable work-item reuse.
- Kept legacy `/campaigns/{campaign_id}/evaluate` behavior while delegating to
  the durable RAGAS-only rerun path. Legacy rows are backfilled into synthetic
  attempts before metric reruns.
- Added owner checks for campaigns, jobs, and work-item attempt history and
  derived job status/count fields, including `completed_with_errors` support.
- Added job-scoped cancellation that preserves append-only attempt history.
- Execution-only reruns suppress automatic downstream RAGAS creation; combined
  reruns let the execution worker enqueue metrics only after promotion.

## Verification

```text
python -m pytest tests/test_evaluation_api.py -k "rerun_route or attempt_history_route" -q -p no:cacheprovider
2 passed

python -m ruff check --no-cache evaluation/campaign_engine.py evaluation/job_store.py \
  evaluation/job_schemas.py evaluation/router.py evaluation/execution_worker.py \
  tests/test_evaluation_api.py
All checks passed
```

The broader API and job-store suites were also attempted. Their fixtures are
currently blocked before test execution because this workspace denies creation
of `output/test_tmp` and `.test-artifacts`; this is an environment ACL issue,
not a code assertion failure.

