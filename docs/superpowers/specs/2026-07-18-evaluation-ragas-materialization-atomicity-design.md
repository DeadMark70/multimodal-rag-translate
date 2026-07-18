# Evaluation RAGAS Materialization Atomicity Design

Date: 2026-07-18  
Status: approved approach, written for review

## Problem

The durable evaluation worker may claim newly committed RAGAS work before the
execution worker finishes calling `CampaignRepository.mark_evaluating()`. A
fast RAGAS batch can therefore complete, recompute
`evaluation_completed_units`, and derive a terminal campaign state before the
late `mark_evaluating()` call resets the completed counter to zero. A final
lifecycle derivation can then expose an internally inconsistent campaign such
as `status=completed`, `evaluation_completed_units=0`, and
`evaluation_total_units=1`.

This race is observable in
`test_production_engine_wires_ragas_accounting_scope_and_event`: repeated runs
alternate between the correct `completed + 1/1` state and the invalid
`completed + 0/1` state.

## Decision

RAGAS work materialization and the campaign transition into the evaluation
phase will be one database transaction. Pending RAGAS items must never become
claimable in a committed database state while their campaign still represents
the earlier execution phase or has reset/stale evaluation counters.

The transaction will:

1. insert the RAGAS job, durable work items, and pending job items;
2. derive the distinct target-result count represented by the newly created
   work, rather than using metric-item count;
3. update the owned campaign to `status=evaluating`, `phase=evaluation`,
   `evaluation_completed_units=0`, and the distinct target-result total;
4. commit the job items and campaign transition together;
5. notify the worker only after the commit succeeds.

`ensure_ragas_work()` remains idempotent and keeps its existing integer return
contract for compatibility. The campaign transition occurs only when at least
one new RAGAS item is created. Callers must stop issuing a second,
post-materialization `mark_evaluating()` write for that newly created work.

## Transaction and Failure Semantics

- If any job, work-item, job-item, or campaign-transition statement fails, the
  entire transaction rolls back. No RAGAS item is claimable and no campaign
  phase change is visible.
- The campaign update must include both `campaign_id` and `user_id`; failure to
  update exactly one owned campaign fails closed and rolls back.
- Worker notification occurs after commit. Notification failure must not undo
  committed durable state, matching the existing post-commit notifier
  boundary; it must remain visible to the caller rather than being silently
  swallowed.
- Historical completed counters are reset only as part of starting newly
  committed RAGAS work. Completion paths remain responsible for recomputing
  and publishing durable terminal counts.
- This correction does not change retry-count persistence, evaluator cohort
  selection, score promotion, or frontend contracts.

## Component Boundaries

`EvaluationJobStore` owns this atomic operation because it already owns the
transaction that creates jobs and job items. The store may extend its internal
job-creation helper with an explicit RAGAS campaign-transition option; generic
execution job creation must retain its current behavior.

`CampaignEngine` and `DatasetExecutionWorker` continue deciding when RAGAS
work is required, but no longer perform the late duplicate
`mark_evaluating()` transition after `ensure_ragas_work()` returns a positive
count. Paths that do not materialize new work retain their existing terminal
or recovery handling.

## Verification Design

Testing must be deterministic and follow RED/GREEN:

1. Add a store-level integration test proving that when the post-commit worker
   notifier observes the new pending RAGAS item, the campaign is already
   `evaluating` with `0/N` counters.
2. Add a transaction rollback test for an ownership mismatch or missing
   campaign: neither the RAGAS job/items nor a campaign transition may remain.
3. Add or adapt a worker/campaign regression that simulates immediate RAGAS
   completion and proves a later execution-stage write cannot regress the
   terminal counter.
4. Verify distinct result totals when more than one metric is created for the
   same campaign result.
5. Run the focused job-store, worker, campaign-engine, accounting, and research
   API suites, followed by the complete backend suite.

Success requires repeated focused execution to remain `completed + 1/1`, all
focused tests to pass, and a fresh complete backend suite with zero failures.

## Alternatives Rejected

- A conditional `mark_evaluating()` update was rejected because a separate
  query/update boundary can still race with terminal derivation and may mask
  stale counters instead of preventing them.
- Serializing all RAGAS work behind all dataset execution was rejected because
  it reduces pipeline throughput and solves a persistence-ordering problem by
  changing scheduling policy.
