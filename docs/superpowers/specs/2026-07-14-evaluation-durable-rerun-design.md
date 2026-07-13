# Durable Evaluation Rerun Design

**Date:** 2026-07-14

**Status:** Approved design

**Scope:** Evaluation Center campaign execution, RAGAS scoring, recovery, reruns, and result selection
**Deployment assumption:** One computer, one FastAPI backend process

## 1. Objective

Make evaluation campaigns recoverable, maintainable, and efficient when dataset execution or RAGAS scoring is interrupted or fails.

The system must preserve every execution attempt, including failures, cancellations, and interruptions. Analytics, comparisons, charts, ECR calculations, and exports must use only the latest successful attempt compatible with the current evaluation version. A failed attempt must never be converted into a zero score or replace an earlier successful result.

## 2. Current Problems

The existing evaluation workflow already supports startup recovery and manual RAGAS reruns, but its recovery boundary is too coarse:

- Campaign work runs in process-local `asyncio.create_task` tasks.
- Raw execution is checkpointed only after one question-mode-run unit finishes.
- Existing failed raw results may be treated as already processed during recovery.
- RAGAS batch results accumulate in memory and are persisted only after all batches finish.
- A late interruption can therefore cause already scored rows to be evaluated again.
- Some metric failures are persisted as `0.0` with an invalid flag, which can be confused with a real low score.
- Missing RAGAS dependencies can clear scores while the campaign still reaches a completed state.
- Reruns overwrite the official score projection and do not retain an explicit attempt history.
- Process-local active-task tracking is not a durable source of truth.

## 3. Chosen Approach

Use SQLite as a durable work ledger and run one in-process evaluation worker supervised by the FastAPI lifespan.

This approach is preferred over adding Redis, Celery, or RQ because the supported deployment is one computer and one backend process. SQLite already owns evaluation persistence and is sufficient for durable work claims, checkpoints, retries, and startup recovery.

The worker provides at-least-once task execution. Successful projection into official results is idempotent and transactional. External model calls cannot be guaranteed exactly once if the process exits after a provider response but before the local checkpoint, unless the provider offers an idempotency mechanism.

## 4. Core Invariants

1. Attempts are append-only and are never overwritten.
2. Failed, cancelled, and interrupted attempts never participate in statistics.
3. A failed rerun never replaces an earlier successful official result.
4. A newer compatible successful attempt becomes the official result atomically.
5. If no compatible successful attempt exists, the value is missing, not zero.
6. A completed checkpoint is never intentionally executed again after restart.
7. One logical work item cannot run twice concurrently in the supported single process.
8. Evaluation signatures prevent scores produced under incompatible policies from being aggregated together.
9. Existing result and analytics APIs remain compatible during staged migration.

## 5. Logical Work Units

### 5.1 Dataset execution

One logical work item represents:

`question_id × mode × run_number × ablation_condition`

Its stable logical key includes every field that changes execution semantics. Completing it produces a candidate campaign result.

### 5.2 RAGAS metric evaluation

One logical work item represents:

`campaign_result_id × metric_name × evaluation_signature`

Work remains individually checkpointable even when the worker groups compatible items into one RAGAS batch.

## 6. Persistent Data Model

### 6.1 `evaluation_jobs`

An evaluation job represents one user-requested operation, such as initial campaign execution, retrying failures, rerunning selected questions, rerunning RAGAS only, or rerunning execution and RAGAS.

Required fields:

- `id`
- `campaign_id`
- `user_id`
- `job_type`
- `status`
- `selection_json`
- `config_snapshot_json`
- `total_items`
- `succeeded_items`
- `failed_items`
- `cancelled_items`
- `created_at`
- `started_at`
- `completed_at`
- `updated_at`

Job status is derived from its linked work rather than being treated as an independent source of truth.

### 6.2 `evaluation_work_items`

A work item is the stable identity of one logical task across automatic and manual reruns.

Required fields:

- `id`
- `campaign_id`
- `user_id`
- `work_type`: `dataset_execution` or `ragas_metric`
- `logical_key`
- `question_id`
- `mode`
- `run_number`
- `condition_id`
- `campaign_result_id`
- `metric_name`
- `evaluation_signature`
- `latest_success_attempt_id`
- `created_at`
- `updated_at`

The unique constraint is scoped to the campaign and stable logical key. A rerun reuses the work item and appends a new attempt.

### 6.3 `evaluation_job_items`

This operational table associates a user-requested job with the stable work items selected for that run. Retry counters and scheduling state live here rather than on the stable work item, so a new manual rerun receives a fresh automatic-retry budget and old job history never changes retroactively.

Required fields:

- `id`
- `job_id`
- `work_item_id`
- `status`
- `max_attempts`
- `automatic_attempt_count`
- `next_retry_at`
- `active_attempt_id`
- `created_at`
- `updated_at`

The pair `(job_id, work_item_id)` is unique.

Job-item statuses are:

- `pending`
- `running`
- `retry_wait`
- `succeeded`
- `failed`
- `interrupted`
- `cancelled`

Only one non-terminal job item may actively target a stable work item. The claim transaction enforces this invariant in the supported single process. Job progress is derived from job-item state and attempts linked to that job, never from the current state of a later rerun.

### 6.4 `evaluation_attempts`

Attempts are append-only execution records.

Required fields:

- `id`
- `job_id`
- `job_item_id`
- `work_item_id`
- `attempt_number`
- `status`
- `retryable`
- `error_type`
- `safe_error_message`
- `error_details_json`
- `input_snapshot_json`
- `output_json`
- `token_usage_json`
- `log_correlation_id`
- `started_at`
- `heartbeat_at`
- `completed_at`
- `created_at`

Attempt statuses are:

- `running`
- `succeeded`
- `failed`
- `interrupted`
- `cancelled`

The full internal exception belongs in local logs. Persisted or API-visible messages must be sanitized to avoid exposing credentials, provider payloads, filesystem paths, or document content.

### 6.5 Official projections

The existing `campaign_results` and `ragas_scores` tables remain the official result projections consumed by current analytics. Add `source_attempt_id` to both projections. Add an evaluation signature field to RAGAS score rows if it is not already available in normalized form.

On success, one SQLite transaction must:

1. Persist the successful attempt.
2. Set `latest_success_attempt_id` on the work item.
3. Upsert the compatible official result or score projection.
4. Clear `active_attempt_id` and update job-item status.
5. Update derived job and campaign progress.

A failed attempt updates only attempt and control state. It must not mutate the official result projection.

Legacy campaign results and scores are migrated into synthetic legacy attempts so existing history remains traceable.

## 7. Evaluation Signature

RAGAS official-result selection requires a deterministic signature containing:

- campaign result ID
- final answer hash
- evaluator model and model configuration relevant to scoring
- metric name and metric implementation version
- context policy version
- effective ground-truth hash
- enabled context-metric policy
- prompt or evaluator policy version when applicable

The signature is serialized canonically and hashed. A successful historical score with a different signature remains visible in attempt history but cannot satisfy the current work item or enter the current aggregate.

## 8. Worker Architecture

### 8.1 Lifecycle

The FastAPI lifespan starts one `EvaluationJobWorker`. The worker:

1. Performs startup recovery.
2. Waits on an `asyncio.Event` when no work is ready.
3. Is awakened immediately when an API request creates or requeues work.
4. Performs a low-frequency fallback scan for due retries.
5. Stops claiming new work during application shutdown.
6. Checkpoints or marks active work interrupted before shutdown completes where possible.

### 8.2 Claiming work

Job-item claiming and the transition to `running` occur in a short SQLite transaction. The transaction creates the new attempt and sets `active_attempt_id` before provider work starts.

Because deployment is restricted to one process, no distributed lease service is required. A process-local guard prevents duplicate claims, while SQLite remains the durable source of truth.

### 8.3 Heartbeats

Active attempts update `heartbeat_at` approximately every 15 seconds. Heartbeats are for progress visibility and interruption diagnosis, not multi-node coordination.

### 8.4 Dataset execution

Dataset execution reuses the existing campaign case runner. Each question-mode-run unit is independently persisted. Successful output is promoted to `campaign_results`; failures are retained only in attempt history until a later success exists.

When all required execution work reaches a terminal state, RAGAS work is generated only for compatible official successful campaign results.

### 8.5 RAGAS batching

RAGAS work is durable per result and metric. For performance, the worker groups ready work items that share:

- metric name
- evaluator model and configuration
- evaluation signature policy
- context-metric policy

Each grouped provider call returns independently checkpointed attempt outcomes. A failure in one metric must not manufacture zero values for every result or erase prior valid scores.

## 9. Recovery and Rerun Semantics

### 9.1 Startup recovery

At startup:

1. Every attempt left in `running` becomes `interrupted`.
2. Its job item becomes `pending` if interruption is retryable and retry budget remains.
3. Existing compatible successful work stays succeeded.
4. Due `retry_wait` job items return to pending.
5. Campaign and job progress is recomputed from durable state.
6. The worker resumes only unresolved work.

### 9.2 Automatic retry

Default limits:

- Dataset execution: three attempts.
- RAGAS metric evaluation: five attempts.

Retry delay uses exponential backoff with bounded jitter. A provider `Retry-After` value takes precedence when available.

Automatic retry exhaustion leaves the job item failed and available for a later manual rerun. A manual rerun creates a new job item and attempt with a fresh automatic-retry budget; it is not blocked by exhausted history from an earlier job.

### 9.3 Manual rerun

Supported scopes:

- failed work only
- selected questions
- all eligible work

Supported stages:

- execution only
- RAGAS only
- execution followed by RAGAS

Rerunning execution invalidates downstream RAGAS compatibility through the answer hash only after the new execution succeeds and is promoted. New RAGAS work is then generated for that answer. If the new execution fails, the previous successful execution remains official and its corresponding RAGAS scores remain the current official scores; they become historical only after a newer execution is successfully promoted.

### 9.4 Cancellation

Cancellation prevents new claims for the selected job. Active provider calls are cancelled cooperatively when supported. The active attempt is recorded as cancelled, while existing successful official results remain intact.

## 10. Error Policy

Standard error types are:

- `rate_limit`
- `provider_unavailable`
- `timeout`
- `transport`
- `process_interrupted`
- `cancelled`
- `authentication`
- `invalid_configuration`
- `invalid_dataset`
- `missing_dependency`
- `metric_failure`
- `persistence_failure`
- `unknown`

Retryable errors include timeouts, transport failures, HTTP 408, HTTP 429, HTTP 5xx, temporary provider unavailability, and process interruption.

Non-retryable errors include invalid datasets, missing required ground truth, unsupported metrics, authentication failures, permanent configuration errors, and missing runtime dependencies.

Missing dependencies or evaluator initialization failures must fail affected work. They must not clear official scores or complete the campaign successfully.

## 11. Campaign and Job Status

Campaign lifecycle adds `completed_with_errors`.

- `completed`: all required current work has compatible successful results.
- `completed_with_errors`: reporting can proceed, but at least one selected work item has no compatible success.
- `failed`: insufficient successful work exists to produce a valid report.
- `cancelled`: the current user-requested operation was cancelled.

The UI and API expose counts for valid results, unresolved failures, retry wait, recovered interruptions, and missing scores. Aggregate responses must include sample counts so partial reports cannot be mistaken for complete reports.

## 12. API Design

### 12.1 Unified rerun endpoint

`POST /api/evaluation/campaigns/{campaign_id}/reruns`

Request fields:

- `scope`: `failed_only`, `selected`, or `all`
- `stages`: `execution`, `ragas`, or `execution_and_ragas`
- `question_ids`: optional selection
- `metric_names`: optional RAGAS selection

The response returns the created job ID and durable status.

### 12.2 Job and attempt inspection

- `GET /api/evaluation/campaigns/{campaign_id}/jobs`
- `GET /api/evaluation/jobs/{job_id}`
- `POST /api/evaluation/jobs/{job_id}/cancel`
- `GET /api/evaluation/work-items/{work_item_id}/attempts`

Existing ownership checks remain enforced through the authenticated user ID.

### 12.3 Compatibility endpoints

The existing `POST /api/evaluation/campaigns/{campaign_id}/evaluate` endpoint remains during migration. It delegates to the new RAGAS-only rerun path. Existing campaign status, result, metric, export, and SSE APIs remain compatible and are extended with optional durable-job fields.

## 13. Frontend Behavior

The Evaluation Center supports:

- retry all failed work
- rerun selected questions
- rerun RAGAS without rerunning RAG generation
- rerun a single failed metric
- inspect job and attempt history
- inspect attempt number, timestamps, safe error reason, retryability, and active official attempt
- display valid, failed, retrying, interrupted, and missing-result counts
- distinguish `completed` from `completed_with_errors`

SSE remains an optimization for live progress. The UI falls back to durable REST polling, and a page refresh reconstructs state entirely from SQLite-backed APIs.

## 14. Performance Controls

SQLite configuration:

- WAL mode
- bounded `busy_timeout`
- short claim transactions
- one transaction per completed result batch
- heartbeat writes no more frequently than approximately 15 seconds

Required indexes:

- job items by `status, next_retry_at`
- attempts by `work_item_id, attempt_number`
- jobs by `campaign_id, created_at`
- official RAGAS rows by result, metric, and signature

Initial concurrency defaults:

- Dataset execution respects campaign configuration with a maximum of four.
- RAGAS batch size defaults to four.
- RAGAS parallel batches default to two.

Rate-limit responses delay future work rather than causing immediate replacement batches. These defaults can be tuned later from measured provider latency and retry data.

## 15. Module Boundaries

Create or extract the following modules:

- `evaluation/job_schemas.py`: job, work-item, attempt, and rerun API models
- `evaluation/job_store.py`: durable control-state repositories and promotion transactions
- `evaluation/job_worker.py`: lifecycle, wakeup, scheduling, recovery, and cancellation
- `evaluation/execution_worker.py`: dataset execution adapter
- `evaluation/ragas_worker.py`: compatible RAGAS batching and score promotion
- `evaluation/error_policy.py`: normalized error classification and retry decisions

`evaluation/campaign_engine.py` becomes a facade for campaign creation, query, cancellation, and rerun requests. `evaluation/router.py` remains limited to transport, authentication, and response mapping.

## 16. Testing Strategy

### 16.1 Persistence and state machine

- Stable logical keys do not create duplicate work items.
- Every automatic or manual execution creates a new attempt.
- A failed attempt never updates the official projection.
- A newer successful attempt atomically replaces the older official projection.
- Signature-incompatible scores are excluded.
- Legacy rows migrate without data loss.

### 16.2 Recovery

- Restart during execution requeues only the interrupted unit.
- Restart after nine of ten RAGAS items requeues only the missing item.
- Running attempts become interrupted during startup recovery.
- Retry-wait work is not claimed before `next_retry_at`.
- Successfully checkpointed work does not call the runner again.

### 16.3 Errors and retries

- HTTP 408, 429, 5xx, timeout, and transport errors retry.
- Authentication, missing dependencies, invalid configuration, and invalid datasets do not auto-retry.
- Retry exhaustion remains manually rerunnable.
- RAGAS failure produces a missing metric, not zero.
- A failed rerun preserves the previous successful official result.

### 16.4 Statistical safety

- Failed, cancelled, and interrupted attempts never enter means, deltas, ECR, charts, or exports.
- A missing compatible success returns an explicit missing value and warning.
- Aggregates report the actual successful sample count.
- Evaluator and context-policy versions are never mixed silently.

### 16.5 Frontend

- Job and attempt histories render correctly.
- Failed-only, selected, RAGAS-only, and combined reruns send correct requests.
- SSE failure falls back to polling without losing durable status.
- Refresh restores active progress.
- `completed_with_errors` displays excluded-item counts and does not imply complete coverage.

## 17. Acceptance Criteria

The design is complete when implementation demonstrates all of the following:

1. Killing and restarting the backend during a campaign resumes unresolved work without intentionally rerunning successful checkpoints.
2. Killing and restarting during RAGAS preserves completed metric checkpoints.
3. Every rerun retains prior attempts and errors.
4. Only the latest compatible successful attempt is promoted as official.
5. Failed attempts and invalid metrics do not contribute zero values to statistics.
6. Existing successful results remain available after a failed rerun.
7. Missing dependencies and permanent configuration failures are visibly failed states.
8. Existing evaluation result, analytics, export, and SSE consumers remain compatible through the migration.
9. Focused backend state-machine, recovery, RAGAS, API, and frontend tests pass.
10. Full backend and frontend verification is run, with any environment-limited failures reported exactly.

## 18. Out of Scope

- Redis, Celery, RQ, or distributed worker coordination
- Multi-host or multi-process work leases
- Provider-side exactly-once billing guarantees
- Automatic deletion of historical attempts
- Replacing SQLite with another database
- Redesigning RAG quality metrics unrelated to failure recovery and score validity
