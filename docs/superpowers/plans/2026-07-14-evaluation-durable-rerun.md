# Durable Evaluation Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dataset execution and RAGAS scoring durably checkpointed, safely rerunnable, and statistically immune to failed attempts.

**Architecture:** SQLite becomes the durable work ledger for one in-process evaluation worker. Stable work items retain identity across reruns, job items own per-request retry state, attempts are append-only, and successful attempts atomically promote backward-compatible `campaign_results` or `ragas_scores` projections.

**Tech Stack:** Python 3.13, FastAPI, asyncio, aiosqlite/SQLite WAL, Pydantic 2, RAGAS 0.4, pytest/pytest-asyncio, React 18, TypeScript 5, Chakra UI, Vitest, Testing Library.

## Global Constraints

- Supported deployment is one computer and one FastAPI backend process; do not add Redis, Celery, RQ, or distributed leases.
- Attempts are append-only; failed, cancelled, and interrupted attempts never enter analytics or exports.
- Only the latest compatible successful attempt may update an official projection.
- A failed rerun must preserve the previous successful official result and score.
- Missing compatible metrics are represented as missing values, never synthetic zero scores.
- RAGAS compatibility is keyed by answer hash, evaluator model/config, metric/version, context policy, effective ground-truth hash, and enabled context-metric policy.
- Keep `evaluation/router.py` transport-only and do not introduce router-to-router imports.
- New and modified Python signatures require type hints; use logging rather than `print`.
- Existing authenticated ownership checks using `user_id` remain mandatory.
- Existing campaign result, analytics, export, and SSE consumers remain backward compatible during migration.
- Dataset execution automatic attempts default to 3; RAGAS metric automatic attempts default to 5.
- Initial RAGAS defaults are batch size 4 and parallel batches 2, while dataset execution remains capped at 4.
- Use TDD for every behavior change and commit each task separately.

---

## File Map

### Backend files to create

- `evaluation/job_schemas.py`: durable job, job-item, work-item, attempt, rerun, and projection models.
- `evaluation/error_policy.py`: normalized error classification, retryability, and retry delay.
- `evaluation/job_store.py`: SQLite work ledger, startup recovery, claims, attempts, and atomic promotion.
- `evaluation/job_worker.py`: one-process worker lifecycle, wakeup, scheduling, heartbeat, and shutdown.
- `evaluation/execution_worker.py`: adapter from durable execution claims to the existing campaign runner.
- `evaluation/ragas_worker.py`: compatible metric grouping, RAGAS batch execution, and per-metric checkpoints.
- `tests/test_evaluation_job_schemas.py`
- `tests/test_evaluation_error_policy.py`
- `tests/test_evaluation_job_store.py`
- `tests/test_evaluation_job_worker.py`
- `tests/test_evaluation_execution_worker.py`
- `tests/test_evaluation_ragas_worker.py`

### Backend files to modify

- `evaluation/db.py`: additive schema migration, WAL busy timeout, official projection source-attempt fields.
- `evaluation/campaign_schemas.py`: `completed_with_errors` and durable count/status fields.
- `evaluation/campaign_engine.py`: facade over durable jobs instead of process-local campaign tasks.
- `evaluation/ragas_evaluator.py`: expose one-metric batch evaluation without campaign-wide delayed replacement.
- `evaluation/router.py`: rerun/job/attempt APIs and compatibility delegation.
- `core/app_factory.py`: start and stop the durable worker in lifespan.
- `tests/test_campaign_engine.py`
- `tests/test_ragas_evaluator.py`
- `tests/test_evaluation_api.py`
- `tests/test_evaluation_analytics_api.py`
- `docs/BACKEND.md`
- `docs/RELIABILITY.md`
- `docs/generated/api-surface.md`

### Frontend files to create

- `Multimodal_RAG_System/src/components/evaluation/EvaluationJobPanel.tsx`
- `Multimodal_RAG_System/src/components/evaluation/EvaluationJobPanel.test.tsx`

### Frontend files to modify

- `Multimodal_RAG_System/src/types/evaluation.ts`
- `Multimodal_RAG_System/src/services/evaluationApi.ts`
- `Multimodal_RAG_System/src/services/evaluationApi.test.ts`
- `Multimodal_RAG_System/src/components/evaluation/EvaluationResults.tsx`
- `Multimodal_RAG_System/src/components/evaluation/EvaluationResults.test.tsx`
- `Multimodal_RAG_System/src/components/evaluation/CampaignRunner.tsx`
- `Multimodal_RAG_System/src/components/evaluation/CampaignRunner.test.tsx`
- `Multimodal_RAG_System/docs/FRONTEND.md`
- `Multimodal_RAG_System/docs/RELIABILITY.md`

---

### Task 1: Durable Schemas and Error Policy

**Files:**
- Create: `evaluation/job_schemas.py`
- Create: `evaluation/error_policy.py`
- Create: `tests/test_evaluation_job_schemas.py`
- Create: `tests/test_evaluation_error_policy.py`

**Interfaces:**
- Produces: `EvaluationJobType`, `EvaluationWorkType`, `EvaluationJobItemStatus`, `EvaluationAttemptStatus`, `EvaluationRerunScope`, `EvaluationRerunStages`.
- Produces: `EvaluationRerunRequest`, `WorkItemSpec`, `EvaluationJob`, `EvaluationJobItem`, `EvaluationAttempt`, `ClaimedEvaluationWork`.
- Produces: `ErrorDecision(error_type: str, retryable: bool, retry_after_seconds: float | None, safe_message: str)`.
- Produces: `classify_evaluation_error(exc: BaseException) -> ErrorDecision` and `retry_delay_seconds(attempt_number: int, retry_after_seconds: float | None) -> float`.

- [x] **Step 1: Write schema normalization tests**

```python
from evaluation.job_schemas import EvaluationRerunRequest


def test_rerun_request_deduplicates_selected_ids() -> None:
    request = EvaluationRerunRequest(
        scope="selected",
        stages="ragas",
        question_ids=["Q1", " Q1 ", "", "Q2"],
        metric_names=["faithfulness", "faithfulness"],
    )

    assert request.question_ids == ["Q1", "Q2"]
    assert request.metric_names == ["faithfulness"]


def test_selected_rerun_requires_question_ids() -> None:
    with pytest.raises(ValueError, match="question_ids"):
        EvaluationRerunRequest(scope="selected", stages="execution")
```

- [x] **Step 2: Write transient/permanent error policy tests**

```python
import httpx
from google.api_core import exceptions as google_exceptions

from evaluation.error_policy import classify_evaluation_error, retry_delay_seconds


def test_rate_limit_is_retryable() -> None:
    decision = classify_evaluation_error(google_exceptions.ResourceExhausted("quota"))
    assert decision.error_type == "rate_limit"
    assert decision.retryable is True


def test_transport_error_is_retryable() -> None:
    decision = classify_evaluation_error(httpx.ConnectError("offline"))
    assert decision.error_type == "transport"
    assert decision.retryable is True


def test_authentication_error_is_permanent() -> None:
    exc = type("ProviderAuthError", (Exception,), {"status_code": 401})("bad key")
    decision = classify_evaluation_error(exc)
    assert decision.error_type == "authentication"
    assert decision.retryable is False


def test_retry_after_takes_precedence() -> None:
    assert retry_delay_seconds(3, 17.0) == 17.0
```

- [x] **Step 3: Run tests and verify they fail**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_job_schemas.py tests/test_evaluation_error_policy.py -q -p no:cacheprovider
```

Expected: collection fails because `evaluation.job_schemas` and `evaluation.error_policy` do not exist.

- [x] **Step 4: Implement the exact status models and request validation**

Implement these public declarations in `evaluation/job_schemas.py`:

```python
class EvaluationJobType(str, Enum):
    INITIAL = "initial"
    RERUN = "rerun"


class EvaluationWorkType(str, Enum):
    DATASET_EXECUTION = "dataset_execution"
    RAGAS_METRIC = "ragas_metric"


class EvaluationJobItemStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class EvaluationAttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


EvaluationRerunScope = Literal["failed_only", "selected", "all"]
EvaluationRerunStages = Literal["execution", "ragas", "execution_and_ragas"]


class EvaluationRerunRequest(BaseModel):
    scope: EvaluationRerunScope
    stages: EvaluationRerunStages
    question_ids: list[str] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_selection(self) -> "EvaluationRerunRequest":
        self.question_ids = list(dict.fromkeys(value.strip() for value in self.question_ids if value.strip()))
        self.metric_names = list(dict.fromkeys(value.strip() for value in self.metric_names if value.strip()))
        if self.scope == "selected" and not self.question_ids:
            raise ValueError("selected reruns require question_ids")
        return self
```

Add fully typed Pydantic models for every interface listed above. `ClaimedEvaluationWork` must contain job, job-item, work-item, and attempt IDs plus the immutable input snapshot.

- [x] **Step 5: Implement explicit error classification**

Implement `ErrorDecision`, status-code extraction through the exception chain, sanitized messages, and retry delay. Retry only timeout, transport, HTTP 408/429/5xx, Google `ResourceExhausted`/`ServiceUnavailable`, and explicit process interruption. Classify 401/403 as authentication, `ModuleNotFoundError` as missing dependency, and Pydantic/value errors as invalid configuration or dataset.

```python
def retry_delay_seconds(
    attempt_number: int,
    retry_after_seconds: float | None,
) -> float:
    if retry_after_seconds is not None:
        return max(0.0, min(retry_after_seconds, 900.0))
    exponent = max(0, attempt_number - 1)
    return min(2.0**exponent, 60.0)
```

- [x] **Step 6: Run focused tests**

Run the Task 1 command again. Expected: all tests pass.

- [x] **Step 7: Commit Task 1**

```powershell
git add evaluation/job_schemas.py evaluation/error_policy.py tests/test_evaluation_job_schemas.py tests/test_evaluation_error_policy.py
git commit -m "feat: define durable evaluation job states"
```

---

### Task 2: SQLite Work Ledger and Atomic Claims

**Files:**
- Create: `evaluation/job_store.py`
- Create: `tests/test_evaluation_job_store.py`
- Modify: `evaluation/db.py`

**Interfaces:**
- Consumes: Task 1 models.
- Produces: `EvaluationJobStore.create_job_with_items(*, user_id: str, campaign_id: str, job_type: EvaluationJobType, selection: dict[str, Any], config_snapshot: dict[str, Any], items: Sequence[WorkItemSpec]) -> EvaluationJob`.
- Produces: `EvaluationJobStore.claim_ready_items(*, limit: int, now: datetime) -> list[ClaimedEvaluationWork]`.
- Produces: `EvaluationJobStore.fail_attempt(claim: ClaimedEvaluationWork, decision: ErrorDecision, *, next_retry_at: datetime | None) -> EvaluationAttempt`.
- Produces: `EvaluationJobStore.cancel_attempt(claim: ClaimedEvaluationWork, *, safe_message: str) -> EvaluationAttempt`.
- Produces: `EvaluationJobStore.heartbeat_attempt(attempt_id: str, *, at: datetime) -> None`.
- Produces: `EvaluationJobStore.recover_interrupted_attempts(*, at: datetime) -> int`.
- Produces: `EvaluationJobStore.list_jobs(*, user_id: str, campaign_id: str) -> list[EvaluationJob]`, `get_job(*, user_id: str, job_id: str) -> EvaluationJob`, and `list_attempts(*, user_id: str, work_item_id: str) -> list[EvaluationAttempt]`.

- [x] **Step 1: Write migration and claim tests**

```python
@pytest.mark.asyncio
async def test_create_job_reuses_stable_work_item_and_creates_new_job_item() -> None:
    store = EvaluationJobStore()
    spec = WorkItemSpec(
        work_type="dataset_execution",
        logical_key="execution:Q1:naive:1:none",
        input_snapshot={"question_id": "Q1"},
        max_attempts=3,
    )

    first = await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[spec],
    )
    second = await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={"scope": "all"},
        config_snapshot={},
        items=[spec],
    )

    assert first.id != second.id
    assert await store.count_work_items(campaign_id="cmp-1") == 1
    assert await store.count_job_items(campaign_id="cmp-1") == 2


@pytest.mark.asyncio
async def test_claim_creates_running_attempt_atomically() -> None:
    claimed = await seeded_store.claim_ready_items(limit=1, now=fixed_now)
    assert len(claimed) == 1
    assert claimed[0].attempt_number == 1
    assert await seeded_store.get_job_item_status(claimed[0].job_item_id) == "running"
```

Patch `evaluation.db.EVALUATION_DB_PATH` to a workspace-local test database and call `force_init_db()` in the fixture.

- [x] **Step 2: Run the store tests and verify failure**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_job_store.py -q -p no:cacheprovider
```

Expected: FAIL because ledger tables and `EvaluationJobStore` do not exist.

- [x] **Step 3: Add the four ledger tables and indexes**

Extend `_INIT_SQL` with `evaluation_jobs`, `evaluation_work_items`, `evaluation_job_items`, and `evaluation_attempts`, matching the approved design. Enforce:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_work_item_logical_key
ON evaluation_work_items(campaign_id, logical_key);

CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_job_item_pair
ON evaluation_job_items(job_id, work_item_id);

CREATE INDEX IF NOT EXISTS idx_eval_job_item_ready
ON evaluation_job_items(status, next_retry_at, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_attempt_number
ON evaluation_attempts(work_item_id, attempt_number);
```

Add nullable `source_attempt_id` to `campaign_results` and `ragas_scores`, plus nullable `evaluation_signature` to `ragas_scores`. Add `PRAGMA busy_timeout=5000;` in `connect_db()` while retaining WAL, `synchronous=NORMAL`, and foreign keys.

- [x] **Step 4: Implement transactional job creation and claims**

`create_job_with_items` must insert one job, upsert work items by stable logical key, and create job items with a fresh retry budget. `claim_ready_items` must use `BEGIN IMMEDIATE`, select only pending or due retry-wait job items, reject a claim if another non-terminal job item already targets the same work item, create the running attempt, set `active_attempt_id`, and commit before returning claims.

Use this attempt-number query inside the claim transaction:

```sql
SELECT COALESCE(MAX(attempt_number), 0) + 1
FROM evaluation_attempts
WHERE work_item_id = ?;
```

- [x] **Step 5: Implement heartbeat, failure, cancellation, and startup interruption**

`fail_attempt` records sanitized error metadata. Retryable attempts below the job-item budget transition to `retry_wait` with `next_retry_at`; exhausted or permanent errors transition to `failed`. `recover_interrupted_attempts` changes all running attempts to interrupted, clears job-item active attempt IDs, and requeues eligible job items.

- [x] **Step 6: Run focused tests and lint**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_job_store.py -q -p no:cacheprovider
$env:RUFF_NO_CACHE='true'; & '.\.venv\Scripts\python.exe' -m ruff check evaluation/job_schemas.py evaluation/error_policy.py evaluation/job_store.py evaluation/db.py tests/test_evaluation_job_store.py
```

Expected: both commands exit 0.

- [x] **Step 7: Commit Task 2**

```powershell
git add evaluation/db.py evaluation/job_store.py tests/test_evaluation_job_store.py
git commit -m "feat: persist evaluation work ledger"
```

---

### Task 3: Atomic Official Projection and Legacy Backfill

**Files:**
- Modify: `evaluation/job_schemas.py`
- Modify: `evaluation/job_store.py`
- Modify: `evaluation/db.py`
- Modify: `tests/test_evaluation_job_store.py`
- Modify: `tests/test_ragas_evaluator.py`

**Interfaces:**
- Produces: `ExecutionAttemptOutput` and `RagasAttemptOutput`.
- Produces: `build_evaluation_signature(*, result: CampaignResult, evaluator_model: str, evaluator_config: dict[str, Any], metric_name: str, metric_version: str, ground_truth_hash: str, context_metrics_enabled: bool) -> str`.
- Produces: `complete_execution_attempt(claim: ClaimedEvaluationWork, output: ExecutionAttemptOutput) -> CampaignResult`.
- Produces: `complete_ragas_attempt(claim: ClaimedEvaluationWork, output: RagasAttemptOutput) -> None`.
- Produces: `backfill_legacy_attempts() -> None`.

- [x] **Step 1: Write promotion safety tests**

```python
@pytest.mark.asyncio
async def test_failed_rerun_preserves_previous_success_projection() -> None:
    first_claim = await claim_execution(store)
    first = await store.complete_execution_attempt(first_claim, successful_output("answer-v1"))
    rerun_claim = await claim_execution_rerun(store)
    await store.fail_attempt(rerun_claim, RuntimeError("provider down"))

    current = await CampaignResultRepository().get(
        user_id="user-a", campaign_id="cmp-1", result_id=first.id
    )
    assert current.answer == "answer-v1"
    assert current.source_attempt_id == first_claim.attempt_id


@pytest.mark.asyncio
async def test_new_success_promotes_and_old_attempt_remains() -> None:
    first_claim = await claim_execution(store)
    first = await store.complete_execution_attempt(first_claim, successful_output("answer-v1"))
    second_claim = await claim_execution_rerun(store)
    second = await store.complete_execution_attempt(second_claim, successful_output("answer-v2"))

    assert second.id == first.id
    assert second.answer == "answer-v2"
    assert second.source_attempt_id == second_claim.attempt_id
    assert len(await store.list_attempts(user_id="user-a", work_item_id=second_claim.work_item_id)) == 2
```

Add equivalent RAGAS tests: a failed rerun preserves the old score; a successful compatible rerun replaces it; an incompatible signature does not enter the current aggregate.

- [x] **Step 2: Run promotion tests and verify failure**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_job_store.py tests/test_ragas_evaluator.py -k "promotion or preserves_previous or incompatible_signature" -q -p no:cacheprovider
```

Expected: FAIL because promotion outputs and signature selection are absent.

- [x] **Step 3: Implement deterministic evaluation signatures**

Canonicalize this dictionary with `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` and hash it with SHA-256:

```python
payload = {
    "campaign_result_id": result.id,
    "final_answer_hash": result.final_answer_hash,
    "evaluator_model": evaluator_model,
    "evaluator_config": evaluator_config,
    "metric_name": metric_name,
    "metric_version": metric_version,
    "context_policy_version": result.context_policy_version,
    "ground_truth_hash": ground_truth_hash,
    "context_metrics_enabled": context_metrics_enabled,
}
```

- [x] **Step 4: Implement atomic execution promotion**

Inside one connection transaction, mark the attempt succeeded, upsert the unique campaign-result unit while keeping its stable result ID, set `source_attempt_id`, update the work-item latest-success pointer, mark the job item succeeded, and recompute job counts. Store the full historical output in `evaluation_attempts.output_json`.

- [x] **Step 5: Implement atomic RAGAS promotion without zero fallbacks**

Upsert official scores by `(campaign_result_id, metric_name)` only after a successful attempt. Persist `source_attempt_id` and `evaluation_signature`. Remove code paths that delete valid official scores before replacement. A failed metric attempt calls `fail_attempt` and produces no score row.

- [x] **Step 6: Backfill legacy rows idempotently**

For each campaign with official rows lacking `source_attempt_id`, create one deterministic legacy job, stable work item, job item, and synthetic attempt. Link successful result/score rows to successful legacy attempts; represent legacy failed campaign results as failed attempts without promoting them. Re-running the migration must not create duplicates.

- [x] **Step 7: Run focused tests**

Run the Task 3 command without `-k`, then run:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_db_route_profile_alias.py tests/test_evaluation_analytics_api.py -q -p no:cacheprovider
```

Expected: all tests pass and existing analytics still read official tables.

- [x] **Step 8: Commit Task 3**

```powershell
git add evaluation/job_schemas.py evaluation/job_store.py evaluation/db.py tests/test_evaluation_job_store.py tests/test_ragas_evaluator.py
git commit -m "feat: promote successful evaluation attempts"
```

---

### Task 4: Single-Process Worker and Recovery

**Files:**
- Create: `evaluation/job_worker.py`
- Create: `tests/test_evaluation_job_worker.py`
- Modify: `core/app_factory.py`

**Interfaces:**
- Consumes: `EvaluationJobStore.claim_ready_items`, recovery and attempt transitions.
- Produces: `EvaluationJobWorker.start()`, `stop()`, `notify()`, and `run_once()`.
- Produces: `get_evaluation_job_worker() -> EvaluationJobWorker`.

- [x] **Step 1: Write worker recovery and wakeup tests**

```python
@pytest.mark.asyncio
async def test_start_recovers_running_attempt_and_executes_only_unresolved_work() -> None:
    await seed_running_attempt(store, logical_key="execution:Q1:naive:1:none")
    await seed_successful_work(store, logical_key="execution:Q2:naive:1:none")
    executor = AsyncMock()
    worker = EvaluationJobWorker(store=store, execution_handler=executor, ragas_handler=AsyncMock())

    await worker.start()
    worker.notify()
    await wait_until(lambda: executor.await_count == 1)
    await worker.stop()

    assert executor.await_args.args[0].logical_key == "execution:Q1:naive:1:none"


@pytest.mark.asyncio
async def test_stop_does_not_claim_new_work() -> None:
    worker = EvaluationJobWorker(store=store, execution_handler=AsyncMock(), ragas_handler=AsyncMock())
    await worker.start()
    await worker.stop()
    await seed_pending_work(store)
    worker.notify()
    assert await worker.run_once() == 0
```

- [x] **Step 2: Run tests and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_job_worker.py -q -p no:cacheprovider
```

Expected: FAIL because the worker does not exist.

- [x] **Step 3: Implement event-driven worker lifecycle**

The worker owns one loop task, one stop event, and one wake event. `start()` runs startup recovery before creating the loop. `notify()` sets the wake event. The loop drains ready work, then waits until notified or the next retry becomes due. `stop()` prevents claims, cancels active handler tasks cooperatively, marks unfinished attempts interrupted, and awaits loop shutdown.

Use an injectable clock and sleep/wait function so tests never block on real retry delays.

- [x] **Step 4: Implement bounded dispatch and heartbeats**

Use separate semaphores: execution maximum 4 and RAGAS-batch maximum 2. Start one heartbeat task per active attempt, updating no more often than every 15 seconds. Always cancel and drain heartbeat tasks in `finally`.

- [x] **Step 5: Wire the worker into FastAPI lifespan**

Replace direct `recover_inflight_campaigns()` startup with:

```python
worker = get_evaluation_job_worker()
await worker.start()
try:
    await _initialize_rag_components()
    await _warm_up_pdf_ocr()
    yield
finally:
    await worker.stop()
```

Keep database initialization before `worker.start()`.

- [x] **Step 6: Run focused lifecycle tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_job_worker.py tests/test_rag_startup.py -q -p no:cacheprovider
```

Expected: all tests pass with no leaked asyncio tasks.

- [x] **Step 7: Commit Task 4**

```powershell
git add evaluation/job_worker.py core/app_factory.py tests/test_evaluation_job_worker.py
git commit -m "feat: run durable evaluation worker"
```

---

### Task 5: Durable Dataset Execution and Campaign Facade

**Files:**
- Create: `evaluation/execution_worker.py`
- Create: `tests/test_evaluation_execution_worker.py`
- Modify: `evaluation/campaign_engine.py`
- Modify: `evaluation/campaign_schemas.py`
- Modify: `evaluation/db.py`
- Modify: `tests/test_campaign_engine.py`

**Interfaces:**
- Consumes: `ClaimedEvaluationWork`, existing `run_campaign_case`, and atomic execution promotion.
- Produces: `DatasetExecutionWorker.execute(claim: ClaimedEvaluationWork) -> None`.
- Produces: CampaignEngine methods that enqueue durable work and notify the worker.

- [x] **Step 1: Write execution checkpoint tests**

```python
@pytest.mark.asyncio
async def test_failed_unit_records_attempt_without_failed_official_result() -> None:
    runner = AsyncMock(side_effect=RuntimeError("temporary outage"))
    execution_worker = DatasetExecutionWorker(store=store, runner=runner)
    claim = await claim_seeded_execution(store)

    await execution_worker.execute(claim)

    attempts = await store.list_attempts(user_id="user-a", work_item_id=claim.work_item_id)
    assert attempts[-1].status == "failed"
    assert await CampaignResultRepository().list_for_campaign(
        user_id="user-a", campaign_id="cmp-1"
    ) == []


@pytest.mark.asyncio
async def test_campaign_creation_enqueues_units_without_process_local_task() -> None:
    response = await engine.create_and_start(user_id="user-a", name="durable", config=config)
    assert response.status == "pending"
    assert await store.count_job_items(campaign_id=response.campaign_id) == expected_unit_count
    worker.notify.assert_called_once_with()
```

- [x] **Step 2: Run focused tests and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py -k "durable or failed_unit" -q -p no:cacheprovider
```

Expected: FAIL because execution still uses campaign-owned tasks.

- [x] **Step 3: Implement immutable execution snapshots**

When enqueueing, store the complete test-case snapshot, mode, run/repeat numbers, condition ID, ablation flags, budget, and model configuration. The worker must execute from this snapshot so later edits to the golden dataset do not change a resumed job.

- [x] **Step 4: Move unit execution into `DatasetExecutionWorker`**

Reuse existing runner and result-normalization logic. On provider exception, classify and fail the attempt. On success, build `ExecutionAttemptOutput`, call `complete_execution_attempt`, then persist observability/trace details using the promoted stable result ID. Trace failure is logged but does not roll back an already committed official result.

- [x] **Step 5: Convert `CampaignEngine` into an enqueue/query facade**

`create_and_start` creates the campaign, builds stable `WorkItemSpec` rows from `_build_units`, creates the initial durable job, notifies the worker, and returns. Remove `_active_tasks`, `_register_active_task`, `_get_active_task`, `_drop_active_task`, and startup reconstruction that treats any result row as completed.

`cancel_campaign` delegates to the active durable job. `ensure_campaign_task` becomes a compatibility method that only notifies the worker and returns durable campaign state.

- [x] **Step 6: Add `completed_with_errors` and derive campaign state**

Add `COMPLETED_WITH_ERRORS` to backend status enums and terminal sets. Derive completed only when every required current job item has a compatible success; derive completed-with-errors when at least one usable result exists but unresolved failures remain; derive failed when no valid report can be produced.

- [x] **Step 7: Run campaign tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py tests/test_evaluation_pipeline.py -q -p no:cacheprovider
```

Expected: all pass; recovery tests assert durable job behavior rather than process-local tasks.

- [x] **Step 8: Commit Task 5**

```powershell
git add evaluation/execution_worker.py evaluation/campaign_engine.py evaluation/campaign_schemas.py evaluation/db.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py
git commit -m "feat: checkpoint evaluation dataset execution"
```

---

### Task 6: Durable RAGAS Metric Batches

**Files:**
- Create: `evaluation/ragas_worker.py`
- Create: `tests/test_evaluation_ragas_worker.py`
- Modify: `evaluation/ragas_evaluator.py`
- Modify: `evaluation/job_worker.py`
- Modify: `evaluation/campaign_engine.py`
- Modify: `tests/test_ragas_evaluator.py`

**Interfaces:**
- Produces: `RagasEvaluator.evaluate_metric_batch(metric_name, rows, evaluator_llm, evaluator_embeddings) -> list[float]`.
- Produces: `RagasBatchWorker.execute(claims: list[ClaimedEvaluationWork]) -> None`.
- Produces: creation of RAGAS work only for current successful official campaign results.

- [x] **Step 1: Write per-metric checkpoint and partial recovery tests**

```python
@pytest.mark.asyncio
async def test_completed_metric_checkpoints_survive_later_batch_failure() -> None:
    worker = RagasBatchWorker(store=store, evaluator=fake_evaluator)
    first_batch = await claim_metric_batch(store, metric="faithfulness", size=4)
    second_batch = await claim_metric_batch(store, metric="faithfulness", size=1)
    fake_evaluator.results = [[0.8, 0.7, 0.9, 0.6], RuntimeError("rate limited")]

    await worker.execute(first_batch)
    await worker.execute(second_batch)

    assert await store.count_official_scores(campaign_id="cmp-1") == 4
    assert (await store.list_attempts_for_job(second_batch[0].job_id))[-1].status == "failed"


@pytest.mark.asyncio
async def test_missing_dependency_is_failure_not_completed_empty_scores() -> None:
    evaluator = AsyncMock(side_effect=ModuleNotFoundError("ragas"))
    worker = RagasBatchWorker(store=store, evaluator=evaluator)
    await worker.execute(await claim_metric_batch(store, metric="faithfulness", size=1))
    assert await store.count_official_scores(campaign_id="cmp-1") == 0
    assert await store.count_failed_job_items(campaign_id="cmp-1") == 1
```

- [x] **Step 2: Run tests and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_ragas_worker.py tests/test_ragas_evaluator.py -k "checkpoint or missing_dependency" -q -p no:cacheprovider
```

Expected: FAIL because RAGAS persists only after campaign-wide completion.

- [x] **Step 3: Extract one-metric batch evaluation**

Refactor `_evaluate_metric_async` into the public typed `evaluate_metric_batch`. It must raise provider/dependency failures to the durable worker, return exactly one value per input row, and never convert exceptions into zero arrays. Keep aggregation methods unchanged except for filtering by current official signature.

- [x] **Step 4: Implement compatible grouping and bounded batching**

The worker groups claimed RAGAS items by metric name and evaluator signature, then chunks groups to configured batch size with maximum 4 by default. Outer parallelism is maximum 2. Each returned value calls `complete_ragas_attempt` independently. A whole-call exception fails each affected attempt using the same classified error.

- [x] **Step 5: Generate downstream RAGAS work idempotently**

After execution job terminalization, enumerate current official successful results and enabled metrics. Build one stable work item per `(result_id, metric, evaluation_signature)`. Do not generate work for failed execution attempts. Existing compatible successes satisfy the work unless the user explicitly requests a RAGAS rerun.

- [x] **Step 6: Remove destructive replacement and zero fallback paths**

Delete campaign-wide score accumulation and `replace_for_campaign` calls from the active execution path. Retain repository methods only for legacy compatibility tests until no caller remains. Missing or invalid current scores must stay absent and appear in warning counts.

- [x] **Step 7: Run RAGAS and campaign regression tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_ragas_worker.py tests/test_ragas_evaluator.py tests/test_campaign_engine.py tests/test_evaluation_analytics_api.py -q -p no:cacheprovider
```

Expected: all tests pass; no failed metric contributes `0.0`.

- [x] **Step 8: Commit Task 6**

```powershell
git add evaluation/ragas_worker.py evaluation/ragas_evaluator.py evaluation/job_worker.py evaluation/campaign_engine.py tests/test_evaluation_ragas_worker.py tests/test_ragas_evaluator.py
git commit -m "feat: checkpoint RAGAS metric batches"
```

---

### Task 7: Rerun, Job, Attempt, and Compatibility APIs

**Files:**
- Modify: `evaluation/job_schemas.py`
- Modify: `evaluation/campaign_engine.py`
- Modify: `evaluation/router.py`
- Modify: `tests/test_evaluation_api.py`

**Interfaces:**
- Produces: `POST /api/evaluation/campaigns/{campaign_id}/reruns`.
- Produces: `GET /api/evaluation/campaigns/{campaign_id}/jobs`.
- Produces: `GET /api/evaluation/jobs/{job_id}`.
- Produces: `POST /api/evaluation/jobs/{job_id}/cancel`.
- Produces: `GET /api/evaluation/work-items/{work_item_id}/attempts`.
- Preserves: `POST /api/evaluation/campaigns/{campaign_id}/evaluate` as a RAGAS-only delegate.

- [x] **Step 1: Write ownership and rerun scope API tests**

```python
def test_failed_only_rerun_returns_durable_job(client, auth_headers, seeded_campaign) -> None:
    response = client.post(
        f"/api/evaluation/campaigns/{seeded_campaign}/reruns",
        headers=auth_headers,
        json={"scope": "failed_only", "stages": "execution_and_ragas"},
    )
    assert response.status_code == 200
    assert response.json()["job_type"] == "rerun"


def test_attempt_history_enforces_owner(client, other_user_headers, seeded_work_item) -> None:
    response = client.get(
        f"/api/evaluation/work-items/{seeded_work_item}/attempts",
        headers=other_user_headers,
    )
    assert response.status_code == 404
```

Also test selected scope validation, RAGAS-only metric selection, cancellation, and old `/evaluate` delegation.

- [x] **Step 2: Run API tests and verify failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_api.py -k "rerun or job or attempt_history" -q -p no:cacheprovider
```

Expected: new routes return 404 or fail collection.

- [x] **Step 3: Implement engine rerun selection**

Resolve failed-only from current job-item state, selected from question IDs, and all from stable campaign work. For execution-and-RAGAS, enqueue execution work first; downstream RAGAS is created only after successful promotion. For RAGAS-only, select current successful official results and requested metrics.

- [x] **Step 4: Add thin authenticated routes**

Each route passes `user_id` into the engine/store and returns typed Pydantic responses. Unknown or cross-user IDs return the existing `AppError` 404. Route functions contain no SQL, task creation, or provider logic.

- [x] **Step 5: Delegate the legacy evaluate endpoint**

Translate omitted payload to `scope="all"`, selected question IDs to `scope="selected"`, and always set `stages="ragas"`. Return the campaign snapshot for response compatibility after creating the durable job.

- [x] **Step 6: Extend SSE terminal handling**

Add `completed_with_errors` to terminal status sets and emit `campaign_completed_with_errors`. The stream continues to read durable campaign snapshots; it never owns execution recovery.

- [x] **Step 7: Run backend API tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_api.py tests/test_evaluation_pipeline.py tests/test_evaluation_analytics_api.py -q -p no:cacheprovider
```

Expected: all pass.

- [x] **Step 8: Commit Task 7**

```powershell
git add evaluation/job_schemas.py evaluation/campaign_engine.py evaluation/router.py tests/test_evaluation_api.py
git commit -m "feat: expose durable evaluation reruns"
```

---

### Task 8: Evaluation Center Job and Attempt UI

**Files:**
- Create: `../Multimodal_RAG_System/src/components/evaluation/EvaluationJobPanel.tsx`
- Create: `../Multimodal_RAG_System/src/components/evaluation/EvaluationJobPanel.test.tsx`
- Modify: `../Multimodal_RAG_System/src/types/evaluation.ts`
- Modify: `../Multimodal_RAG_System/src/services/evaluationApi.ts`
- Modify: `../Multimodal_RAG_System/src/services/evaluationApi.test.ts`
- Modify: `../Multimodal_RAG_System/src/components/evaluation/EvaluationResults.tsx`
- Modify: `../Multimodal_RAG_System/src/components/evaluation/EvaluationResults.test.tsx`
- Modify: `../Multimodal_RAG_System/src/components/evaluation/CampaignRunner.tsx`
- Modify: `../Multimodal_RAG_System/src/components/evaluation/CampaignRunner.test.tsx`

**Interfaces:**
- Consumes: Task 7 REST and SSE contracts.
- Produces: `createCampaignRerun`, `listCampaignJobs`, `getEvaluationJob`, `cancelEvaluationJob`, `listWorkItemAttempts`.
- Produces: `EvaluationJobPanel` with valid/failed/retrying/interrupted/missing counts and safe attempt history.

- [x] **Step 1: Write API client contract tests**

```typescript
it('creates a failed-only durable rerun', async () => {
  mockedApi.post.mockResolvedValueOnce({ data: { id: 'job-1', job_type: 'rerun' } });
  await createCampaignRerun('cmp-1', {
    scope: 'failed_only',
    stages: 'execution_and_ragas',
    question_ids: [],
    metric_names: [],
  });
  expect(mockedApi.post).toHaveBeenCalledWith(
    '/api/evaluation/campaigns/cmp-1/reruns',
    expect.objectContaining({ scope: 'failed_only' }),
  );
});
```

- [x] **Step 2: Write UI behavior tests**

Test that `completed_with_errors` is terminal but warning-colored, failed attempts are excluded from displayed sample counts, retry-failed calls the unified endpoint, RAGAS-only does not request execution, attempt history displays safe messages, and polling refreshes an active job after reload.

- [x] **Step 3: Run frontend tests and verify failure**

```powershell
npx vitest run src/services/evaluationApi.test.ts src/components/evaluation/EvaluationJobPanel.test.tsx src/components/evaluation/EvaluationResults.test.tsx src/components/evaluation/CampaignRunner.test.tsx
```

Run from `D:\flutterserver\Multimodal_RAG_System`. Expected: FAIL because new types, client methods, and panel are absent.

- [x] **Step 4: Add exact TypeScript contracts**

Add `completed_with_errors` to `CampaignLifecycleStatus` and the SSE union. Add typed rerun request, job, job item counts, and attempt summary interfaces matching backend JSON field names. Do not use `Record<string, unknown>` for core status fields.

- [x] **Step 5: Implement API client methods**

Use the shared authenticated `api` instance for all REST calls. Extend the SSE parser with `campaign_completed_with_errors`. Keep the existing `evaluateCampaign` export as a compatibility wrapper until callers are migrated.

- [x] **Step 6: Implement `EvaluationJobPanel`**

Render one compact summary card with valid, failed, retry-wait, interrupted, and missing counts; job status; latest safe error; and actions for retry-failed, RAGAS-only, cancel, and attempt-history expansion. Disable conflicting actions while a non-terminal job targets the same campaign.

- [x] **Step 7: Integrate the panel and durable polling**

`EvaluationResults` loads campaign jobs for the selected campaign and refreshes metrics only after the selected job becomes terminal. Existing selected-question rerun uses `scope="selected", stages="ragas"`. `CampaignRunner` treats `completed_with_errors` as terminal and displays a warning rather than success styling.

- [x] **Step 8: Run focused frontend verification**

```powershell
npx vitest run src/services/evaluationApi.test.ts src/components/evaluation/EvaluationJobPanel.test.tsx src/components/evaluation/EvaluationResults.test.tsx src/components/evaluation/CampaignRunner.test.tsx
npm run lint:ci
npx tsc --noEmit
```

Expected: all commands exit 0.

- [x] **Step 9: Commit Task 8 in the frontend repository**

```powershell
git add src/types/evaluation.ts src/services/evaluationApi.ts src/services/evaluationApi.test.ts src/components/evaluation/EvaluationJobPanel.tsx src/components/evaluation/EvaluationJobPanel.test.tsx src/components/evaluation/EvaluationResults.tsx src/components/evaluation/EvaluationResults.test.tsx src/components/evaluation/CampaignRunner.tsx src/components/evaluation/CampaignRunner.test.tsx
git commit -m "feat: manage durable evaluation reruns"
```

---

### Task 9: Statistical Safety, Documentation, and Full Verification

**Files:**
- Modify: `evaluation/ragas_evaluator.py`
- Modify: `evaluation/analytics.py`
- Modify: `tests/test_ragas_evaluator.py`
- Modify: `tests/test_evaluation_analytics_api.py`
- Modify: `docs/BACKEND.md`
- Modify: `docs/RELIABILITY.md`
- Modify: `docs/generated/api-surface.md`
- Modify: `../Multimodal_RAG_System/docs/FRONTEND.md`
- Modify: `../Multimodal_RAG_System/docs/RELIABILITY.md`

**Interfaces:**
- Consumes: official success projections and signatures from Tasks 3 and 6.
- Produces: aggregate sample counts and missing/failed warning counts that cannot include failed attempts.

- [x] **Step 1: Write aggregate safety regressions**

```python
@pytest.mark.asyncio
async def test_aggregate_excludes_failed_attempts_and_reports_missing_sample() -> None:
    response = await evaluator.get_metrics(user_id="user-a", campaign=campaign)
    assert response.summary_by_mode["naive"].sample_count == 1
    assert response.evaluation_warnings.missing_metric_rows == 1
    assert response.rows[0].metric_values == {"faithfulness": 0.8}
    assert "answer_correctness" not in response.rows[0].metric_values
```

Add export and delta/ECR tests proving that missing metrics never become zero and incompatible evaluator/context signatures never mix.

- [x] **Step 2: Run safety tests and verify failures**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_ragas_evaluator.py tests/test_evaluation_analytics_api.py -k "missing or failed_attempt or incompatible" -q -p no:cacheprovider
```

Expected: at least the new warning/count assertions fail before the final projection filters are complete.

- [x] **Step 3: Finalize analytics filters and warnings**

Every aggregate reads official projections only. RAGAS joins require the active evaluation signature. Metric dictionaries omit missing values. Add explicit `missing_metric_rows`, `failed_work_items`, and `valid_sample_count` fields to warning/summary schemas and frontend types. Keep invalid legacy rows visible as warnings but exclude them from means.

- [x] **Step 4: Update backend and frontend documentation**

Document work units, attempt retention, official-result promotion, restart behavior, retry classification, default concurrency, rerun endpoints, `completed_with_errors`, statistical exclusion rules, and the unavoidable provider-response/checkpoint billing window. Update generated API documentation using the exact implemented route names.

- [x] **Step 5: Run all focused backend evaluation tests**

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; & '.\.venv\Scripts\python.exe' -m pytest tests/test_evaluation_job_schemas.py tests/test_evaluation_error_policy.py tests/test_evaluation_job_store.py tests/test_evaluation_job_worker.py tests/test_evaluation_execution_worker.py tests/test_evaluation_ragas_worker.py tests/test_campaign_engine.py tests/test_ragas_evaluator.py tests/test_evaluation_api.py tests/test_evaluation_pipeline.py tests/test_evaluation_analytics_api.py -q -p no:cacheprovider
```

Expected: all tests pass.

- [x] **Step 6: Run full backend verification**

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; $env:RUFF_NO_CACHE='true'; & '.\.venv\Scripts\python.exe' -m ruff check core/app_factory.py evaluation tests/test_evaluation_job_schemas.py tests/test_evaluation_error_policy.py tests/test_evaluation_job_store.py tests/test_evaluation_job_worker.py tests/test_evaluation_execution_worker.py tests/test_evaluation_ragas_worker.py tests/test_campaign_engine.py tests/test_ragas_evaluator.py tests/test_evaluation_api.py tests/test_evaluation_analytics_api.py
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; & '.\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

Expected: ruff exits 0 and pytest reports zero failures. Record exact passed/skipped counts; if legacy environment collection is blocked, report the exact module and exception rather than claiming full success.

- [x] **Step 7: Run full frontend verification**

From `D:\flutterserver\Multimodal_RAG_System`:

```powershell
npm run lint:ci
npx tsc --noEmit
npx vitest run
npm run build
```

Expected: all commands exit 0 and Vitest reports zero failed tests.

- [x] **Step 8: Commit backend documentation and safety changes**

```powershell
git add evaluation/ragas_evaluator.py evaluation/analytics.py tests/test_ragas_evaluator.py tests/test_evaluation_analytics_api.py docs/BACKEND.md docs/RELIABILITY.md docs/generated/api-surface.md
git commit -m "docs: document durable evaluation recovery"
```

- [x] **Step 9: Commit frontend documentation**

From `D:\flutterserver\Multimodal_RAG_System`:

```powershell
git add docs/FRONTEND.md docs/RELIABILITY.md
git commit -m "docs: explain evaluation rerun recovery"
```

---

## Execution Order and Review Gates

1. Tasks 1–3 establish durable types, persistence, and atomic promotion. Review the schema and legacy migration before starting worker integration.
2. Tasks 4–6 move runtime execution to the durable worker. Review restart, cancellation, and partial RAGAS checkpoint tests before exposing APIs.
3. Task 7 exposes backend contracts. Confirm OpenAPI and ownership behavior before frontend work.
4. Task 8 updates the frontend in its separate repository.
5. Task 9 is the release gate; do not claim completion without exact full-suite results.

## Closeout Evidence (2026-07-15)

- Cold-start recovery now constructs the production campaign engine before
  checking the process worker, so a clean process configures handlers before
  recovery. Embedded isolated workers are recovered without an idle-start race
  and are stopped when their event loop is still running.
- Generated API documentation is synchronized: the durable route is
  `POST /api/evaluation/campaigns/{campaign_id}/reruns`, and `openapi.json`
  contains all durable job, attempt, and rerun routes (85 paths; all six
  required route families present).
- Focused backend evaluation command from Task 9 Step 5: 127 passed; the
  additional startup/lifecycle regression file raises the final focused run
  to 132 passed. Ruff checks for all changed backend/runtime/test files:
  passed.
- Backend suite was executed from the final 124-file, 783-test collection in
  five bounded groups because the local runner interrupts one process at about
  60 seconds. Group totals: 247 + 122 + 143 + 127 + 144 = 783 passed.
- Frontend verification: 72 Vitest files / 287 tests passed; lint, TypeScript
  check, and production build passed.

## Rollback Strategy

- All database changes are additive; old official projection tables remain authoritative during migration.
- Keep the compatibility `/evaluate` endpoint and existing result/metric reads until durable paths pass full verification.
- If runtime migration must be rolled back before release, revert the runtime enqueue/worker commits while leaving additive ledger tables and history intact; the existing official projection tables remain readable.
- Never delete attempt history as part of rollback.
