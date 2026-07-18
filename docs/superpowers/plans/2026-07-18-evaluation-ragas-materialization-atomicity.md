# Evaluation RAGAS Materialization Atomicity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make newly materialized RAGAS work and its campaign evaluation transition one atomic database commit so a fast worker cannot publish `completed` with stale `0/N` counters.

**Architecture:** `EvaluationJobStore.create_job_with_items()` receives an optional RAGAS target-result count and updates the owned campaign inside the same `BEGIN IMMEDIATE` transaction that inserts the pending work. `ensure_ragas_work()` supplies the distinct result count, and callers stop issuing a later duplicate `mark_evaluating()` write. Worker notification remains post-commit.

**Tech Stack:** Python 3.13, asyncio, aiosqlite/SQLite, FastAPI campaign services, pytest, Ruff.

## Global Constraints

- Pending RAGAS items and the campaign `evaluating` transition must commit atomically.
- The transition uses both `campaign_id` and `user_id`; failure to update exactly one non-cancelled campaign rolls back all new job/work rows.
- `evaluation_total_units` is the number of distinct targeted campaign results, not the number of metric work items.
- Worker notification occurs only after the successful commit and notifier exceptions remain visible.
- `ensure_ragas_work()` retains its integer return contract and idempotent behavior.
- No post-materialization caller may reset counters with a second `mark_evaluating()` write.
- Do not alter retry persistence, score promotion, evaluator cohort selection, or frontend contracts.
- Follow `agent.md`; add type hints and keep changes incremental.

---

### Task 1: Atomically Materialize RAGAS Work and Campaign State

**Files:**
- Modify: `evaluation/job_store.py`
- Modify: `evaluation/execution_worker.py`
- Modify: `evaluation/campaign_engine.py`
- Modify: `tests/test_evaluation_job_store.py`
- Modify: `tests/test_evaluation_execution_worker.py`
- Modify: `tests/test_campaign_engine.py`
- Modify: `docs/BACKEND.md`

**Interfaces:**
- Extend: `EvaluationJobStore.create_job_with_items(..., ragas_evaluation_total_units: int | None = None) -> EvaluationJob`.
- Preserve: `EvaluationJobStore.ensure_ragas_work(...) -> int`.
- Consume: `WorkItemSpec.input_snapshot["campaign_result_id"]` for distinct result totals.

- [ ] **Step 1: Add a failing post-commit visibility test**

Create a campaign with one completed official result and an `EvaluationJobStore`
whose synchronous `on_job_created` callback opens the configured SQLite database
and records the campaign plus pending RAGAS rows visible at notification time.

```python
observed: list[tuple[str, int, int, int]] = []

def observe_committed_state() -> None:
    with sqlite3.connect(db_path) as connection:
        campaign = connection.execute(
            "SELECT status, evaluation_completed_units, evaluation_total_units "
            "FROM campaigns WHERE id = ? AND user_id = ?",
            (campaign_id, user_id),
        ).fetchone()
        pending = connection.execute(
            "SELECT COUNT(*) FROM evaluation_job_items WHERE status = 'pending'"
        ).fetchone()[0]
    observed.append((*campaign, pending))

created = await EvaluationJobStore(on_job_created=observe_committed_state).ensure_ragas_work(
    user_id=user_id,
    campaign_id=campaign_id,
    evaluator_model="judge-v1",
    evaluator_config={},
    enabled_metrics=("faithfulness", "answer_correctness"),
)
assert created == 2
assert observed == [("evaluating", 0, 1, 2)]
```

- [ ] **Step 2: Add failing rollback and orchestration tests**

Test a direct RAGAS-aware `create_job_with_items()` call against a missing or
wrongly owned campaign. It must raise and leave no job/item rows.

```python
with pytest.raises(ValueError, match="owned campaign"):
    await store.create_job_with_items(
        user_id="wrong-user",
        campaign_id=campaign_id,
        job_type=EvaluationJobType.INITIAL,
        selection={"stage": "ragas"},
        config_snapshot={},
        items=(ragas_spec,),
        ragas_evaluation_total_units=1,
    )
assert await _job_count(db_path) == 0
assert await _job_item_count(db_path) == 0
```

Add an execution-worker orchestration regression using the existing fakes or
`AsyncMock`: when `ensure_ragas_work()` returns a positive count, `_derive_campaign_state()`
must notify work but must not call `campaign_repository.mark_evaluating()` after
materialization.

```python
await worker._derive_campaign_state(claim)
campaign_repository.mark_evaluating.assert_not_awaited()
notify.assert_called_once()
```

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_job_store.py -k "atomic or notification or distinct_result" -q
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_execution_worker.py -k "materialized_ragas" -q
```

Expected: notification observes the pre-evaluation campaign state, the new
keyword is unsupported or ownership does not roll back atomically, and the
execution worker still calls the late `mark_evaluating()` transition.

- [ ] **Step 4: Add the optional atomic campaign transition**

Extend the job creation signature:

```python
async def create_job_with_items(
    self,
    *,
    user_id: str,
    campaign_id: str,
    job_type: EvaluationJobType,
    selection: dict[str, Any],
    config_snapshot: dict[str, Any],
    items: Sequence[WorkItemSpec],
    ragas_evaluation_total_units: int | None = None,
) -> EvaluationJob:
```

Inside the existing `BEGIN IMMEDIATE` transaction, after inserting all job
items and before `commit()`, perform the guarded transition only when the new
argument is not `None`:

```python
cursor = await connection.execute(
    """
    UPDATE campaigns
    SET status = 'evaluating', phase = 'evaluation',
        evaluation_completed_units = 0, evaluation_total_units = ?,
        error_message = NULL, completed_at = NULL, updated_at = ?
    WHERE id = ? AND user_id = ? AND cancel_requested = 0
    """,
    (ragas_evaluation_total_units, now, campaign_id, user_id),
)
if cursor.rowcount != 1:
    raise ValueError("RAGAS work requires exactly one owned campaign")
```

The existing `except BaseException` rollback and post-commit notifier remain
the transaction/error boundary.

- [ ] **Step 5: Supply the distinct target-result total**

After building non-empty `specs`, derive the result total from the actual work
being created and pass it into `create_job_with_items()`:

```python
target_result_ids = {
    str(spec.input_snapshot["campaign_result_id"])
    for spec in specs
    if spec.input_snapshot.get("campaign_result_id")
}
if not target_result_ids:
    raise ValueError("RAGAS work requires campaign result targets")

await self.create_job_with_items(
    # existing arguments
    items=specs,
    ragas_evaluation_total_units=len(target_result_ids),
)
```

Keep the return value `len(specs)` so compatibility callers still receive the
number of created metric items.

- [ ] **Step 6: Remove late duplicate transitions**

In `DatasetExecutionWorker._derive_campaign_state()` and the two
`CampaignEngine` paths that call `ensure_ragas_work()`, remove only the
post-success `mark_evaluating()` calls. Preserve notification, no-work errors,
recovery completion, and legacy direct-evaluator paths.

The execution-worker positive branch becomes:

```python
if created and self._notify is not None:
    self._notify()
```

Do not remove `mark_evaluating()` from legacy paths that do not materialize
work through `ensure_ragas_work()`.

- [ ] **Step 7: Verify GREEN and regression stability**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_job_store.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py::test_production_engine_wires_ragas_accounting_scope_and_event -q
```

Then run the previously flaky production test five times in separate pytest
processes; all five must report one pass and `completed + 1/1` semantics.

- [ ] **Step 8: Update documentation and run task checks**

Document in `docs/BACKEND.md` that RAGAS job visibility and the evaluating
campaign transition share one commit, and that totals count distinct results.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_job_store.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py tests/test_evaluation_ragas_worker.py tests/test_evaluation_research_end_to_end.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/job_store.py evaluation/execution_worker.py evaluation/campaign_engine.py tests/test_evaluation_job_store.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py
.\.venv\Scripts\python.exe -m ruff format --check evaluation/job_store.py evaluation/execution_worker.py evaluation/campaign_engine.py tests/test_evaluation_job_store.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py
git diff --check
```

- [ ] **Step 9: Commit**

```powershell
git add evaluation/job_store.py evaluation/execution_worker.py evaluation/campaign_engine.py tests/test_evaluation_job_store.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py docs/BACKEND.md
git commit -m "fix(evaluation): atomically activate RAGAS work"
```

---

### Task 2: Resume Final Verification and Review

**Files:**
- Modify local ledger: `.superpowers/sdd/progress.md`
- Modify local report: `.superpowers/sdd/research-accounting-task-11-report.md`

**Interfaces:**
- Consumes the approved Task 1 commit plus P0 correction commits `fc5909b`, `47f6482`, and frontend `93565f1`.
- Produces fresh completion evidence and the renewed cross-repository verdict.

- [ ] **Step 1: Re-run backend focused and complete verification**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_schema.py tests/test_evaluation_accounting_store.py tests/test_evaluation_token_normalizers.py tests/test_llm_usage_callback.py tests/test_evaluation_accounting_runtime.py tests/test_evaluation_phase_attribution.py tests/test_evaluation_ragas_worker.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_research_end_to_end.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Both commands must exit zero. Record exact passed and warning counts.

- [ ] **Step 2: Re-run complete frontend verification**

```powershell
npm test -- --run
npm run lint:ci
npm run build
```

Record exact test-file/test counts, lint exit status, build exit status, and
pre-existing warning/advisory output separately.

- [ ] **Step 3: Run changed-file and fallback checks**

Run Ruff on every changed Python file from `ac4dbec..HEAD`, run the existing
whole-file format-debt inventory without modifying unrelated files, and verify
the forbidden frontend fallback search remains empty.

- [ ] **Step 4: Update evidence and request final Terra review**

Record all original P0 Tasks 1–11, correction commits, the atomicity fix,
fresh verification, known warning/format debt, and clean repository states in
the local ledger/report. Dispatch a fresh `gpt-5.6-terra` reviewer over the
complete backend and frontend ranges. Fix every Critical or Important finding
and repeat the affected verification and review.

- [ ] **Step 5: Confirm clean repositories**

```powershell
git status --short
```

Both repositories must have no tracked or untracked task artifacts before the
original P0 plan is marked complete.
