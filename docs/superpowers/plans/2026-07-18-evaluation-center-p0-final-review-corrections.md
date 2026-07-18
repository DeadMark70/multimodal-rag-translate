# Evaluation Center P0 Final Review Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the four remaining strict-research defects found by the cross-repository final review and close the P0 plan with durable retry metadata and renewed verification.

**Architecture:** Backend analytics will fail closed for wholly unmeasured token usage and select one campaign-level evaluator cohort per metric. RAGAS retry counts will be persisted on accounting scopes through an async retry callback instead of inferred from scope keys. The frontend will consume nullable retry metadata and render the backend's explicit unclassified phase value.

**Tech Stack:** Python 3.11, Pydantic v2, aiosqlite, Tenacity, FastAPI/TestClient, pytest, React 18, TypeScript, Chakra UI, Vitest.

## Global Constraints

- Missing, failed, partial, legacy, unpriced, and historically unknown values never become synthetic zero.
- Per-result `evaluation_signature` remains the currentness/idempotency identity; evaluator model, metric version, and `compatibility_signature` define comparison cohorts.
- Benchmark cost uses only official successful execution attempts; operational cost includes all attempts; RAGAS usage remains evaluation overhead.
- Existing research-summary URLs and compatibility endpoints remain available.
- Historical retry metadata is not backfilled or estimated.
- Normal chat creates no evaluation accounting rows.

---

## File Structure

Backend repository (`D:\flutterserver\pdftopng`):

- `evaluation/research_analytics.py`: canonical evaluator cohort selection, strict token projection, per-result quality sample classification, retry aggregation.
- `evaluation/accounting_schemas.py`: nullable durable scope retry counter and nullable overhead retry response.
- `evaluation/accounting_store.py`: retry-counter persistence and atomic increment.
- `evaluation/db.py`: fresh schema plus additive nullable migration.
- `evaluation/retry.py`: optional awaited retry callback.
- `evaluation/ragas_worker.py`: connect the active RAGAS accounting scope to the retry callback.
- `tests/test_evaluation_research_analytics.py`: token, cohort, failed-sample, and historical retry-summary regressions.
- `tests/test_evaluation_accounting_schema.py`: nullable retry field contract.
- `tests/test_evaluation_accounting_store.py`: retry round-trip and idempotent increment behavior.
- `tests/test_evaluation_ragas_worker.py`: successful, exhausted, and callback-recorded retry paths.
- `tests/test_evaluation_research_api.py`: JSON null serialization for unknown historical retry metadata.
- `docs/BACKEND.md`: measured-subtotal, canonical-cohort, and retry-counter semantics.

Frontend repository (`D:\flutterserver\Multimodal_RAG_System`):

- `src/types/evaluation.ts`: nullable `evaluation_overhead.retry_count` mirror.
- `src/components/evaluation/TokenBreakdownChart.tsx`: direct unclassified rendering.
- `src/components/evaluation/TokenBreakdownChart.test.tsx`: explicit/nonexistent unclassified and missing-category tests.
- `src/components/evaluation/researchSummaryFixtures.ts`: strict retry and phase fixtures.
- `docs/design-docs/evaluation-center.md`: corrected phase and measured-subtotal semantics.
- `docs/product-specs/evaluation-results-and-traces.md`: nullable historical retry semantics.

---

### Task 1: Strict Tokens, Campaign Cohorts, and Per-Result Quality Counts

**Files:**
- Modify: `evaluation/research_analytics.py`
- Modify: `tests/test_evaluation_research_analytics.py`
- Modify: `docs/BACKEND.md`

**Interfaces:**
- Consumes: `_evaluator_identity(row) -> tuple[str, str, str]`, official current scores, `AccountingScope` and `UsageEvent` lists.
- Produces: `_canonical_identities_by_metric(...) -> dict[str, tuple[str, str, str]]` and `_quality_for_results(..., canonical_identities)`.

- [ ] **Step 1: Add failing tests for wholly missing and mixed usage**

Add fixtures with one official completed scope whose sole event has `usage_status="missing"`, and a second scope with one balanced measured event plus one missing event.

```python
missing = _usage_event(
    scope_id="scope-missing",
    usage_status="missing",
    reconciliation_status="unavailable",
    input_tokens=0,
    output_text_tokens=0,
    reasoning_tokens=0,
    other_tokens=0,
)
summary = await service.get_summary(user_id="user-1", campaign_id="missing-usage")
assert summary.tokens.input_tokens is None
assert summary.tokens.output_text_tokens is None
assert summary.tokens.reasoning_tokens is None
assert summary.tokens.other_tokens is None
assert summary.tokens.total_tokens is None
assert summary.tokens.by_phase == {}

mixed = await service.get_summary(user_id="user-1", campaign_id="mixed-usage")
assert mixed.tokens.accounting_status == "partial"
assert mixed.tokens.input_tokens == 10
assert mixed.tokens.total_tokens is None
```

- [ ] **Step 2: Run the token tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_analytics.py -k "missing_usage or mixed_usage" -q
```

Expected: the wholly missing fixture returns category zeroes before the fix.

- [ ] **Step 3: Make `_tokens()` distinguish measured events from placeholders**

Use measured balanced events for category and phase subtotals. Return nullable categories when that list is empty.

```python
measured = [
    event
    for event in events
    if event.usage_status == "measured"
    and event.reconciliation_status == "balanced"
]
if not measured:
    return TokenBreakdown(
        input_tokens=None,
        output_text_tokens=None,
        reasoning_tokens=None,
        other_tokens=None,
        total_tokens=None,
        by_phase={},
        accounting_status="partial",
        phase_attribution_status="not_available",
    )
```

For mixed usage, sum only `measured`, keep `total_tokens=None`, and retain `accounting_status="partial"`.

- [ ] **Step 4: Add failing cross-mode evaluator cohort tests**

Seed Naive with `judge-v1/policy-a` and Graph with `judge-v2/policy-b`, with each mode internally complete for all three primary metrics. Also seed a control campaign where both modes share `judge-v1/policy-a`.

```python
summary = await service.get_summary(user_id="user-1", campaign_id="mixed-cohorts")
by_mode = {mode.mode: mode for mode in summary.modes}
assert sum(mode.comparable for mode in summary.modes) <= 1
assert any(
    "evaluator_metadata_mismatch" in mode.not_comparable_reasons
    for mode in summary.modes
)

control = await service.get_summary(user_id="user-1", campaign_id="shared-cohort")
assert all(mode.comparable for mode in control.modes)
```

- [ ] **Step 5: Run the cohort tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_analytics.py -k "cohort" -q
```

Expected: both differently judged modes are currently comparable.

- [ ] **Step 6: Select one canonical identity per metric at campaign level**

Add a deterministic selector over official current scores:

```python
def _canonical_identities_by_metric(results, scores):
    attempts = {
        result.id: result.source_attempt_id
        for result in results
        if result.source_attempt_id
    }
    grouped = defaultdict(lambda: defaultdict(set))
    for row in scores:
        result_id = row["campaign_result_id"]
        if (
            result_id in attempts
            and row.get("source_attempt_id") == attempts[result_id]
        ):
            grouped[row["metric_name"]][_evaluator_identity(row)].add(result_id)
    return {
        metric: min(
            identities,
            key=lambda identity: (-len(identities[identity]), identity),
        )
        for metric, identities in grouped.items()
        if identities
    }
```

Compute this once in `get_summary()`, pass it into campaign and mode quality aggregation, and mark a mode with current noncanonical scores as `evaluator_metadata_mismatch`.

- [ ] **Step 7: Add failing mixed valid/terminal-failed sample test**

```python
observation = summary.quality["faithfulness"]
assert observation.valid_samples == 4
assert observation.failed_samples == 1
assert observation.missing_samples == 0
assert observation.status == "partial"
```

Seed five official results, four canonical scores, and one terminal failed `ragas_batch` target for the fifth result.

- [ ] **Step 8: Classify quality state per result**

Build per-result terminal/running work status from RAGAS scope targets. Count each result exactly once as valid, evaluating, failed, or missing. `missing_samples` excludes terminal failures; `failed_samples` includes them even when other values exist.

- [ ] **Step 9: Run Task 1 verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/research_analytics.py tests/test_evaluation_research_analytics.py
.\.venv\Scripts\python.exe -m ruff format --check evaluation/research_analytics.py tests/test_evaluation_research_analytics.py
```

Expected: all tests and checks pass.

- [ ] **Step 10: Commit Task 1**

```powershell
git add evaluation/research_analytics.py tests/test_evaluation_research_analytics.py docs/BACKEND.md
git commit -m "fix(evaluation): enforce strict research cohorts"
```

---

### Task 2: Durable RAGAS Retry Counters

**Files:**
- Modify: `evaluation/accounting_schemas.py`
- Modify: `evaluation/accounting_store.py`
- Modify: `evaluation/db.py`
- Modify: `evaluation/retry.py`
- Modify: `evaluation/ragas_worker.py`
- Modify: `evaluation/research_analytics.py`
- Modify: `tests/test_evaluation_accounting_schema.py`
- Modify: `tests/test_evaluation_accounting_store.py`
- Modify: `tests/test_evaluation_ragas_worker.py`
- Modify: `tests/test_evaluation_research_analytics.py`
- Modify: `tests/test_evaluation_research_api.py`
- Modify: `docs/BACKEND.md`

**Interfaces:**
- Produces: `AccountingScope.retry_count: int | None`, `EvaluationAccountingStore.increment_scope_retry(scope_id)`, optional `run_with_retry(..., on_retry=...)`, and `EvaluationOverheadSummary.retry_count: int | None`.
- Consumes: the active `ragas_batch` `scope_id` created by `start_ragas_batch_scope()`.

- [ ] **Step 1: Add failing schema, migration, and store tests**

```python
scope = await store.start_scope(ragas_scope_start)
assert scope.retry_count == 0

await store.increment_scope_retry(scope.scope_id)
await store.increment_scope_retry(scope.scope_id)
assert (await store.get_scope(scope.scope_id)).retry_count == 2

await run_migrations_on_legacy_accounting_db()
legacy = await store.get_scope("legacy-ragas-scope")
assert legacy.retry_count is None
```

- [ ] **Step 2: Run persistence tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_schema.py tests/test_evaluation_accounting_store.py -k "retry_count" -q
```

Expected: `AccountingScope` and the database do not yet expose the field.

- [ ] **Step 3: Add nullable durable retry storage**

Add:

```python
class AccountingScope(BaseModel):
    retry_count: int | None = Field(default=None, ge=0)
```

Fresh DDL includes `retry_count INTEGER DEFAULT 0 CHECK (retry_count >= 0)`. The migration uses only:

```sql
ALTER TABLE evaluation_accounting_scopes ADD COLUMN retry_count INTEGER
```

so historical rows stay `NULL`. `start_scope()` explicitly inserts `0` for fresh scopes. `_scope_from_row()` reads the nullable value.

- [ ] **Step 4: Add atomic increment**

```python
async def increment_scope_retry(self, scope_id: str) -> AccountingScope:
    now = _utc_now_iso()
    await init_db()
    async with connect_db() as connection:
        cursor = await connection.execute(
            """UPDATE evaluation_accounting_scopes
               SET retry_count = COALESCE(retry_count, 0) + 1, updated_at = ?
               WHERE scope_id = ? AND scope_type = 'ragas_batch' AND status = 'running'""",
            (now, scope_id),
        )
        if cursor.rowcount != 1:
            raise AccountingScopeMismatchError(
                "Retry counter requires one running RAGAS accounting scope"
            )
        await connection.commit()
    return await self.get_scope(scope_id)
```

- [ ] **Step 5: Add failing retry-callback tests**

Cover a first transient failure followed by success, five exhausted transient attempts, and a transient failure after a failed usage callback event was persisted.

```python
assert await worker.process_ready(...) == 1
assert (await accounting.get_scope(scope_id)).retry_count == 1

with pytest.raises(google_exceptions.ServiceUnavailable):
    await worker.process_ready(...)
assert (await accounting.get_scope(scope_id)).retry_count == 4
```

- [ ] **Step 6: Add an awaited `on_retry` hook to `run_with_retry()`**

```python
async def run_with_retry(
    operation: Callable[..., Awaitable[_T]],
    *args,
    on_retry: Callable[[int, BaseException], Awaitable[None]] | None = None,
    **kwargs,
) -> _T:
    async def before_sleep(state) -> None:
        error = state.outcome.exception() if state.outcome else RuntimeError("unknown")
        logger.warning("Evaluation retry %s after %s", state.attempt_number, error)
        if on_retry is not None:
            await on_retry(state.attempt_number, error)
```

Pass `before_sleep=before_sleep` to `AsyncRetrying`. Existing callers that do not supply `on_retry` retain their behavior.

- [ ] **Step 7: Connect the RAGAS scope counter**

Inside the active `llm_accounting_scope` block:

```python
async def record_retry(_attempt_number: int, _error: BaseException) -> None:
    await self._accounting_store.increment_scope_retry(scope.scope_id)

values = await run_with_retry(
    self._evaluator.evaluate_metric_batch,
    metric_name,
    rows,
    self._evaluator_llm,
    self._evaluator_embeddings,
    on_retry=record_retry,
)
```

- [ ] **Step 8: Make the API retry count nullable and durable**

Change backend and frontend contract in their respective tasks to `int | None` / `number | null`. In analytics:

```python
retry_count = (
    None
    if any(scope.retry_count is None for scope in overhead_scopes)
    else sum(scope.retry_count or 0 for scope in overhead_scopes)
)
```

When the value is unknown, append `ResearchWarning(code="unknown_ragas_retry_count", ...)`. With no RAGAS scopes, return the known value `0`.

- [ ] **Step 9: Run Task 2 verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_schema.py tests/test_evaluation_accounting_store.py tests/test_evaluation_ragas_worker.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/accounting_schemas.py evaluation/accounting_store.py evaluation/db.py evaluation/retry.py evaluation/ragas_worker.py evaluation/research_analytics.py tests/test_evaluation_accounting_schema.py tests/test_evaluation_accounting_store.py tests/test_evaluation_ragas_worker.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py
```

Expected: all tests and lint checks pass.

- [ ] **Step 10: Commit Task 2**

```powershell
git add evaluation/accounting_schemas.py evaluation/accounting_store.py evaluation/db.py evaluation/retry.py evaluation/ragas_worker.py evaluation/research_analytics.py tests/test_evaluation_accounting_schema.py tests/test_evaluation_accounting_store.py tests/test_evaluation_ragas_worker.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py docs/BACKEND.md
git commit -m "fix(evaluation): persist RAGAS retry counts"
```

---

### Task 3: Frontend Unclassified and Nullable Retry Rendering

**Files:**
- Modify frontend: `src/types/evaluation.ts`
- Modify frontend: `src/components/evaluation/TokenBreakdownChart.tsx`
- Modify frontend: `src/components/evaluation/TokenBreakdownChart.test.tsx`
- Modify frontend: `src/components/evaluation/researchSummaryFixtures.ts`
- Modify frontend: `docs/design-docs/evaluation-center.md`
- Modify frontend: `docs/product-specs/evaluation-results-and-traces.md`

**Interfaces:**
- Consumes: `ResearchTokenBreakdown.by_phase`, `phase_attribution_status`, and nullable `EvaluationOverheadSummary.retry_count`.
- Produces: `unclassified(breakdown) -> number | null` with no subtraction fallback.

- [ ] **Step 1: Add failing explicit and absent-unclassified tests**

```tsx
it('renders the explicit unclassified phase value', () => {
  renderChart({
    total_tokens: 100,
    by_phase: { answer_generation: 80, unclassified: 20 },
    phase_attribution_status: 'partial',
  });
  expect(screen.getByText('Unclassified: 20')).toBeInTheDocument();
});

it.each([
  ['complete', 'Unclassified: 0'],
  ['partial', 'Unclassified: N/A'],
  ['not_available', 'Unclassified: N/A'],
] as const)('renders absent unclassified for %s attribution', (status, expected) => {
  renderChart({ by_phase: { answer_generation: 100 }, phase_attribution_status: status });
  expect(screen.getByText(expected)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run frontend test and confirm RED**

Run:

```powershell
npm test -- --run src/components/evaluation/TokenBreakdownChart.test.tsx
```

Expected: the explicit value is rendered as zero before the fix.

- [ ] **Step 3: Read the explicit phase value**

```typescript
function unclassified(breakdown: ResearchTokenBreakdown): number | null {
  if (Object.prototype.hasOwnProperty.call(breakdown.by_phase, 'unclassified')) {
    return breakdown.by_phase.unclassified;
  }
  return breakdown.phase_attribution_status === 'complete' ? 0 : null;
}
```

- [ ] **Step 4: Mirror nullable retry metadata**

```typescript
export interface EvaluationOverheadSummary {
  // existing fields remain unchanged
  retry_count: number | null;
}
```

Update strict fixtures to include both a measured retry count and an unknown historical `null`. Document that partial token category values are measured subtotals and that historical retry count can be unknown.

- [ ] **Step 5: Run Task 3 verification**

Run:

```powershell
npm test -- --run src/components/evaluation/TokenBreakdownChart.test.tsx src/components/evaluation/CampaignOverviewTab.test.tsx src/pages/EvaluationCenter.ui.test.tsx src/services/evaluationApi.test.ts
npm run lint:ci
npm run build
```

Expected: all tests, lint, and build pass.

- [ ] **Step 6: Commit Task 3 in the frontend repository**

```powershell
git add src/types/evaluation.ts src/components/evaluation/TokenBreakdownChart.tsx src/components/evaluation/TokenBreakdownChart.test.tsx src/components/evaluation/researchSummaryFixtures.ts docs/design-docs/evaluation-center.md docs/product-specs/evaluation-results-and-traces.md
git commit -m "fix(evaluation): render strict phase attribution"
```

---

### Task 4: Ledger, Full Verification, and Renewed Final Review

**Files:**
- Modify local ledger: `.superpowers/sdd/progress.md`
- Modify local reports: `.superpowers/sdd/research-accounting-task-11-report.md`

**Interfaces:**
- Consumes: approved commits and reviewer reports from Tasks 1–3.
- Produces: final completion evidence for the original P0 plan.

- [ ] **Step 1: Run focused backend verification**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_schema.py tests/test_evaluation_accounting_store.py tests/test_evaluation_token_normalizers.py tests/test_llm_usage_callback.py tests/test_evaluation_accounting_runtime.py tests/test_evaluation_phase_attribution.py tests/test_evaluation_ragas_worker.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_research_end_to_end.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run full backend verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: zero failures; record warnings separately.

- [ ] **Step 3: Run full frontend verification**

```powershell
npm test -- --run
npm run lint:ci
npm run build
```

Expected: zero failed test files, zero lint warnings, and a successful build.

- [ ] **Step 4: Verify changed-file lint and forbidden fallbacks**

Backend:

```powershell
$files = @(git diff --name-only --diff-filter=ACMR ac4dbec..HEAD -- '*.py')
.\.venv\Scripts\python.exe -m ruff check $files
```

Frontend:

```powershell
rg -n "1 - unsupported|unsupported_claim_ratio_mean|p50Ms: overview\.avg|p95Ms: overview\.avg|completionTokens: 0|reasoningTokens: 0|total_cost_usd \?\? 0|total_tokens - Object\.values" src/pages/EvaluationCenter.tsx src/components/evaluation
```

Expected: Ruff passes and the frontend search has no matches.

- [ ] **Step 5: Update the durable ledger**

Record Tasks 1–11, the final correction commits, focused and full verification counts, existing warning counts, the 14-file pre-existing whole-file format debt, and both repository clean states. Mark the original P0 plan complete only after the renewed final reviewer approves it.

- [ ] **Step 6: Request independent reviews**

Use one fresh reviewer for backend Tasks 1–2, one fresh reviewer for frontend Task 3, then one fresh cross-repository reviewer over the complete original ranges plus correction commits. Fix every Critical or Important finding and repeat the relevant review.

- [ ] **Step 7: Confirm both repositories are clean**

```powershell
git status --short
```

Run in both repositories. Expected: no tracked or untracked task artifacts.

