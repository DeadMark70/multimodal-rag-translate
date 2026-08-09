# Evaluation Center Load Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Evaluation Center's initial loading blocker and make its database reads bounded as campaign size grows, while retaining historical-data compatibility.

**Architecture:** Treat a missing `benchmark_id` as an explicit `not_applicable` Release Metrics state. Replace per-run release and accounting reads with campaign-scoped snapshots, use dedicated bounded result projections for aggregate pages, and reserve raw answers and traces for detail/export reads only. Add additive schema fields and indexes so old databases retain readable fallbacks.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, aiosqlite, pytest; React, TypeScript, Chakra UI, Vitest.

## Global Constraints

- Missing `benchmark_id` is normal: the Release Metrics route returns HTTP 200 with `availability="not_applicable"` and `not_applicable_reason="benchmark_not_configured"`.
- Do not change scoring, release-gate semantics, benchmark identity rules, or existing full-result/detail/export contracts.
- Analytics projections must not select or deserialize `answer`, `contexts_json`, or `ground_truth` unless the route is explicit detail/export.
- No Release Metrics repository call may scale with the number of runs in a campaign; campaign-scoped reads are constant in query count.
- Historical trace rows remain readable; summary fields are additive and use a safe one-row detail fallback only when a selected legacy trace lacks a stored summary.
- WAL, foreign keys, busy timeout, transaction semantics, and isolated test database paths must remain intact.
- The answer payload limit is **1,048,576 UTF-8 bytes**. Oversized provider output must become an explicit failed campaign result with error code `EVALUATION_ANSWER_TOO_LARGE`; it must never be silently truncated or reported as a completed/scored answer.
- Every production change receives a focused test first. Do not stage existing untracked `data/`, `.pytest-tmp/`, or unrelated plan files.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `evaluation/release_metrics.py` | Fast not-applicable path, campaign-scoped release snapshot, terminal cache, pure release derivation. |
| `evaluation/campaign_schemas.py` | Additive Release Metrics availability contract. |
| `evaluation/db.py` | Bounded result read models, trace summary migration/index, answer-size enforcement, one-time WAL initialization. |
| `evaluation/accounting_store.py` | Bulk scope-target loader and campaign accounting snapshot. |
| `evaluation/observability_storage.py` | Bulk V9/release-observability snapshot by campaign. |
| `evaluation/research_analytics.py` | Consume research-result projection rather than full campaign results. |
| `Multimodal_RAG_System/src/pages/EvaluationCenter.tsx` | Do not request Release Metrics when selected campaign lacks `benchmark_id`. |
| `Multimodal_RAG_System/src/components/evaluation/CampaignOverviewTab.tsx` | Render the normal not-applicable Release Metrics state. |

### Task 1: Add the P0 not-applicable Release Metrics contract

**Files:**

- Modify: `D:\flutterserver\pdftopng\evaluation\release_metrics.py:42-76,358-397`
- Modify: `D:\flutterserver\Multimodal_RAG_System\src\types\evaluation.ts:538-565`
- Modify: `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.tsx:140-165`
- Modify: `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\CampaignOverviewTab.tsx:60-110`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_release_metrics.py`
- Test: `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.integration.test.tsx`

**Interfaces:**

- Consumes: `CampaignStatus.config.benchmark_id`.
- Produces: `ReleaseMetricsReport.availability: Literal["available", "not_applicable"]` and nullable `not_applicable_reason`.

- [ ] **Step 1: Write the failing backend fast-path tests.**

Create a stub `CampaignRepository` whose anchor has `benchmark_id=None`; make every result/score/accounting/observability method raise if called. Assert the route/service returns HTTP-model data, not an exception:

```python
report = await ReleaseMetricsService(
    campaigns=campaigns,
    results=FailIfCalledResults(),
    ragas_scores=FailIfCalledScores(),
    accounting=FailIfCalledAccounting(),
).get_report(user_id="user-1", campaign_id="campaign-1")

assert report.availability == "not_applicable"
assert report.not_applicable_reason == "benchmark_not_configured"
assert report.comparable is False
assert report.gate_reasons == ["benchmark_not_configured"]
```

Run:

```powershell
cd D:\flutterserver\pdftopng
.venv\Scripts\python.exe -m pytest tests/test_evaluation_release_metrics.py -q
```

Expected: FAIL because the report has no availability fields and calls downstream repositories.

- [ ] **Step 2: Add the backwards-compatible response fields and fast return.**

In `ReleaseMetricsReport`, add defaults so old clients still parse successful reports:

```python
availability: Literal["available", "not_applicable"] = "available"
not_applicable_reason: str | None = None
```

At the beginning of `get_report()`, immediately after loading `anchor`, return this report if no benchmark ID exists:

```python
if not anchor.config.benchmark_id:
    return ReleaseMetricsReport(
        benchmark_id="",
        benchmark_kind="not_applicable",
        comparable=False,
        gate_reasons=["benchmark_not_configured"],
        availability="not_applicable",
        not_applicable_reason="benchmark_not_configured",
    )
```

Delete the old end-of-method `benchmark_id_missing` gate because the no-ID branch never derives a release report.

- [ ] **Step 3: Write the failing frontend no-request and display tests.**

Mock a selected `CampaignStatus` with `config.benchmark_id: null`. Assert `getCampaignReleaseMetrics` is not called, and assert the overview renders the exact normal-state message `Release Metrics 不適用：尚未設定 benchmark。`. Add a benchmark-configured fixture and assert the API call remains present.

Run:

```powershell
cd D:\flutterserver\Multimodal_RAG_System
npm test -- --run src/pages/EvaluationCenter.integration.test.tsx
```

Expected: FAIL because the page always requests the endpoint and the overview has no not-applicable rendering.

- [ ] **Step 4: Gate the frontend request and render the state.**

Change the initial campaign load so the release request is conditional:

```ts
const selected = campaigns.find((item) => item.id === selectedCampaignId);
const releaseMetrics = selected?.config?.benchmark_id
  ? await getCampaignReleaseMetrics(selectedCampaignId).catch(() => undefined)
  : undefined;
```

Keep `researchSummary` unconditional. In `CampaignOverviewTab`, render an informational `Alert` whenever `releaseMetrics?.availability === 'not_applicable'` or no report exists because the selected campaign has no benchmark ID; do not show a failure, loading spinner, or a “Comparable: no” gate warning for this normal state.

- [ ] **Step 5: Verify P0 and commit.**

```powershell
cd D:\flutterserver\pdftopng
.venv\Scripts\python.exe -m pytest tests/test_evaluation_release_metrics.py tests/test_evaluation_analytics_api.py -q
git add evaluation/release_metrics.py tests/test_evaluation_release_metrics.py
git commit -m "fix(evaluation): skip release metrics without benchmark"

cd D:\flutterserver\Multimodal_RAG_System
npm test -- --run src/pages/EvaluationCenter.integration.test.tsx
npm run lint:ci
git add src/types/evaluation.ts src/pages/EvaluationCenter.tsx src/components/evaluation/CampaignOverviewTab.tsx src/pages/EvaluationCenter.integration.test.tsx
git commit -m "fix(evaluation): treat missing benchmark as not applicable"
```

Expected: missing-benchmark campaigns complete initial load without any Release Metrics request; configured campaigns retain the existing report request.

### Task 2: Replace per-run Release Metrics reads with one campaign snapshot

**Files:**

- Modify: `D:\flutterserver\pdftopng\evaluation\release_metrics.py:334-500`
- Modify: `D:\flutterserver\pdftopng\evaluation\db.py:1928-2201`
- Modify: `D:\flutterserver\pdftopng\evaluation\accounting_store.py:295-323`
- Modify: `D:\flutterserver\pdftopng\evaluation\observability_storage.py`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_release_metrics.py`

**Interfaces:**

- Consumes: `CampaignReleaseResult`, `CampaignAccountingSnapshot`, and `CampaignReleaseObservabilitySnapshot`, each scoped to exactly one campaign.
- Produces: `ReleaseMetricsService._build_release_runs(...) -> list[ReleaseRun]` without calling `run_detail`, `_run_tokens`, or any repository in a result loop.

- [ ] **Step 1: Write query-bound and parity tests.**

Use fake repositories that record calls. Build fixtures with one and 160 results. Assert both fixtures make the same number of repository calls per selected campaign and derive identical per-run values for a two-result golden fixture:

```python
assert repositories.calls == {
    "results.list_for_campaign_release": 1,
    "scores.list_for_campaign": 1,
    "scores.list_work_metadata_for_campaign": 1,
    "accounting.load_campaign_snapshot": 1,
    "observability.load_release_snapshot": 1,
}
assert [(run.run_id, run.quality_score) for run in runs] == [
    ("naive-1", 0.4), ("v9-1", 0.7),
]
```

Run the release test file and confirm the call-count assertion fails against the current per-result loop.

- [ ] **Step 2: Add the bounded release result projection.**

Add a frozen `CampaignReleaseResult` dataclass in `evaluation/db.py`. Select only result identity, campaign/run/mode metadata, status/error, latency, category, `source_attempt_id`, and JSON snapshots consumed by release fingerprints/gates. Exclude `answer`, `contexts_json`, `ground_truth`, `ground_truth_short`, source IDs, and RAGAS text. Add:

```python
async def list_for_campaign_release(
    self, *, user_id: str, campaign_id: str
) -> list[CampaignReleaseResult]:
    ...
```

The SQL must use `WHERE campaign_id = ? AND user_id = ?` and the existing `idx_campaign_results_campaign_user_order` index.

- [ ] **Step 3: Add one accounting snapshot per campaign.**

Create `CampaignAccountingSnapshot` in `accounting_store.py` with `scopes_by_run_id` and `events_by_scope_id`. Implement `load_campaign_snapshot(campaign_id)` with one bulk scope/target load and one usage-event load. `_build_release_runs()` receives this snapshot and computes tokens from its grouped objects; remove `_run_tokens()`.

- [ ] **Step 4: Add one release-observability snapshot per campaign.**

Create `CampaignReleaseObservabilitySnapshot` in `observability_storage.py`, grouped by `run_id`, containing only the V9 materializations, evidence packets, slot resolutions, claims, context packs, and graph events needed by `_release_run`. Implement `load_release_snapshot(campaign_id)` with campaign-level queries and no `SELECT *` from unrelated observability tables.

- [ ] **Step 5: Refactor the service to build purely in memory.**

In `get_report()`, use `asyncio.gather` for independent per-campaign snapshot reads, then pass their groups to a synchronous `_release_run_from_snapshot(...)`. The inner `for result in results` loop may only read dictionaries/lists already in memory. Delete calls to `_analytics.run_detail()` and `list_graph_events_for_run()` from this path.

- [ ] **Step 6: Verify parity and commit.**

```powershell
cd D:\flutterserver\pdftopng
.venv\Scripts\python.exe -m pytest tests/test_evaluation_release_metrics.py tests/test_evaluation_accounting_store.py tests/test_evaluation_observability_repository.py -q
git add evaluation/release_metrics.py evaluation/db.py evaluation/accounting_store.py evaluation/observability_storage.py tests/test_evaluation_release_metrics.py
git commit -m "perf(evaluation): bulk load release metrics"
```

Expected: the 160-result fixture has constant repository-call count and returns the same release metrics/gates as the pre-refactor golden fixture.

### Task 3: Cache terminal Release Metrics reports safely

**Files:**

- Modify: `D:\flutterserver\pdftopng\evaluation\release_metrics.py`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_release_metrics.py`

**Interfaces:**

- Consumes: selected campaign IDs plus each campaign `updated_at` and terminal status.
- Produces: `ReleaseMetricsReport` cache entries only when every campaign included in the benchmark is terminal.

- [ ] **Step 1: Write failing cache tests.**

Assert two reads of an unchanged completed benchmark call the bulk loaders once; changing one selected campaign marker reloads; any running/evaluating campaign reloads each time.

- [ ] **Step 2: Implement a complete cache key.**

Use a tuple rather than a single anchor marker:

```python
cache_key = tuple(
    (campaign.id, campaign.updated_at.isoformat(), campaign.status.value)
    for campaign in sorted(selected, key=lambda item: item.id)
)
```

Store only when every status is `completed`, `completed_with_errors`, `failed`, or `cancelled`. Keep the cache process-local and do not cache `not_applicable` reports.

- [ ] **Step 3: Verify and commit.**

```powershell
cd D:\flutterserver\pdftopng
.venv\Scripts\python.exe -m pytest tests/test_evaluation_release_metrics.py -q
git add evaluation/release_metrics.py tests/test_evaluation_release_metrics.py
git commit -m "perf(evaluation): cache terminal release reports"
```

### Task 4: Use bounded research-result projections

**Files:**

- Modify: `D:\flutterserver\pdftopng\evaluation\db.py`
- Modify: `D:\flutterserver\pdftopng\evaluation\research_analytics.py:93-670`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_research_analytics.py`

**Interfaces:**

- Consumes: `CampaignResearchResult` with all compatibility/accounting fields used by summary, question comparison, and agent behavior.
- Produces: unchanged public `CampaignResearchSummaryResponse`, `ResearchQuestionComparisonResponse`, and `AgentBehaviorResponse`.

- [ ] **Step 1: Write failing projection tests.**

Seed a completed result with a 2 MB answer, contexts, and ground truth plus ordinary metric/snapshot fields. Use a repository spy and assert each research endpoint calls `list_for_campaign_research`, never `list_for_campaign`; assert its selected-column SQL omits `answer`, `contexts_json`, and `ground_truth`.

- [ ] **Step 2: Implement `CampaignResearchResult` and its repository method.**

Add a dataclass and `list_for_campaign_research(...)` beside existing analytics projections. Include all fields actually accessed by `research_analytics.py`; parse only necessary JSON (`derived_metrics`, snapshots, key points/focus where used). Derive `answer_preview` with `substr(answer, 1, 240)` only if a response field displays it.

- [ ] **Step 3: Switch the three research endpoints and retain compatibility doubles.**

Use `getattr(repository, "list_for_campaign_research", None)` with the existing full method only as a test-double fallback. Confirm no aggregate method accesses `.answer` or `.contexts` after the replacement.

- [ ] **Step 4: Verify response parity and commit.**

```powershell
cd D:\flutterserver\pdftopng
.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_question_comparison.py -q
git add evaluation/db.py evaluation/research_analytics.py tests/test_evaluation_research_analytics.py
git commit -m "perf(evaluation): bound research analytics reads"
```

### Task 5: Store trace summaries and index campaign trace lists

**Files:**

- Modify: `D:\flutterserver\pdftopng\evaluation\db.py:720-735,2228-2375`
- Modify: `D:\flutterserver\pdftopng\evaluation\trace_schemas.py:380-420`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_db.py`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_api.py`

**Interfaces:**

- Consumes: a complete `AgentTraceDetail` at write time.
- Produces: a `summary_json` column and `AgentTraceSummary` list read that never selects `trace_json`.

- [ ] **Step 1: Write failing migration and query-plan tests.**

Create a legacy `agent_traces` table without `summary_json`, run `force_init_db()`, then assert the column and index exist. Seed traces and run `EXPLAIN QUERY PLAN` for the list query; assert it contains `idx_agent_traces_campaign_user_created`. Assert the list SQL is a projection without `trace_json`.

- [ ] **Step 2: Add additive schema/migration support.**

Add `summary_json TEXT NOT NULL DEFAULT '{}'` to new schemas and `_MIGRATION_COLUMNS`; add:

```sql
CREATE INDEX IF NOT EXISTS idx_agent_traces_campaign_user_created
ON agent_traces(campaign_id, user_id, created_at DESC);
```

At `replace_for_result()`, serialize `summarize_agent_trace(detail).model_dump(mode="json")` into `summary_json` alongside the raw trace.

- [ ] **Step 3: Make trace lists summary-only and preserve legacy details.**

`list_for_campaign()` selects `id, campaign_id, campaign_result_id, user_id, summary_json, created_at`, parses the summary, and returns it. For blank legacy summaries, return a minimal `AgentTraceSummary` marked `trace_status="not_instrumented"` without parsing raw trace. `get_for_result()` remains the sole path that loads `trace_json` and can lazily populate a missing summary in the same transaction.

- [ ] **Step 4: Verify and commit.**

```powershell
cd D:\flutterserver\pdftopng
.venv\Scripts\python.exe -m pytest tests/test_evaluation_db.py tests/test_evaluation_api.py tests/test_evaluation_research_analytics.py -q
git add evaluation/db.py evaluation/trace_schemas.py tests/test_evaluation_db.py tests/test_evaluation_api.py
git commit -m "perf(evaluation): summarize campaign trace lists"
```

### Task 6: Bulk-load accounting targets and set WAL once

**Files:**

- Modify: `D:\flutterserver\pdftopng\evaluation\accounting_store.py:295-415`
- Modify: `D:\flutterserver\pdftopng\evaluation\db.py:766-820`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_accounting_store.py`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_db.py`

**Interfaces:**

- Consumes: `campaign_id`.
- Produces: `list_campaign_scopes()` using exactly two SQL statements: all scopes, then all targets grouped by scope ID.

- [ ] **Step 1: Write failing SQL-count and lifecycle tests.**

Seed three scopes with two targets each, instrument `connection.execute`, and assert `list_campaign_scopes()` performs one scopes statement plus one `WHERE scope_id IN (...)` target statement. Assert `connect_db()` no longer executes `PRAGMA journal_mode=WAL`, while `init_db()` leaves `PRAGMA journal_mode` equal to `wal`, `foreign_keys` enabled, and busy timeout configured.

- [ ] **Step 2: Implement the bulk target loader.**

Replace the list comprehension that awaits `_load_scope_targets()` for each row with `_load_campaign_scope_targets(connection, scope_ids)`. Use placeholders generated from the known scope-ID list, order by `scope_id, created_at, attempt_id`, and group rows with `defaultdict(list)` before calling `_scope_from_row`.

- [ ] **Step 3: Move WAL setup into initialization.**

Execute `PRAGMA journal_mode=WAL;` in `init_db()` and `force_init_db()` before schema work. Remove it from `connect_db()`. Retain `synchronous=NORMAL`, `busy_timeout=5000`, and `foreign_keys=ON` as required connection setup after confirming their current test expectations.

- [ ] **Step 4: Verify and commit.**

```powershell
cd D:\flutterserver\pdftopng
.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_store.py tests/test_evaluation_db.py tests/test_evaluation_accounting_runtime.py -q
git add evaluation/accounting_store.py evaluation/db.py tests/test_evaluation_accounting_store.py tests/test_evaluation_db.py
git commit -m "perf(evaluation): batch accounting database reads"
```

### Task 7: Reject oversized evaluation answers without silent truncation

**Files:**

- Modify: `D:\flutterserver\pdftopng\evaluation\db.py:1928-2115`
- Modify: `D:\flutterserver\pdftopng\evaluation\campaign_engine.py`
- Modify: `D:\flutterserver\pdftopng\evaluation\execution_worker.py`
- Test: `D:\flutterserver\pdftopng\tests\test_campaign_engine.py`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_execution_worker.py`

**Interfaces:**

- Consumes: candidate answer text before `CampaignResultRepository.create()`.
- Produces: a failed `CampaignResult` with `error_message="EVALUATION_ANSWER_TOO_LARGE"` when UTF-8 answer bytes exceed 1,048,576.

- [ ] **Step 1: Write failing boundary tests.**

Create a result at exactly `1_048_576` UTF-8 bytes and assert it persists completed. Create one at `1_048_577` bytes and assert no completed result is promoted, the result status is failed, `answer == ""`, and the error message is exactly `EVALUATION_ANSWER_TOO_LARGE`.

- [ ] **Step 2: Enforce the limit at the repository boundary.**

Define `MAX_EVALUATION_ANSWER_BYTES = 1_048_576` beside existing trace payload limits. In `CampaignResultRepository.create()`, normalize oversized inputs before the insert:

```python
if len(answer.encode("utf-8")) > MAX_EVALUATION_ANSWER_BYTES:
    answer = ""
    status = CampaignResultStatus.FAILED
    error_message = "EVALUATION_ANSWER_TOO_LARGE"
```

Preserve the original worker attempt's diagnostic metadata and do not schedule RAGAS work for this failed result.

- [ ] **Step 3: Verify worker behavior and commit.**

```powershell
cd D:\flutterserver\pdftopng
.venv\Scripts\python.exe -m pytest tests/test_campaign_engine.py tests/test_evaluation_execution_worker.py tests/test_evaluation_db.py -q
git add evaluation/db.py evaluation/campaign_engine.py evaluation/execution_worker.py tests/test_campaign_engine.py tests/test_evaluation_execution_worker.py
git commit -m "fix(evaluation): reject oversized answers"
```

### Task 8: Document, benchmark, and run full verification

**Files:**

- Modify: `D:\flutterserver\pdftopng\docs\BACKEND.md`
- Modify: `D:\flutterserver\pdftopng\docs\backend-maintenance.md`
- Modify: `D:\flutterserver\pdftopng\docs\generated\api-surface.md`
- Modify: `D:\flutterserver\pdftopng\docs\superpowers\specs\2026-07-24-evaluation-center-load-performance-design.md`
- Test: `D:\flutterserver\pdftopng\tests\test_evaluation_release_metrics.py`

- [ ] **Step 1: Add a 160-run performance fixture.**

Seed 160 result rows with bounded payloads and assert a missing-benchmark `get_report()` never calls bulk loaders. Seed a benchmark ID and assert the bulk implementation has constant repository call count; record elapsed wall time only as diagnostic output, not a flaky assertion.

- [ ] **Step 2: Update runtime/API documentation.**

Document `availability` and `not_applicable_reason`, the rule that missing benchmark configuration is normal, the bounded/detail-only endpoints, terminal cache invalidation, trace-summary schema migration, and the answer-size failure code.

- [ ] **Step 3: Run backend verification.**

```powershell
cd D:\flutterserver\pdftopng
.venv\Scripts\python.exe -m pytest tests/test_evaluation_release_metrics.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_accounting_store.py tests/test_evaluation_accounting_runtime.py tests/test_evaluation_db.py tests/test_evaluation_api.py tests/test_campaign_engine.py tests/test_evaluation_execution_worker.py -q --tb=short
```

Expected: all focused evaluation tests pass with no SQLite lock error, no unbounded release query path, and no change to successful benchmark metrics.

- [ ] **Step 4: Run frontend verification.**

```powershell
cd D:\flutterserver\Multimodal_RAG_System
npm run lint:ci
npx tsc --noEmit
npm test -- --run src/pages/EvaluationCenter.integration.test.tsx src/components/evaluation/CampaignOverviewTab.test.tsx
npm run build
```

Expected: zero lint warnings/errors, typecheck and build pass, and the no-benchmark page makes no Release Metrics request.

- [ ] **Step 5: Commit documentation.**

```powershell
cd D:\flutterserver\pdftopng
git add docs/BACKEND.md docs/backend-maintenance.md docs/generated/api-surface.md docs/superpowers/specs/2026-07-24-evaluation-center-load-performance-design.md tests/test_evaluation_release_metrics.py
git commit -m "docs(evaluation): document bounded analytics loading"
```

## Self-Review

- P0 is independently shippable and explicitly handles missing benchmark configuration as normal.
- P1 removes both release-report and research-summary full-payload reads before addressing storage limits.
- P2 keeps SQLite lifecycle semantics and removes avoidable N+1/setup overhead.
- All migrations are additive; no historical result, answer, or trace is deleted.
- The test plan measures query cardinality and public-response parity rather than relying on timing-only assertions.
