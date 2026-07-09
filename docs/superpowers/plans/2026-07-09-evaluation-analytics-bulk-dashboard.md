# Evaluation Analytics Bulk Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Evaluation Center backend latency by replacing per-run observability reads with campaign-level bulk reads, sharing campaign analytics context, and exposing a single dashboard bundle endpoint.

**Architecture:** Keep existing response schemas and public endpoints compatible. Add campaign-level repository reads in `EvaluationObservabilityRepository`, introduce a private analytics context in `EvaluationAnalyticsService`, refactor aggregate builders to consume that context, and add `/campaigns/{campaign_id}/analytics-dashboard` as the preferred frontend bundle endpoint.

**Tech Stack:** Python 3, FastAPI, Pydantic, SQLite via `aiosqlite`, pytest, existing `evaluation` package.

## Global Constraints

- Preserve existing API contracts for current endpoints unless a new endpoint is explicitly added.
- Do not cache mutable running campaign data in this pass.
- Keep redaction behavior unchanged for export.
- Keep user ownership checks through existing campaign/result repository access.
- Use test-first changes for new repository APIs, analytics context behavior, export/errors bulk behavior, and the dashboard bundle endpoint.
- Commit by backend task boundary.

---

## File Structure

- Modify `evaluation/observability_storage.py`: add campaign-level bulk list methods grouped by `run_id`.
- Modify `evaluation/analytics.py`: add `_CampaignAnalyticsContext`, context loader, pure builders, bulk export/errors, and `analytics_dashboard`.
- Modify `evaluation/campaign_schemas.py`: add `CampaignAnalyticsDashboardResponse`.
- Modify `evaluation/router.py`: add `GET /api/evaluation/campaigns/{campaign_id}/analytics-dashboard`.
- Modify migration code in `evaluation/db.py` only if indexes are missing.
- Modify `tests/test_evaluation_observability_repository.py`: repository bulk query tests.
- Modify `tests/test_evaluation_analytics_api.py`: dashboard endpoint API contract test.
- Modify `tests/test_evaluation_export_redaction.py`: bulk path behavior/regression around export/errors.
- Add or modify focused service tests if repository call-count verification cannot be cleanly expressed via API tests.

---

### Task 1: Campaign-Level Observability Bulk Queries

**Files:**
- Modify: `evaluation/observability_storage.py`
- Test: `tests/test_evaluation_observability_repository.py`

**Interfaces:**
- Produces:
  - `async def list_trace_events_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationTraceEvent]]`
  - `async def list_llm_calls_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationLlmCall]]`
  - `async def list_retrieval_chunks_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationRetrievalChunk]]`
  - `async def list_claims_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationClaim]]`
  - `async def list_human_ratings_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationHumanRating]]`

- [ ] **Step 1: Write failing repository bulk test**

Add a test that seeds two runs in one campaign plus one row in another campaign, calls each bulk method, and asserts rows are grouped by `run_id`, sorted like the existing per-run methods, and scoped to the requested `campaign_id`.

- [ ] **Step 2: Run red test**

Run: `pytest tests/test_evaluation_observability_repository.py -q`
Expected: FAIL with missing `list_*_for_campaign` attributes.

- [ ] **Step 3: Implement bulk methods**

Use one `SELECT * FROM table WHERE campaign_id = ? ORDER BY ...` query per table and group rows with `defaultdict(list)`.

- [ ] **Step 4: Run green test**

Run: `pytest tests/test_evaluation_observability_repository.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat(evaluation): add campaign observability bulk reads`

---

### Task 2: Shared Campaign Analytics Context

**Files:**
- Modify: `evaluation/analytics.py`
- Test: focused analytics service test or existing `tests/test_evaluation_analytics_api.py`

**Interfaces:**
- Consumes Task 1 bulk LLM calls.
- Produces:
  - `_CampaignAnalyticsContext`
  - `_load_campaign_context(user_id: str, campaign_id: str) -> _CampaignAnalyticsContext`
  - `_build_campaign_overview(context: _CampaignAnalyticsContext) -> CampaignOverviewResponse`

- [ ] **Step 1: Write failing context/reuse test**

Use fake repositories to call `campaign_overview()`. Assert the fake observability repository's `list_llm_calls_for_campaign` count is `1` and per-run `list_llm_calls_for_run` count is `0`.

- [ ] **Step 2: Run red test**

Run focused pytest command for the new/modified test.
Expected: FAIL because `campaign_overview()` still calls `list_llm_calls_for_run`.

- [ ] **Step 3: Implement context loader**

Load campaign, results, and `llm_calls_by_run` once. Compute overview from context and make `campaign_overview()` return `_build_campaign_overview(context)`.

- [ ] **Step 4: Run green test**

Run focused pytest command and then `pytest tests/test_evaluation_analytics_api.py -q`.

- [ ] **Step 5: Commit**

Commit message: `refactor(evaluation): share campaign analytics context`

---

### Task 3: Reuse Context Across Aggregate Analytics

**Files:**
- Modify: `evaluation/analytics.py`
- Test: focused analytics service call-count test and `tests/test_evaluation_analytics_api.py`

**Interfaces:**
- Consumes `_CampaignAnalyticsContext`.
- Produces pure builder methods:
  - `_build_mode_comparison(context)`
  - `_build_question_comparison(context)`
  - `_build_cost_latency(context)`
  - `_build_router_analysis(context, routing_decisions_by_run)`
  - `_build_ablation(context)`
  - `_build_repeat_stability(context)`

- [ ] **Step 1: Write failing aggregate call-count test**

Call `mode_comparison`, `question_comparison`, `cost_latency`, `ablation`, and `repeat_stability` on a fake service. Assert each method does not call `campaign_overview()` internally and uses one context load.

- [ ] **Step 2: Run red test**

Run focused pytest command.
Expected: FAIL because aggregate endpoints call `campaign_overview()` and list results again.

- [ ] **Step 3: Refactor aggregate methods**

Each public method should load context and call the matching builder. Router analysis may still read routing decisions per run unless a routing bulk query is added later; it must not recompute overview.

- [ ] **Step 4: Run green test**

Run focused pytest command and `pytest tests/test_evaluation_analytics_api.py -q`.

- [ ] **Step 5: Commit**

Commit message: `refactor(evaluation): reuse analytics context builders`

---

### Task 4: Bulk Export And Errors

**Files:**
- Modify: `evaluation/analytics.py`
- Test: `tests/test_evaluation_export_redaction.py`

**Interfaces:**
- Consumes Task 1 bulk trace, LLM, retrieval chunk, and claim reads.
- Produces export/errors paths that do not call per-run trace/LLM/retrieval/claim methods.

- [ ] **Step 1: Write failing bulk export/errors test**

Wrap observability repository methods or use a fake repository to assert `campaign_errors()` and `export_campaign()` call campaign bulk methods once each and do not call per-run methods for trace, LLM calls, retrieval chunks, or claims.

- [ ] **Step 2: Run red test**

Run: `pytest tests/test_evaluation_export_redaction.py -q`
Expected: FAIL because current implementation still calls per-run methods.

- [ ] **Step 3: Implement bulk export/errors**

`campaign_errors()` should use context, `list_trace_events_for_campaign`, and `list_llm_calls_for_campaign`. `export_campaign()` should use context, bulk trace/LLM/retrieval/claims, `_build_campaign_errors(context, trace_events_by_run, llm_calls_by_run)`, and context overview.

- [ ] **Step 4: Run green test**

Run: `pytest tests/test_evaluation_export_redaction.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `perf(evaluation): bulk load export and errors`

---

### Task 5: Dashboard Bundle Endpoint

**Files:**
- Modify: `evaluation/campaign_schemas.py`
- Modify: `evaluation/analytics.py`
- Modify: `evaluation/router.py`
- Test: `tests/test_evaluation_analytics_api.py`

**Interfaces:**
- Produces:
  - `CampaignAnalyticsDashboardResponse`
  - `EvaluationAnalyticsService.analytics_dashboard(user_id: str, campaign_id: str) -> CampaignAnalyticsDashboardResponse`
  - `GET /api/evaluation/campaigns/{campaign_id}/analytics-dashboard`

- [ ] **Step 1: Write failing API test**

Extend analytics API test to request `/analytics-dashboard` and assert it returns `overview`, `mode_comparison`, `question_comparison`, `cost_latency`, `router_analysis`, `ablation`, `human_vs_auto`, `human_queue`, `errors`, and `runs`.

- [ ] **Step 2: Run red test**

Run: `pytest tests/test_evaluation_analytics_api.py -q`
Expected: FAIL with 404 for `/analytics-dashboard`.

- [ ] **Step 3: Add schema, service method, and router endpoint**

Build the response from one shared context plus existing builder methods. Do not call public endpoint methods from inside `analytics_dashboard()` because that would reload context.

- [ ] **Step 4: Run green test**

Run: `pytest tests/test_evaluation_analytics_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat(evaluation): add analytics dashboard bundle endpoint`

---

### Task 6: Observability Index Check

**Files:**
- Modify: `evaluation/db.py`
- Test: `tests/test_evaluation_observability_schema.py` or `tests/test_migration_metadata.py`

**Interfaces:**
- Produces SQLite indexes on `(campaign_id, run_id)` for:
  - `evaluation_llm_calls`
  - `evaluation_trace_events`
  - `evaluation_retrieval_chunks`
  - `evaluation_claims`
  - `evaluation_human_ratings`

- [ ] **Step 1: Inspect existing indexes**

Use `rg "CREATE INDEX|evaluation_llm_calls|evaluation_trace_events" evaluation/db.py tests/test_evaluation_observability_schema.py`.

- [ ] **Step 2: Add failing schema test if indexes are missing**

Query `PRAGMA index_list(table)` and assert a campaign/run index exists for each table.

- [ ] **Step 3: Add missing indexes**

Add `CREATE INDEX IF NOT EXISTS` statements near existing observability table creation.

- [ ] **Step 4: Run schema tests**

Run: `pytest tests/test_evaluation_observability_schema.py tests/test_migration_metadata.py -q`

- [ ] **Step 5: Commit**

Commit message: `perf(evaluation): index campaign observability reads`

---

## Final Verification

- [ ] Run `pytest tests/test_evaluation_observability_repository.py -q`
- [ ] Run `pytest tests/test_evaluation_export_redaction.py -q`
- [ ] Run `pytest tests/test_evaluation_analytics_api.py -q`
- [ ] Run `pytest tests/test_evaluation_observability_schema.py tests/test_migration_metadata.py -q`
- [ ] Run a broader focused suite: `pytest tests/test_evaluation_api.py tests/test_evaluation_human_ratings.py tests/test_evaluation_recorder.py tests/test_evaluation_token_cost.py -q`

## Self-Review

- Spec coverage: Tasks 1-4 cover bulk observability, shared context, aggregate reuse, export/errors. Task 5 covers Plan B bundle endpoint. Task 6 covers indexes.
- Placeholder scan: No implementation placeholders remain; each task has concrete files, commands, and expected behavior.
- Type consistency: Bulk method names and dashboard response names are used consistently across tasks.
