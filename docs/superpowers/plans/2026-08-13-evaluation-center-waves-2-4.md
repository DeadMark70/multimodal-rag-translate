# Evaluation Center Waves 2-4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` task-by-task. Wave 1 is complete.
> This file supersedes Wave 2-4 in
> `2026-08-12-evaluation-center-observability-export-v2.md`. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Finish truthful Evaluation Center panels, replace the campaign export
with strict Schema v2, and prove panel/export parity without relying on local
database contents.

**Architecture:** Wave 2 repairs only contracts and components used by the
mounted Evaluation Center. Wave 3 introduces an internal campaign
observability snapshot, separate interactive/export projectors, a typed export
composer, and a strict frontend decoder. Wave 4 proves serialized HTTP parity
and runs the release/documentation gates.

**Tech Stack:** Python 3.13 local environment, FastAPI, Pydantic v2, SQLite
repositories, pytest, Ruff; React 18, TypeScript 5.9, Chakra UI, Axios, Zod 4,
Vitest, Testing Library.

## Global Constraints

- Base claims on code contracts and tests; never use local `.db` rows as
  production evidence.
- Work directly in `D:/flutterserver/pdftopng` (`main`) and
  `D:/flutterserver/Multimodal_RAG_System` (`master`), as explicitly approved.
- Preserve the completed Wave 1 selected-run endpoint and interactive redaction
  boundary.
- Keep nullable values nullable. Unknown/non-finite values must not become zero.
- Export Schema v2 replaces the old response; do not add compatibility aliases.
- `include_run_observability` is explicitly sent and defaults to `false`.
- Provider response bodies, credentials, authorization headers, unrestricted
  errors/payloads, and stack traces never leave the backend.
- Full export uses campaign-bounded loaders and never silently truncates.
- Do not refactor inactive run-diff or repeat-stability APIs in this plan.
- Each Task changes one repository, uses RED-GREEN, self-reviews, and creates
  exactly one focused commit.
- Use a fresh implementation subagent for each Task. Do not run per-Task review
  agents; run one consolidated review after all Tasks in a Wave, as approved.
- Fix consolidated-review Critical/Important findings inside the same Wave with
  separate corrective commits and one scoped re-review.
- At each Wave checkpoint report hashes and exact checks, provide the safe real
  system checklist, then stop for the user's push and validation.
- Never begin the next Wave without explicit acceptance.

## Task and Commit Ledger

| Wave | Task | Repository | Required commit subject |
| --- | --- | --- | --- |
| 2 | 5 | backend | `chore(api): synchronize evaluation openapi baseline` |
| 2 | 6 | backend | `fix(evaluation): remove unused result progress field` |
| 2 | 7 | backend | `fix(evaluation): preserve unknown run tokens` |
| 2 | 8 | backend | `fix(evaluation): type retrospective router analysis` |
| 2 | 9 | frontend | `fix(evaluation-ui): synchronize active api contracts` |
| 2 | 10 | frontend | `fix(evaluation-ui): separate router analysis and execution` |
| 2 | 11 | frontend | `fix(evaluation-ui): remove uninstrumented placeholders` |
| 2 | 12 | frontend | `fix(evaluation-ui): preserve unknown and zero values` |
| 2 | 13 | frontend | `feat(evaluation-ui): mount durable campaign jobs` |
| 3 | 14 | backend | `feat(evaluation): define export schema v2` |
| 3 | 15 | backend | `refactor(evaluation): build campaign observability snapshots` |
| 3 | 16 | backend | `feat(evaluation): serve sanitized export schema v2` |
| 3 | 17 | frontend | `feat(evaluation-ui): consume export schema v2` |
| 4 | 18 | backend | `test(evaluation): prove panel export http parity` |
| 4 | 19 | frontend | `test(evaluation-ui): lock export contract release gate` |

---

# Wave 2 — Panel Truthfulness and Operations

### Task 5: Repair the pre-existing OpenAPI baseline

**Repository:** `pdftopng`

**Files:**
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`
- Generate: `docs/generated/api-surface.md`

**Interfaces:**
- No production schema or route change.
- Generated artifacts must match the post-Wave 1 runtime before new Wave 2
  schema changes begin.

- [ ] **Step 1: Capture the existing RED contract check**

```powershell
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
```

Expected: exit 1 naming stale `openapi.json` and
`contracts/openapi-contract.json`. Record the exact output in the Task report.

- [ ] **Step 2: Regenerate only the declared artifacts**

```powershell
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --write
```

- [ ] **Step 3: Verify the baseline is clean and non-mutating**

```powershell
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_openapi_artifacts.py -q
git diff --check
```

- [ ] **Step 4: Commit only generated baseline drift**

```powershell
git add openapi.json contracts/openapi-contract.json docs/generated/api-surface.md
git commit -m "chore(api): synchronize evaluation openapi baseline"
```

### Task 6: Remove the unused progress result field

**Repository:** `pdftopng`

**Files:**
- Modify: `evaluation/campaign_schemas.py`
- Modify: `tests/test_campaign_schemas.py`
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`
- Generate: `docs/generated/api-surface.md`

**Interfaces:** `CampaignProgressEvent` has no `latest_result_id`. No router,
repository, engine, or SSE lookup is added because the field was never populated.

- [ ] **Step 1: Write and run the RED schema test**

```python
def test_campaign_progress_event_does_not_publish_latest_result_id() -> None:
    assert "latest_result_id" not in CampaignProgressEvent.model_fields
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py -k "progress_event" -q
```

Expected: RED because the field still exists.

- [ ] **Step 2: Remove only the schema field**

Delete `latest_result_id` from `CampaignProgressEvent`. Do not change
`evaluation/router.py`; its model dump will automatically stop publishing the
key.

- [ ] **Step 3: Synchronize, verify, and commit once**

```powershell
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/campaign_schemas.py tests/test_campaign_schemas.py
git add evaluation/campaign_schemas.py tests/test_campaign_schemas.py openapi.json contracts/openapi-contract.json docs/generated/api-surface.md
git commit -m "fix(evaluation): remove unused result progress field"
```

### Task 7: Preserve unknown run-list token totals

**Repository:** `pdftopng`

**Files:**
- Modify: `evaluation/campaign_schemas.py`
- Modify: `evaluation/analytics.py`
- Modify: `tests/test_campaign_schemas.py`
- Modify: `tests/test_evaluation_analytics_api.py`
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`
- Generate: `docs/generated/api-surface.md`
- Modify: `docs/product-specs/evaluation-api.md`

**Interfaces:**
- `EvaluationRunListItem.total_tokens: int | None = None`.
- `_build_campaign_runs()` passes `CampaignResult.total_tokens` unchanged.
- Do not change run diff, overview, or repeat stability.

- [ ] **Step 1: Write RED schema/API tests**

Construct `EvaluationRunListItem(total_tokens=None)`. In the existing legacy
campaign API fixture request `/api/evaluation/campaigns/{id}/runs` and assert:

```python
assert legacy_runs.json()["runs"][0]["total_tokens"] is None
```

- [ ] **Step 2: Run RED tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py tests/test_evaluation_analytics_api.py -k "run_list or owned_run_details" -q
```

- [ ] **Step 3: Apply the two-line production repair**

```python
total_tokens: int | None = Field(default=None, ge=0)
# _build_campaign_runs:
total_tokens=item.total_tokens
```

- [ ] **Step 4: Synchronize, verify, and commit once**

```powershell
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py tests/test_evaluation_analytics_api.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/campaign_schemas.py evaluation/analytics.py tests/test_campaign_schemas.py tests/test_evaluation_analytics_api.py
git add evaluation/campaign_schemas.py evaluation/analytics.py tests/test_campaign_schemas.py tests/test_evaluation_analytics_api.py openapi.json contracts/openapi-contract.json docs/generated/api-surface.md docs/product-specs/evaluation-api.md
git commit -m "fix(evaluation): preserve unknown run tokens"
```

### Task 8: Type and filter retrospective router analysis

**Repository:** `pdftopng`

**Files:**
- Modify: `evaluation/campaign_schemas.py`
- Modify: `evaluation/analytics.py`
- Modify: `tests/test_campaign_schemas.py`
- Modify: `tests/test_evaluation_analytics_context.py`
- Modify: `tests/test_evaluation_analytics_api.py`
- Verify: `tests/test_evaluation_execution_observability.py`
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`
- Generate: `docs/generated/api-surface.md`
- Modify: `docs/product-specs/evaluation-api.md`

**Interfaces:**

```python
class RouterAnalysisRow(BaseModel):
    routing_decision_id: str
    run_id: str
    campaign_id: str
    question_id: str
    repeat_number: int = Field(ge=1)
    span_id: str | None = None
    selected_mode: CampaignMode
    analysis_type: Literal["retrospective"]
    decision_source: Literal["deterministic", "llm_planner", "safe_fallback"] | None
    candidate_routes: list[str]
    matched_rules: list[str]
    fallback_reason: str | None
    confidence: float | None
    reason: str | None
    created_at: datetime


class RouterAnalysisResponse(AnalyticsAggregateResponse):
    analysis_type: Literal["retrospective"] = "retrospective"
    rows: list[RouterAnalysisRow] = Field(default_factory=list)
```

The response never exposes arbitrary persisted `payload` and never includes an
`analysis_type="actual"` row.

- [ ] **Step 1: Write RED mixed-row tests**

Seed one retrospective and one actual decision for the same campaign. Assert:

```python
response = await service.router_analysis(user_id="user-1", campaign_id="cmp-1")
assert [row.routing_decision_id for row in response.rows] == ["retro-1"]
assert response.analysis_type == "retrospective"
assert "payload" not in response.rows[0].model_dump()
```

The API test must assert the same serialized keys and that the generated schema
uses `RouterAnalysisRow`, not an untyped row dictionary.

- [ ] **Step 2: Run RED tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py tests/test_evaluation_analytics_context.py tests/test_evaluation_analytics_api.py -k "router" -q
```

Expected: the actual row is currently mixed into the retrospective response.

- [ ] **Step 3: Filter before projection and validate typed rows**

In `_routing_decisions_for_context`, retain only decisions whose
`analysis_type == "retrospective"`. Build `RouterAnalysisRow` explicitly from
allow-listed fields plus the result's `question_id`, `repeat_number`, and
`run_id`. Do not pass `_dump(item)` wholesale into the response.

- [ ] **Step 4: Synchronize artifacts and verify**

```powershell
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py tests/test_evaluation_analytics_context.py tests/test_evaluation_analytics_api.py tests/test_evaluation_execution_observability.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/campaign_schemas.py evaluation/analytics.py tests/test_evaluation_analytics_context.py tests/test_evaluation_analytics_api.py
```

- [ ] **Step 5: Commit once**

```powershell
git add evaluation/campaign_schemas.py evaluation/analytics.py tests/test_campaign_schemas.py tests/test_evaluation_analytics_context.py tests/test_evaluation_analytics_api.py openapi.json contracts/openapi-contract.json docs/generated/api-surface.md docs/product-specs/evaluation-api.md
git commit -m "fix(evaluation): type retrospective router analysis"
```

### Task 9: Synchronize the frontend active API contracts

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/types/evaluation.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `src/services/evaluationApi.test.ts`
- Generate: `src/test/fixtures/agenticV9ApiContract.ts`

**Interfaces:**
- `EvaluationRunListItem.total_tokens: number | null`.
- Frontend progress event type has no `latest_result_id`.
- `RouterAnalysisResponse.rows: RouterAnalysisRow[]` with the exact Task 8
  fields and retrospective-only analysis type.

- [ ] **Step 1: Pin the committed backend contract, then write RED fixture tests**

```powershell
npm run contract:pin
```

Add a non-empty run-list/router fixture and compile-time assignments:

```ts
const unknownTokens: EvaluationRunListItem = {
  run_id: 'run-1', campaign_id: 'cmp-1', question_id: 'q-1',
  question: 'Question', mode: 'agentic', run_number: 1, repeat_number: 1,
  condition_id: null, execution_profile: null,
  agentic_execution_version: 'v9', response_status: null,
  status: 'completed', total_latency_ms: null,
  created_at: '2026-08-13T00:00:00Z',
  total_tokens: null,
};

expect(router.rows[0]).not.toHaveProperty('payload');
expect(router.rows[0].analysis_type).toBe('retrospective');
```

Assert the progress fixture has no `latest_result_id` property.

- [ ] **Step 2: Run RED tests and type build**

```powershell
npm test -- --run src/types/evaluation.contract.test.ts src/services/evaluationApi.test.ts
npm run build
```

- [ ] **Step 3: Replace aliases with exact frontend interfaces**

Do not add optional compatibility aliases. Update the three contracts exactly
and keep service URLs unchanged.

- [ ] **Step 4: Verify contract and build**

```powershell
npm test -- --run src/types/evaluation.contract.test.ts src/services/evaluationApi.test.ts
npm run contract:check
npm run test:scripts
npm run build
```

- [ ] **Step 5: Commit once**

```powershell
git add src/types/evaluation.ts src/types/evaluation.contract.test.ts src/services/evaluationApi.test.ts src/test/fixtures/agenticV9ApiContract.ts
git commit -m "fix(evaluation-ui): synchronize active api contracts"
```

### Task 10: Separate Router retrospective analysis from actual execution

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/pages/EvaluationCenter.tsx`
- Modify: `src/pages/EvaluationCenter.mappers.ts`
- Modify: `src/pages/EvaluationCenter.mappers.test.ts`
- Modify: `src/pages/EvaluationCenter.integration.test.tsx`
- Modify: `src/components/evaluation/RouterLabTab.tsx`
- Modify: `src/components/evaluation/RouterLabTab.test.tsx`
- Modify: `src/components/evaluation/RouterDecisionCard.tsx`
- Modify: `docs/design-docs/evaluation-center.md`

**Interfaces:**
- Router tab loader returns `{ routerAnalysis, runs, runDetail,
  selectedV9Evidence }`.
- Actual route uses `selectedV9Evidence.queryContract.route` even when
  `route_decision` is null.
- `ExecutionRoute` contains route plus optional provenance fields.
- Retrospective UI shows only recorded decision fields from Task 8.

- [ ] **Step 1: Write RED integration tests with real Router components**

In `EvaluationCenter.integration.test.tsx`, cover:

```ts
it('loads actual route on direct Router Lab entry', async () => {
  // open tab 6 without opening Run Trace first
  expect(apiMocks.getCampaignRuns).toHaveBeenCalledWith('cmp-1');
  expect(apiMocks.getRunObservability).toHaveBeenCalledWith('cmp-1', 'run-1');
  expect(await screen.findByText('graph')).toBeInTheDocument();
});

it('shows contract route without route_decision provenance', async () => {
  // agentic_v9.contract.route = 'visual'; route_decision = null
  expect(await screen.findByText('visual')).toBeInTheDocument();
});
```

Also assert selected-run failure leaves retrospective rows visible;
retrospective failure leaves the execution route visible; campaign switch
clears the old route, stays on Router Lab, and ignores a late previous response.

- [ ] **Step 2: Write RED mapper/component assertions**

Assert `mapRouterData` has no tier, complexity, KPI, utility, oracle, or matrix
properties. Assert the rendered page has no columns/labels for saved tokens,
quality loss/gain, latency, tokens, regret, formula, oracle, or confusion.

- [ ] **Step 3: Run RED tests**

```powershell
npm test -- --run src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RouterLabTab.test.tsx
```

- [ ] **Step 4: Implement independent loads and truthful components**

Make tab 6 load router analysis and selected-run data independently with the
existing generation guards. Derive execution route as:

```ts
const contract = selectedV9Evidence?.queryContract;
const decision = contract?.route_decision;
const executionRoute = contract ? {
  route: contract.route,
  decisionSource: decision?.decision_source ?? null,
  routeReason: decision?.route_reason ?? null,
  matchedRules: decision?.matched_rules ?? [],
  candidateRoutes: decision?.candidate_routes ?? [],
  fallbackReason: decision?.fallback_reason ?? null,
} : undefined;
```

Remove fabricated Router fields from mapper, props, cards, tables, and tests.

- [ ] **Step 5: Verify and commit once**

```powershell
npm test -- --run src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RouterLabTab.test.tsx
npm run build
git add src/pages/EvaluationCenter.tsx src/pages/EvaluationCenter.mappers.ts src/pages/EvaluationCenter.mappers.test.ts src/pages/EvaluationCenter.integration.test.tsx src/components/evaluation/RouterLabTab.tsx src/components/evaluation/RouterLabTab.test.tsx src/components/evaluation/RouterDecisionCard.tsx docs/design-docs/evaluation-center.md
git commit -m "fix(evaluation-ui): separate router analysis and execution"
```

### Task 11: Remove uninstrumented panel placeholders

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/pages/EvaluationCenter.mappers.ts`
- Modify: `src/pages/EvaluationCenter.mappers.test.ts`
- Modify: `src/components/evaluation/QuestionAnalysisTab.tsx`
- Modify: `src/components/evaluation/QuestionAnalysisTab.test.tsx`
- Modify: `src/components/evaluation/QuestionDeltaHeatmap.tsx`
- Modify: `src/components/evaluation/V9EvidenceExplorer.tsx`
- Modify: `src/components/evaluation/V9EvidenceExplorer.test.tsx`
- Modify: `src/components/evaluation/ClaimEvidenceTab.tsx`
- Modify: `src/components/evaluation/ClaimEvidenceTab.test.tsx`
- Modify: `src/components/evaluation/AgentBehaviorTab.tsx`
- Modify: `src/components/evaluation/AgentBehaviorTab.test.tsx`
- Modify: `docs/design-docs/evaluation-center.md`

**Interfaces:**
- Question rows have no `routerSelectedMode` or `ablationFlags`.
- V9 Evidence has no packet `Cited` or uninstrumented token breakdown rows.
- Claim alignment has no per-slot Graph column.
- Agent Behavior has no atomic-completeness column until a measured contract
  exists.
- Each affected section uses one local notice with stable test id
  `capability-notice`; do not add an API capability framework.

- [ ] **Step 1: Write RED absence and notice tests**

Use explicit header names (`Router Selected Mode`, `Cited`, `Graph`, and
`Atomic Completeness`) with `queryByRole('columnheader', { name })`. Mapper
tests assert the properties are absent, not merely set to `N/A`. Component
tests scope `getByTestId('capability-notice')` to the affected section and
assert exactly one notice there.

- [ ] **Step 2: Run RED tests**

```powershell
npm test -- --run src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/QuestionAnalysisTab.test.tsx src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/AgentBehaviorTab.test.tsx
```

- [ ] **Step 3: Remove the unsupported data paths**

Delete the mapper properties, component props, columns, and repeated `N/A`
cells. Use plain section-specific notice copy; do not create a reusable
capability engine.

- [ ] **Step 4: Verify and commit once**

```powershell
npm test -- --run src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/QuestionAnalysisTab.test.tsx src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/AgentBehaviorTab.test.tsx
npm run build
git add src/pages/EvaluationCenter.mappers.ts src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/QuestionAnalysisTab.tsx src/components/evaluation/QuestionAnalysisTab.test.tsx src/components/evaluation/QuestionDeltaHeatmap.tsx src/components/evaluation/V9EvidenceExplorer.tsx src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/ClaimEvidenceTab.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/AgentBehaviorTab.tsx src/components/evaluation/AgentBehaviorTab.test.tsx docs/design-docs/evaluation-center.md
git commit -m "fix(evaluation-ui): remove uninstrumented placeholders"
```

### Task 12: Preserve missing counts and zero durations

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/components/evaluation/AblationDashboardTab.tsx`
- Modify: `src/components/evaluation/AblationDashboardTab.test.tsx`
- Modify: `src/components/evaluation/RunTraceTree.tsx`
- Modify: `src/components/evaluation/RunTraceTree.test.tsx`
- Modify: `src/components/evaluation/RunTraceTab.tsx`
- Modify: `src/components/evaluation/RunTraceTab.test.tsx`

**Interfaces:**
- `formatCount(null | undefined | NaN | Infinity) === "N/A"`.
- `formatCount(0) === "0"`.
- `conditionRows().sampleCount` remains nullable.
- Normalized and legacy trace duration `0` renders `0 ms`.

- [ ] **Step 1: Write RED tests scoped by label/row**

```ts
expect(within(samplesCard).getByText('N/A')).toBeInTheDocument();
expect(within(failedCard).getByText('0')).toBeInTheDocument();
expect(within(zeroDurationRow).getByText('0 ms')).toBeInTheDocument();
```

Cover both `RunTraceTree` and the legacy steps branch of `RunTraceTab`.

- [ ] **Step 2: Run RED tests**

```powershell
npm test -- --run src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/RunTraceTree.test.tsx src/components/evaluation/RunTraceTab.test.tsx
```

- [ ] **Step 3: Replace truthy/default coercions**

Use finite-number checks and `value !== undefined && value !== null` for
durations. Do not change genuine backend zeros.

- [ ] **Step 4: Verify and commit once**

```powershell
npm test -- --run src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/RunTraceTree.test.tsx src/components/evaluation/RunTraceTab.test.tsx
npm run build
git add src/components/evaluation/AblationDashboardTab.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/RunTraceTree.tsx src/components/evaluation/RunTraceTree.test.tsx src/components/evaluation/RunTraceTab.tsx src/components/evaluation/RunTraceTab.test.tsx
git commit -m "fix(evaluation-ui): preserve unknown and zero values"
```

### Task 13: Mount durable jobs with stable terminal refresh

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/pages/EvaluationCenter.tsx`
- Modify: `src/pages/EvaluationCenter.integration.test.tsx`
- Modify: `src/pages/EvaluationCenter.ui.test.tsx`
- Modify: `src/components/evaluation/EvaluationJobPanel.tsx`
- Modify: `src/components/evaluation/EvaluationJobPanel.test.tsx`
- Modify: `docs/FRONTEND.md`
- Modify: `docs/design-docs/evaluation-center.md`
- Generate: `docs/generated/ui-surface.md`

**Interfaces:**
- Extract stable `loadCampaignInventory(): Promise<CampaignStatus[]>`.
- `EvaluationJobPanel` renders heading plus `No durable evaluation jobs` when
  the selected campaign has none.
- `onJobTerminal(job)` fires once per terminal job ID.
- Parent terminal handler refreshes inventory and invalidates the selected
  campaign's loaded tab key without changing `activeTabIndex`.
- Mount with `key={selectedCampaignId}`. Keep the `Durable evaluation jobs`
  heading visible during loading, empty, error, and populated states; load
  errors must be visible in the panel rather than toast-only.
- Campaign Overview no longer calls `getCampaignErrors`; diagnostics continue
  to load errors and stage warnings in the Ablation surface that renders them.

- [ ] **Step 1: Write RED empty-state and one-shot tests**

```ts
expect(await screen.findByText('No durable evaluation jobs')).toBeInTheDocument();
expect(onJobTerminal).toHaveBeenCalledTimes(1);
```

Rerender/poll the same already-terminal job and assert the callback remains one.

- [ ] **Step 2: Write RED page integration tests**

In `EvaluationCenter.ui.test.tsx`, mock `EvaluationJobPanel`, capture
`campaignId` and `onJobTerminal`, and use automatic first-campaign selection.
Switch through the real `Campaign selector`, invoke the callback, and assert
`listCampaigns` is called exactly once more. In the integration test assert the
current tab stays selected and its data refetches without resetting to Campaign
Overview. Add an initial-overview assertion that `getCampaignErrors` has not
been called, then open Ablation and assert it is called there.

- [ ] **Step 3: Run RED tests**

```powershell
npm test -- --run src/components/evaluation/EvaluationJobPanel.test.tsx src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.ui.test.tsx
```

- [ ] **Step 4: Refactor inventory loading and mount the panel**

Keep campaign inventory separate from tab payload reset logic. Mount
`EvaluationJobPanel` in the selected campaign operation area before the tabs.
Use the existing terminal ref keyed by job ID; reset it only when campaign/job
identity changes. Do not introduce `latest_result_id`.

- [ ] **Step 5: Verify, synchronize UI docs, and commit once**

```powershell
npm test -- --run src/components/evaluation/EvaluationJobPanel.test.tsx src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.ui.test.tsx
npm run docs:sync
npm run docs:check
npm run build
git add src/pages/EvaluationCenter.tsx src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.ui.test.tsx src/components/evaluation/EvaluationJobPanel.tsx src/components/evaluation/EvaluationJobPanel.test.tsx docs/FRONTEND.md docs/design-docs/evaluation-center.md docs/generated/ui-surface.md
git commit -m "feat(evaluation-ui): mount durable campaign jobs"
```

## Wave 2 Consolidated Review and Mandatory Checkpoint

- [ ] Run one consolidated review over Tasks 5-13 only. Provide the reviewer
  both repository diff ranges, the approved design, and this plan. Do not call
  per-Task reviewers.
- [ ] Correct every Critical/Important finding in separate repository-specific
  corrective commits, rerun affected suites, then call one scoped re-review.
- [ ] Run backend Wave 2 regression:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py tests/test_evaluation_analytics_context.py tests/test_evaluation_analytics_api.py tests/test_evaluation_research_api.py tests/test_evaluation_execution_observability.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/campaign_schemas.py evaluation/analytics.py
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
```

- [ ] Run frontend Wave 2 regression:

```powershell
npm test -- --run src/types/evaluation.contract.test.ts src/services/evaluationApi.test.ts src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RouterLabTab.test.tsx src/components/evaluation/QuestionAnalysisTab.test.tsx src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/AgentBehaviorTab.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/RunTraceTree.test.tsx src/components/evaluation/RunTraceTab.test.tsx src/components/evaluation/EvaluationJobPanel.test.tsx
npm run contract:check
npm run docs:check
npm run build
```

- [ ] Real system: direct-open Router Lab; verify retrospective decisions and
  actual route are separate, contract route appears without provenance, and
  campaign switching cannot show the previous route.
- [ ] Real system: verify removed placeholder columns are absent, unknown counts
  show N/A, real zeros remain zero, and zero-duration events show `0 ms`.
- [ ] Real system: verify durable jobs empty/running/terminal states, and terminal
  refresh does not switch tabs.
- [ ] Report hashes and exact results, then stop. Do not start Wave 3 until the
  user pushes and explicitly accepts Wave 2.

---

# Wave 3 — Export Schema v2

### Task 14: Define typed Export Schema v2 and content policy

**Repository:** `pdftopng`

**Files:**
- Create: `evaluation/export_schemas.py`
- Create: `tests/test_evaluation_export_v2_schemas.py`
- Modify: `tests/test_evaluation_export_redaction.py`
- Modify: `docs/product-specs/evaluation-api.md`

**Interfaces:**
- Define the v2 `ExportCampaignRequest` and `ExportCampaignResponse` in the new
  module to avoid the existing `trace_schemas`/`campaign_schemas` import cycle.
  Keep the legacy classes in `campaign_schemas.py` until Task 16 atomically
  switches the route/service imports and deletes them; no HTTP route exposes
  both shapes.
- `ExportCampaignRequest` has required/defaulted booleans
  `include_run_observability=false`, `include_raw_trace_payloads=false`,
  `include_prompt_previews=true`, `include_full_prompts=false`,
  `include_answers=true`, `include_retrieved_excerpts=true`, and
  `format: Literal["json"]="json"`.
- Define required `ExportAvailability`, generic `ExportSection[T]`,
  `ExportCampaignIdentityV2`, `ExportResultV2`, `ExportRunLatencyV2`,
  `ExportTraceEventV2`, `ExportLlmCallV2`, `ExportRunObservabilityDataV2`,
  `ExportRunV2`, `ExportSectionsV2`, `ExportMetadataV2`, and
  `ExportCampaignResponse`.
- `schema_version: Literal["2.0"]` and every named section are required.
- `ExportAvailability.status` is exactly `complete | partial |
  not_instrumented | not_available | not_applicable`; reasons are strings.
- `ExportCampaignIdentityV2` allow-lists campaign ID/name/status, benchmark ID,
  modes, repeat count, `created_at`, and `updated_at`.
- `ExportResultV2` has fixed run/question identity, question text, mode/repeat/
  condition/profile/context-policy/execution-version, response/status, nullable
  answer/ground-truth/contexts, source document IDs, nullable latency/tokens,
  and `created_at`. It has no arbitrary token/snapshot/derived dictionaries.
- `ExportRunV2` requires `result`, finite `ragas_metrics`, `accounting:
  TokenBreakdown`, `latency: ExportRunLatencyV2`, and `observability`.
- `ExportTraceEventV2` allow-lists event/run/campaign/span identity,
  parent IDs, event/schema/sequence/stage fields, timestamps/duration/status/
  retry count, plus sanitized trace payload. It has no error field.
- `ExportLlmCallV2` allow-lists call/run/campaign/span identity, provider/model,
  phase/purpose/reservation/attempt, normalized token/cost/latency/status,
  prompt/response hashes, prompt capture statuses, nullable preview/full prompt,
  and timestamp. It has no provider body, error, or arbitrary payload.
- `ExportRunObservabilityDataV2` requires the run identity/summary/accounting
  diagnostics plus typed trace, LLM, retrieval, context, tool, routing, graph,
  claim, human-rating, evidence coverage, status, and nullable v9 families.
- `ExportSectionsV2` uses exact typed composites: overview (research summary +
  nested release section), question analysis, agent behavior, router analysis,
  ablation, human comparison/queue, and errors/stage warnings.

- [ ] **Step 1: Write RED exact-shape schema tests**

```python
assert ExportCampaignRequest().include_run_observability is False
assert set(ExportSectionsV2.model_fields) == {
    "overview", "question_analysis", "agent_behavior", "router_analysis",
    "ablation", "human_evaluation", "diagnostics",
}
assert ExportCampaignResponse.model_fields["schema_version"].default == "2.0"
assert "derived_metrics" not in ExportResultV2.model_fields
assert "provider_error" not in ExportLlmCallV2.model_fields
```

Construct one fully non-empty response. Prove redacted answer/reference fields
accept `None`, while arbitrary dictionaries are absent from identity/result
models.

- [ ] **Step 2: Write RED policy-table tests**

Parameterize all 32 combinations of the five content flags. The pure policy
expectation table must state:

```python
raw_trace_allowed = request.include_raw_trace_payloads
answer_text_allowed = request.include_answers
excerpt_text_allowed = request.include_retrieved_excerpts
full_prompt_allowed = request.include_full_prompts and captured_at_execution
```

For every row, assert provider body, credentials, stack trace, unrestricted
errors, and non-trace arbitrary payloads remain excluded.

- [ ] **Step 3: Run RED tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py -k "schema_v2 or content_policy" -q
```

- [ ] **Step 4: Implement exact models and policy helpers**

Use export-specific fixed models; do not reuse `CampaignResult` or
`EvaluationRunObservabilityDetail` for custom export. Define redaction metadata
with permanent invariants:

```python
class ExportRedactionMetadata(BaseModel):
    provider_errors: Literal["excluded"] = "excluded"
    stack_traces: Literal["excluded"] = "excluded"
    credentials: Literal["redacted"] = "redacted"
```

Keep only the request/response model definitions and pure policy predicates in
this Task; do not switch the route yet.

- [ ] **Step 5: Verify and commit once**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py -k "schema_v2 or content_policy" -q
.\.venv\Scripts\python.exe -m ruff check evaluation/export_schemas.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py
git add evaluation/export_schemas.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py docs/product-specs/evaluation-api.md
git commit -m "feat(evaluation): define export schema v2"
```

### Task 15: Build campaign snapshots and shared canonical projectors

**Repository:** `pdftopng`

**Files:**
- Modify: `evaluation/observability_storage.py`
- Modify: `evaluation/research_analytics.py`
- Modify: `tests/test_evaluation_observability_repository.py`
- Modify: `tests/test_evaluation_research_analytics.py`
- Create: `tests/test_evaluation_export_v2_bulk.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class CampaignObservabilitySnapshot:
    trace_events_by_run_id: dict[str, list[EvaluationTraceEvent]]
    llm_calls_by_run_id: dict[str, list[EvaluationLlmCall]]
    retrieval_events_by_run_id: dict[str, list[EvaluationRetrievalEvent]]
    retrieval_chunks_by_run_id: dict[str, list[EvaluationRetrievalChunk]]
    context_packs_by_run_id: dict[str, list[EvaluationContextPack]]
    tool_calls_by_run_id: dict[str, list[EvaluationToolCall]]
    routing_decisions_by_run_id: dict[str, list[EvaluationRoutingDecision]]
    graph_events_by_run_id: dict[str, list[EvaluationGraphEvent]]
    graph_evidence_items_by_run_id: dict[str, list[EvaluationGraphEvidenceItem]]
    claims_by_run_id: dict[str, list[EvaluationClaim]]
    human_ratings_by_run_id: dict[str, list[EvaluationHumanRating]]
    materializations_by_run_id: dict[str, list[EvaluationV9AttemptMaterialization]]
    evidence_packets_by_run_id: dict[str, list[EvaluationEvidencePacket]]
    slot_resolutions_by_run_id: dict[str, list[EvaluationSlotResolution]]

@dataclass(frozen=True, slots=True)
class CanonicalRunObservability:
    result: CampaignResult
    token_breakdown: TokenBreakdown
    trace_events: Sequence[EvaluationTraceEvent]
    llm_calls: Sequence[EvaluationLlmCall]
    retrieval_events: Sequence[EvaluationRetrievalEvent]
    retrieval_chunks: Sequence[EvaluationRetrievalChunk]
    context_packs: Sequence[EvaluationContextPack]
    tool_calls: Sequence[EvaluationToolCall]
    routing_decisions: Sequence[EvaluationRoutingDecision]
    graph_events: Sequence[EvaluationGraphEvent]
    graph_evidence_items: Sequence[EvaluationGraphEvidenceItem]
    claims: Sequence[EvaluationClaim]
    human_ratings: Sequence[EvaluationHumanRating]
    agentic_v9: V9ExecutionObservability | None
    graph_observability_status: Literal[
        "recorded", "fallback", "not_instrumented"
    ]
    claim_extraction_status: Literal["recorded", "empty", "not_instrumented"]
    evidence_coverage: list[dict[str, Any]] | None
    evidence_coverage_status: Literal[
        "complete", "partial", "not_available", "not_instrumented"
    ]
```

Exact callable signatures are:

- `EvaluationObservabilityRepository.load_campaign_observability_snapshot(
  campaign_id: str) -> CampaignObservabilitySnapshot`;
- `_build_canonical_run_observability(*, result: CampaignResult,
  observability: CampaignObservabilitySnapshot, accounting:
  CampaignAccountingSnapshot) -> CanonicalRunObservability`;
- `_project_interactive_run_observability(canonical:
  CanonicalRunObservability) -> EvaluationRunObservabilityDetail`;
- `_token_breakdown_for_run(*, result: CampaignResult, accounting:
  CampaignAccountingSnapshot, llm_calls: Sequence[EvaluationLlmCall]) ->
  TokenBreakdown`;
- `_official_ragas_by_run(*, results: Sequence[CampaignResult |
  CampaignResearchResult], scores: Sequence[dict[str, Any]], work_metadata:
  Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]`;
- `ResearchAnalyticsService.get_official_ragas_by_run(*, user_id: str,
  campaign_id: str, results: Sequence[CampaignResult]) -> dict[str,
  dict[str, float]]`; and
- `ResearchAnalyticsService.get_campaign_run_observability(*, user_id: str,
  campaign_id: str, results: Sequence[CampaignResult]) -> dict[str,
  CanonicalRunObservability]`.

- [ ] **Step 1: Write RED repository snapshot test**

Seed two target runs, another-campaign run, two attempts for one run, and every
normalized family. Assert campaign isolation, deterministic ordering, and that
both attempts survive. Reuse `load_campaign_release_snapshot`; add only missing
campaign loaders for retrieval events and tool calls.

- [ ] **Step 2: Write RED canonical/no-N+1 tests**

Use `AsyncMock(spec_set=EvaluationObservabilityRepository)` and an accounting
mock spec-set to `EvaluationAccountingStore`. Make every per-run/per-attempt method raise
`AssertionError`. Assert one observability snapshot and one
`accounting.load_campaign_snapshot()` for 1 run and again for 50 runs. Reverse
attempt order and assert the builder selects `result.source_attempt_id`.

Seed more than 100 rows plus a final sentinel in each event family and assert
the sentinel remains. Missing run container or malformed current v9
materialization must raise `AppError`, not become an empty/legacy detail.

- [ ] **Step 3: Write RED parity tests for shared calculations**

Assert selected-run and bulk paths produce identical `TokenBreakdown`. Include
v9 LLM provider attempts so the helper cannot omit `llm_calls`. Assert official
RAGAS excludes wrong attempt, noncanonical evaluator identity, and non-finite
scores; Question and Agent Behavior use the same helper.

- [ ] **Step 4: Run RED tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_observability_repository.py tests/test_evaluation_research_analytics.py tests/test_evaluation_export_v2_bulk.py -k "snapshot or canonical or bulk or official_ragas or token_breakdown" -q
```

- [ ] **Step 5: Implement the snapshot and pure projectors**

Preserve materializations as lists; never use the existing run-keyed
`list_v9_attempt_materializations_for_campaign()` as the full-export source.
Make the existing selected-run method load/build/project through the new pure
boundary without changing its HTTP JSON.

- [ ] **Step 6: Verify and commit once**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_observability_repository.py tests/test_evaluation_research_analytics.py tests/test_evaluation_export_v2_bulk.py tests/test_evaluation_research_api.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/observability_storage.py evaluation/research_analytics.py tests/test_evaluation_observability_repository.py tests/test_evaluation_research_analytics.py tests/test_evaluation_export_v2_bulk.py
git add evaluation/observability_storage.py evaluation/research_analytics.py tests/test_evaluation_observability_repository.py tests/test_evaluation_research_analytics.py tests/test_evaluation_export_v2_bulk.py
git commit -m "refactor(evaluation): build campaign observability snapshots"
```

### Task 16: Compose, sanitize, and serve Export Schema v2

**Repository:** `pdftopng`

**Files:**
- Create: `evaluation/export_service.py`
- Modify: `evaluation/campaign_schemas.py`
- Modify: `evaluation/router.py`
- Modify: `evaluation/analytics.py`
- Modify: `tests/test_evaluation_export_redaction.py`
- Modify: `tests/test_evaluation_analytics_api.py`
- Modify: `tests/test_evaluation_analytics_context.py`
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`
- Generate: `docs/generated/api-surface.md`
- Modify: `docs/product-specs/evaluation-api.md`
- Modify: `docs/BACKEND.md`

**Interfaces:**

`EvaluationExportService.export_campaign(*, user_id: str, campaign_id: str,
request: ExportCampaignRequest) -> ExportCampaignResponse` is the only export
composer entry point. `_project_export_run_observability(*, canonical:
CanonicalRunObservability, request: ExportCampaignRequest) ->
ExportRunObservabilityDataV2` is the only detailed-run content-policy boundary.

Router dependency is `get_evaluation_export_service()`. Remove the old
`EvaluationAnalyticsService.export_campaign()` and export-only direct SQL/helper
code after `git grep -n "export_campaign"` proves no old caller remains.

- [ ] **Step 1: Write RED composer/section tests**

Mock canonical services with non-empty typed responses. Assert exact named
sections, official RAGAS/accounting/latency per run, default observability
`included=false/data=null`, and that no legacy overview/comparison helper is
called. Assert the campaign observability snapshot loader is not called when
`include_run_observability=false`. A no-benchmark release report is nested as
`not_applicable`.

- [ ] **Step 2: Complete the 32-row redaction matrix**

Assert `include_answers=false` clears result answer, answer preview, claim/final
claim statements, and fact text. Assert `include_retrieved_excerpts=false`
clears chunk excerpts and evidence statements but retains locators. Raw trace
controls only sanitized `trace_events[].payload`. Full prompt requires both the
flag and captured-at-execution status.

- [ ] **Step 3: Write RED all-or-error/auth tests**

Assert required section failure returns a non-200 API response with no partial
v2 body, missing canonical run fails, result IDs match exactly, event arrays are
not truncated, and another user cannot export the campaign.

- [ ] **Step 4: Run RED tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_api.py tests/test_evaluation_analytics_context.py tests/test_evaluation_export_v2_bulk.py -q
```

- [ ] **Step 5: Implement the composer and switch the route**

Load ownership/full results first. Assemble independent active panel services
with `asyncio.gather`. Load full observability only when requested. Propagate
required failures. Apply the export allow list once in the export projector.
Delete old response assembly and direct export RAGAS SQL.

- [ ] **Step 6: Synchronize OpenAPI and verify**

```powershell
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_evaluation_export_v2_bulk.py tests/test_evaluation_analytics_api.py tests/test_evaluation_analytics_context.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/export_schemas.py evaluation/export_service.py evaluation/campaign_schemas.py evaluation/router.py evaluation/analytics.py tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_api.py
```

- [ ] **Step 7: Commit once**

```powershell
git add evaluation/export_service.py evaluation/campaign_schemas.py evaluation/router.py evaluation/analytics.py tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_api.py tests/test_evaluation_analytics_context.py openapi.json contracts/openapi-contract.json docs/generated/api-surface.md docs/product-specs/evaluation-api.md docs/BACKEND.md
git commit -m "feat(evaluation): serve sanitized export schema v2"
```

### Task 17: Consume Export v2 with a strict runtime boundary

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/types/evaluation.ts`
- Create: `src/services/evaluationExportSchema.ts`
- Create: `src/services/evaluationExportSchema.test.ts`
- Modify: `src/services/evaluationApi.ts`
- Modify: `src/services/evaluationApi.test.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `src/components/evaluation/AblationDashboardTab.tsx`
- Modify: `src/components/evaluation/AblationDashboardTab.test.tsx`
- Generate: `src/test/fixtures/agenticV9ApiContract.ts`
- Modify: `docs/FRONTEND.md`
- Modify: `docs/design-docs/evaluation-center.md`
- Generate: `docs/generated/ui-surface.md`

**Interfaces:**
- Exact TS interfaces mirror the Task 14 names; `ExportRunV2.result` is not
  `CampaignResult` and the response does not extend `Record<string, unknown>`.
- `evaluationExportV2Schema` is a Zod schema and
  `parseExportCampaignResponse(value: unknown): ExportCampaignResponse`
  validates before download. Validation errors are sanitized and never echo
  response payload content.
- `ExportCampaignRequest.include_run_observability: boolean` is required in the
  request sent by this UI.

- [ ] **Step 1: Pin backend and write RED contract/decoder tests**

```powershell
npm run contract:pin
```

Use one non-empty exact v2 fixture. Assert parsing succeeds, and missing
`sections.diagnostics`, wrong `schema_version`, or a run result with arbitrary
shape throws. This is the real RED gate; a loose Axios generic alone is not.

- [ ] **Step 2: Write RED UI/request/filename tests**

Assert checkbox `Include all run observability` is unchecked and the POST body
contains `include_run_observability: false`. Cover all filenames:

```text
cmp-1-summary-redacted-v2.json
cmp-1-observability-redacted-v2.json
cmp-1-summary-custom-v2.json
cmp-1-observability-custom-v2.json
```

Assert server `export_metadata.options`, redaction, and warnings drive preview;
summary mode does not display a fabricated zero LLM-call count. Cover pending
disable, one click/one request, rejection with no download and old preview
preserved, campaign-change stale response ignored, and object URL revoked.

- [ ] **Step 3: Run RED tests/build**

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx
npm run build
```

- [ ] **Step 4: Implement strict types, decoder, and default-off UI**

Parse `response.data` at the API boundary. Calculate filename only from the
request scope/custom flags. Read preview exclusively from v2 metadata, sections,
and runs; remove all old `summary`, `llm_calls`, and top-level `redaction`
fallbacks.

- [ ] **Step 5: Verify contract/docs/build and commit once**

Manually replace the legacy export/request wording in `docs/FRONTEND.md`,
`docs/design-docs/evaluation-center.md`, and `docs/generated/ui-surface.md`.
`docs:sync` only refreshes generated markers and does not prove those narrative
sections are current.

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx src/pages/EvaluationCenter.ui.test.tsx
npm run contract:check
npm run docs:sync
npm run docs:check
npm run docs:links
npm run test:scripts
npm run build
git add src/types/evaluation.ts src/services/evaluationExportSchema.ts src/services/evaluationExportSchema.test.ts src/services/evaluationApi.ts src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/test/fixtures/agenticV9ApiContract.ts docs/FRONTEND.md docs/design-docs/evaluation-center.md docs/generated/ui-surface.md
git commit -m "feat(evaluation-ui): consume export schema v2"
```

## Wave 3 Consolidated Review and Mandatory Checkpoint

- [ ] Call one consolidated reviewer for Tasks 14-17 across both repositories.
  Give special attention to content-policy leaks, exact source-attempt selection,
  N+1 behavior, fixed v2 shapes, and frontend runtime validation.
- [ ] Correct Critical/Important findings inside Wave 3 and call one scoped
  re-review.
- [ ] Run backend Wave 3 suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_evaluation_export_v2_bulk.py tests/test_evaluation_observability_repository.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_analytics_api.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/export_schemas.py evaluation/export_service.py evaluation/observability_storage.py evaluation/research_analytics.py evaluation/router.py
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
```

- [ ] Run frontend Wave 3 suite:

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx src/pages/EvaluationCenter.ui.test.tsx
npm run contract:check
npm run docs:check
npm run docs:links
npm run test:scripts
npm run build
```

- [ ] Real system: export default summary and confirm every run has accounting,
  latency, official RAGAS, and `observability.data=null`.
- [ ] Real system: enable all-run observability and confirm exact run-ID equality,
  selected sample parity, no missing tail events, and sensible file size.
- [ ] Real system: test four filenames and answer/excerpt/raw/full-prompt policies
  without copying secrets into the report.
- [ ] Report hashes, test totals, and bulk-loader call counts, then stop until the
  user pushes and accepts Wave 3.

---

# Wave 4 — HTTP Parity and Release Gate

### Task 18: Prove authenticated HTTP panel/export parity

**Repository:** `pdftopng`

**Files:**
- Create: `tests/test_evaluation_export_v2_http_parity.py`
- Modify: `docs/evaluation-center.md`
- Modify: `docs/BACKEND.md`

**Interfaces:**
- Use real FastAPI routes, authenticated ownership, temporary durable storage,
  and one fixture containing two modes, official RAGAS/accounting, v9 evidence,
  routing, ablation, human rating, error, and stage warning.
- Compare complete serialized section objects, not selected scalar fields.

- [ ] **Step 1: Write the authenticated parity test**

Call every active endpoint and both export modes. Assert:

```python
assert exported["sections"]["overview"]["data"]["research_summary"] == research
assert exported["sections"]["overview"]["data"]["release_metrics"]["data"] == release
assert exported["sections"]["question_analysis"]["data"] == question
assert exported["sections"]["agent_behavior"]["data"] == behavior
assert exported["sections"]["router_analysis"]["data"] == router
assert exported["sections"]["ablation"]["data"] == ablation
assert exported["sections"]["human_evaluation"]["data"]["comparison"] == human
assert exported["sections"]["human_evaluation"]["data"]["queue"] == queue
assert exported["sections"]["diagnostics"]["data"]["errors"] == errors
assert exported["sections"]["diagnostics"]["data"]["stage_warnings"] == warnings
assert exported_full["runs"][0]["observability"]["data"] == interactive
```

The last equality uses safe-default content options. Test no-benchmark nested
`not_applicable`, unauthorized 404, required-section failure/no partial v2, and
exact campaign result ID equality.

- [ ] **Step 2: Add query-count invariance**

Run the same full export fixture with 1 and 50 runs. Assert campaign snapshot
and accounting loads remain one each and query count stays within a fixed
constant allowance, not a per-run slope.

- [ ] **Step 3: Run the RED/GREEN parity test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_v2_http_parity.py -q
```

If the test exposes a production parity defect, stop this Task and create one
separate corrective implementation Task/commit with the failing assertion.
Rerun Step 3 after that corrective commit. Do not hide production changes in
the test/documentation commit and do not weaken equality assertions.

- [ ] **Step 4: Document the verified contract and run backend release suite**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_v2_http_parity.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_evaluation_export_v2_bulk.py tests/test_evaluation_research_api.py tests/test_evaluation_research_analytics.py tests/test_evaluation_analytics_api.py tests/test_evaluation_observability_repository.py tests/test_evaluation_execution_observability.py tests/test_evaluation_graph_events.py tests/test_evaluation_human_ratings.py tests/test_evaluation_job_worker.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation tests/test_evaluation_export_v2_http_parity.py
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
```

- [ ] **Step 5: Commit once**

```powershell
git add tests/test_evaluation_export_v2_http_parity.py docs/evaluation-center.md docs/BACKEND.md
git commit -m "test(evaluation): prove panel export http parity"
```

### Task 19: Lock frontend runtime contract and release documentation

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/services/evaluationApi.test.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `src/components/evaluation/AblationDashboardTab.test.tsx`
- Modify: `docs/FRONTEND.md`
- Modify: `docs/design-docs/evaluation-center.md`
- Generate: `docs/generated/ui-surface.md`
- Generate only if backend HEAD changed: `src/test/fixtures/agenticV9ApiContract.ts`

**Interfaces:**
- Runtime decoder rejects missing/wrong named sections before download.
- Export component integration covers summary/full request and preview only.
- Router, campaign-switch, and durable-job behavior remain in their focused
  Wave 2 tests; do not duplicate them into one giant scenario.

- [ ] **Step 1: Add final non-empty contract regression cases**

Assert the decoder rejects wrong nested release shape, missing run accounting,
and observability `included=true/data=null`. Assert the UI makes no download for
an invalid response and retains the previous valid preview.

- [ ] **Step 2: Run focused frontend tests**

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx
```

- [ ] **Step 3: Synchronize docs and contract**

Read all three Evaluation Center documentation sections and compare them to the
mounted service calls, durable-job UI, and v2 export action before running the
mechanical checks; a passing `docs:check` alone is insufficient.

```powershell
npm run contract:check
npm run docs:sync
npm run docs:check
npm run docs:links
npm run test:scripts
```

Run `npm run contract:pin` only if the committed backend OpenAPI HEAD changed
after Task 17; rerun `contract:check` afterward.

- [ ] **Step 4: Run the frontend release suite**

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.ui.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RunTraceTab.test.tsx src/components/evaluation/RunTraceTree.test.tsx src/components/evaluation/RetrievalEvidenceTab.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/RouterLabTab.test.tsx src/components/evaluation/QuestionAnalysisTab.test.tsx src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/AgentBehaviorTab.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/EvaluationJobPanel.test.tsx
npm run lint:ci
npm run build
npm run contract:check
npm run docs:check
npm run docs:links
npm run test:scripts
```

- [ ] **Step 5: Commit once**

```powershell
git add src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx docs/FRONTEND.md docs/design-docs/evaluation-center.md docs/generated/ui-surface.md
git add src/test/fixtures/agenticV9ApiContract.ts  # only when Step 3 regenerated it
git commit -m "test(evaluation-ui): lock export contract release gate"
```

## Wave 4 Consolidated Review and Final Mandatory Checkpoint

- [ ] Call exactly one consolidated reviewer over Tasks 18-19 and any Task 18
  parity corrections. Point it to the complete Wave 2-4 ledger and deferred
  findings.
- [ ] If review finds Critical/Important issues, dispatch one corrective
  subagent with the complete list, create separate corrective commits, run the
  affected release suites, and call one scoped re-review.
- [ ] Re-run Task 18 backend release commands and Task 19 frontend release
  commands without omitting failures or warning counts.
- [ ] Confirm `scripts/sync_openapi_artifacts.py --check`, frontend
  `contract:check`, docs checks, lint, and build all pass against committed HEADs.
- [ ] Confirm both repositories have no task-related uncommitted files.
- [ ] Report the complete Wave 1-4 commit ledger and ask the user to push both
  repositories.
- [ ] Real system: execute `docs/evaluation-center.md` using a v9 evidence run,
  legacy/v8 run, missing-instrumentation run, failed/partial run if available,
  durable terminal job, and multi-run export campaign.
- [ ] Stop for release acceptance. Do not merge, tag, deploy, or start unrelated
  cleanup without a new instruction.

## Corrective Task Protocol

If a real-system checkpoint fails:

1. remain in the current Wave;
2. record safe request/response shape and UI symptom without secrets;
3. add one failing regression test for the observed contract;
4. implement the smallest fix;
5. run the Wave suite;
6. create a separate `fix(evaluation): correct wave N checkpoint failure` or
   `fix(evaluation-ui): correct wave N checkpoint failure` commit;
7. run one scoped consolidated re-review; and
8. repeat the same checkpoint.

The next Wave begins only after explicit acceptance of the corrected checkpoint.
