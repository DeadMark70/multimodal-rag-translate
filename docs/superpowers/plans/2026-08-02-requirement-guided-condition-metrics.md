# Requirement-Guided Condition Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show reproducible Requirement-guided V9 baseline/guided quality, token, and latency comparisons in the Ablation tab and redacted campaign JSON.

**Architecture:** The backend is the sole condition-metric aggregator. It joins persisted run snapshots with owned RAGAS score rows, groups by immutable `condition_id`, and creates finite-only matched-pair deltas using `(question_id, repeat_number)`. The `/ablation` API, export, and React display consume this same projection; mode-level analytics is unchanged.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, SQLite/aiosqlite, pytest; React, TypeScript, Chakra UI, Vitest.

## Global Constraints

- Use persisted `condition_id`, label, and flags; never read current environment state for historical analytics.
- Failed runs, missing/non-finite metrics, and missing opposite conditions are exclusion reasons, never zero values.
- Preserve legacy `/ablation` summaries when fewer than two conditions exist.
- Do not change Agentic V9 runtime behavior or its process-wide default.
- Update backend/frontend evaluation docs and generated inventories in the code change.

---

## File Structure

- `pdftopng/evaluation/campaign_schemas.py`: typed aggregate and paired-delta response models.
- `pdftopng/evaluation/analytics.py`: shared condition comparison builder for `/ablation` and export.
- `pdftopng/tests/test_evaluation_analytics_api.py`: condition aggregate and pair-exclusion regression coverage.
- `pdftopng/tests/test_evaluation_export_redaction.py`: export metrics and redaction coverage.
- `Multimodal_RAG_System/src/types/evaluation.ts`: matching client contract.
- `Multimodal_RAG_System/src/components/evaluation/AblationDashboardTab.tsx`: condition and delta presentation.
- `Multimodal_RAG_System/src/components/evaluation/AblationDashboardTab.test.tsx`: UI coverage.

### Task 1: Typed backend condition aggregate

**Files:**
- Modify: `pdftopng/evaluation/campaign_schemas.py:535`
- Modify: `pdftopng/evaluation/analytics.py:755`
- Test: `pdftopng/tests/test_evaluation_analytics_api.py`

**Interfaces:**
- Consumes: `CampaignResult.derived_metrics`, `_ragas_metrics_for_campaign(...)`, `AblationResponse`.
- Produces: `_build_condition_comparison(context, ragas_by_run) -> dict[str, Any] | None`.

- [ ] **Step 1: Write failing aggregate and failed-pair tests**

```python
comparison = body["summaries"]["condition_comparison"]
assert comparison["conditions"]["v9-guided"]["quality"]["answer_correctness"]["mean"] == pytest.approx(0.8)
assert comparison["paired"]["completed_pair_count"] == 1
assert comparison["paired"]["excluded_pairs"]["run_not_completed"] == 1
assert comparison["paired"]["delta"]["answer_correctness"]["mean"] == pytest.approx(0.2)
```

- [ ] **Step 2: Verify the test fails**

Run: `..\\.venv\\Scripts\\python.exe -m pytest tests/test_evaluation_analytics_api.py -k condition_comparison -q`

Expected: `condition_comparison` is absent.

- [ ] **Step 3: Add models and finite-only aggregate builder**

```python
class ConditionMetricSummary(BaseModel):
    mean: float | None = None
    valid_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)

class ConditionComparisonResponse(BaseModel):
    conditions: dict[str, ConditionAggregate] = Field(default_factory=dict)
    paired: ConditionPairedComparison | None = None
    availability: ConditionMetricAvailability
```

Group completed and failed rows by `condition_id`. Pair only completed runs with
equal question id/repeat number and finite values in both arms. Prefer
`v9-baseline` as baseline; otherwise use campaign configuration order and then
lexical order. Preserve the numeric count of each excluded reason.

- [ ] **Step 4: Wire RAGAS rows into the existing ablation endpoint**

```python
async def ablation(self, *, user_id: str, campaign_id: str) -> AblationResponse:
    context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
    ragas_by_run = await self._ragas_metrics_for_campaign(user_id=user_id, campaign_id=campaign_id)
    return self._build_ablation(context, ragas_by_run=ragas_by_run)
```

Insert `condition_comparison` only for campaigns with two or more persisted
condition ids. Retain `condition_counts` and graph-family summaries unchanged.

- [ ] **Step 5: Verify focused backend analytics tests pass**

Run: `..\\.venv\\Scripts\\python.exe -m pytest tests/test_evaluation_analytics_api.py tests/test_evaluation_research_analytics.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add evaluation/campaign_schemas.py evaluation/analytics.py tests/test_evaluation_analytics_api.py
git commit -m feat-evaluation-condition-analytics
```

### Task 2: Reuse the condition projection in redacted export

**Files:**
- Modify: `pdftopng/evaluation/analytics.py:export_campaign`
- Test: `pdftopng/tests/test_evaluation_export_redaction.py`
- Modify: `pdftopng/docs/BACKEND.md`
- Modify: `pdftopng/docs/generated/api-surface.md`
- Modify: `pdftopng/docs/product-specs/evaluation-api.md`

**Interfaces:**
- Consumes: `_build_condition_comparison(...)`, owned RAGAS rows, `ExportCampaignResponse.metrics`.
- Produces: `metrics.condition_comparison` and finite-only `runs[*].ragas_metrics`.

- [ ] **Step 1: Write failing export tests**

```python
assert payload["metrics"]["condition_comparison"]["paired"]["completed_pair_count"] == 1
assert payload["runs"][0]["ragas_metrics"] == {"answer_correctness": 0.7}
assert "full_prompt" not in json.dumps(payload)
```

- [ ] **Step 2: Verify the export test fails**

Run: `..\\.venv\\Scripts\\python.exe -m pytest tests/test_evaluation_export_redaction.py -k condition -q`

Expected: per-run RAGAS metrics and condition comparison are absent.

- [ ] **Step 3: Add export projection without changing redaction**

```python
ragas_by_run = await self._ragas_metrics_for_campaign(user_id=user_id, campaign_id=campaign_id)
row["ragas_metrics"] = dict(sorted(ragas_by_run.get(result.id, {}).items()))
metrics["condition_comparison"] = self._build_condition_comparison(
    context=context, ragas_by_run=ragas_by_run
)
```

Filter non-finite scores before serializing. Historical unscored runs export an
empty metric map. Prompts, answers, excerpts, and provider-error redaction
rules remain exactly as they are.

- [ ] **Step 4: Document backend fields and verify export tests**

Run: `..\\.venv\\Scripts\\python.exe -m pytest tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_api.py -q`

Expected: PASS after adding the API guide, generated API inventory, and product
spec descriptions for the new condition fields.

- [ ] **Step 5: Commit Task 2**

```bash
git add evaluation/analytics.py tests/test_evaluation_export_redaction.py docs/BACKEND.md docs/generated/api-surface.md docs/product-specs/evaluation-api.md
git commit -m feat-evaluation-export-condition-metrics
```

### Task 3: Render condition metrics in the Ablation tab

**Files:**
- Modify: `Multimodal_RAG_System/src/types/evaluation.ts:483`
- Modify: `Multimodal_RAG_System/src/components/evaluation/AblationDashboardTab.tsx:86`
- Test: `Multimodal_RAG_System/src/components/evaluation/AblationDashboardTab.test.tsx`

**Interfaces:**
- Consumes: `AblationResponse.summaries.condition_comparison` from Task 1.
- Produces: `Condition Metrics` and `Paired Delta (guided - baseline)` sections.

- [ ] **Step 1: Write failing A/B and N/A display tests**

```tsx
expect(screen.getByText('Condition Metrics')).toBeInTheDocument();
expect(screen.getByText('Requirement guidance on')).toBeInTheDocument();
expect(screen.getByText('Paired Delta (guided - baseline)')).toBeInTheDocument();
expect(screen.getAllByText('N/A').length).toBeGreaterThan(0);
```

- [ ] **Step 2: Verify the component test fails**

Run: `npm test -- --run src/components/evaluation/AblationDashboardTab.test.tsx`

Expected: the condition-metrics section is absent.

- [ ] **Step 3: Add client types and presentation-only mappers**

```ts
export interface ConditionMetricSummary {
  mean: number | null;
  valid_count: number;
  missing_count: number;
}

export interface ConditionComparisonSummary {
  conditions: Record<string, ConditionAggregate>;
  paired?: ConditionPairedComparison | null;
  availability: ConditionMetricAvailability;
}
```

Read `condition_comparison` from the existing summary record. Render each
condition's quality means, tokens, latency, completed/failed counts, and a
paired delta table. Use `N/A` for `null`; display exclusion counts rather than
fabricating complete pairs. React must not aggregate metrics or read the env
flag.

- [ ] **Step 4: Preserve legacy rendering and verify focused tests**

```tsx
{conditionComparison ? <ConditionMetricsSection comparison={conditionComparison} /> : null}
```

The original condition-count and graph-family sections remain. Campaigns
without condition comparison data retain the current display.

Run: `npm test -- --run src/components/evaluation/AblationDashboardTab.test.tsx src/services/evaluationApi.test.ts`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/types/evaluation.ts src/components/evaluation/AblationDashboardTab.tsx src/components/evaluation/AblationDashboardTab.test.tsx
git commit -m feat-evaluation-ui-condition-metrics
```

### Task 4: Documentation and CI-equivalent verification

**Files:**
- Modify: `Multimodal_RAG_System/docs/FRONTEND.md`
- Modify: `Multimodal_RAG_System/docs/generated/ui-surface.md`
- Modify: `Multimodal_RAG_System/docs/product-specs/evaluation-results-and-traces.md`
- Test: backend/frontend suites from Tasks 1–3

**Interfaces:**
- Consumes: final `/ablation` and export response contract.
- Produces: documented A/B behavior and verified commits.

- [ ] **Step 1: Document the presentation contract**

State that the Ablation tab, not Mode Comparison, renders condition-level
quality/token/latency and finite-only paired deltas. State that missing
metrics are `N/A` and failed/unpaired rows are excluded with a reason.

- [ ] **Step 2: Run backend verification**

Run: `..\\.venv\\Scripts\\python.exe -m pytest tests/test_evaluation_analytics_api.py tests/test_evaluation_export_redaction.py tests/test_evaluation_research_analytics.py -q`

Expected: PASS.

- [ ] **Step 3: Run frontend CI-equivalent checks**

Run: `npm run lint:ci && npx tsc --noEmit && npm test -- --run src/components/evaluation/AblationDashboardTab.test.tsx && npm run build`

Expected: PASS.

- [ ] **Step 4: Commit Task 4**

```bash
git add docs/FRONTEND.md docs/generated/ui-surface.md docs/product-specs/evaluation-results-and-traces.md
git commit -m docs-evaluation-ui-condition-comparison
```

## Plan Self-Review

- Spec coverage: Tasks 1–2 provide condition aggregation, valid-pair deltas,
  export metrics, and fail-closed handling; Task 3 provides the A/B UI; Task 4
  updates both stacks' contract documentation and verification.
- Placeholder scan: no deferred implementation markers or unspecified testing
  steps remain.
- Type consistency: Task 1 produces the condition comparison, Task 2 reuses
  it unchanged, and Task 3 mirrors the response shape without client-side
  recomputation.
