# Evaluation Center Trustworthy Analytics Implementation Plan

> **For implementation:** the primary agent writes all production code.  If the user requests it at implementation time, use exactly one subagent as a read-only reviewer after the relevant code and tests exist; do not delegate coding to that subagent.

**Goal:** Make every Evaluation Center page distinguish a measured value from an unavailable value, and connect Question Analysis, Run Trace, Retrieval Evidence, Agent Behavior, and Router Lab to durable per-run observability instead of placeholders.

**Architecture:** Keep the existing campaign-overview aggregates as the canonical campaign-level view. Add typed, durable read projections for (1) per-question/per-mode comparison, (2) selected-run detail and accounting, and (3) bulk agent behavior. The frontend consumes those projections directly and uses nullable fields plus explicit status instead of numeric fallback values.

**Tech stack:** FastAPI, Pydantic, SQLAlchemy/repositories, pytest; React + TypeScript + Vitest.

## Scope and non-negotiable rules

- This is a **token-only** Evaluation Center. Do not add or restore USD cost columns, `$0.000` fallbacks, pricing comparability, or cost-based router metrics.
- `0`, `0.00`, `0.0%`, and `1.00` may be shown only when the backend explicitly supplies that measured value. `null`, unavailable, partial, absent instrumentation, or inapplicable data must render as `N/A` (with a concise reason where available).
- Preserve fail-closed token comparability. A question or mode with incomplete accounting must not receive a token delta, ECR, or tokens-vs-quality comparison.
- Quality fields come only from durable RAGAS records. Do not infer correctness or faithfulness from missing claim ratios.
- Do not backfill old campaigns by guessing. Historical runs without telemetry stay `N/A` / `not instrumented`; new runs gain full telemetry through normal persistence.
- Retrospective router analyses must remain explicitly labelled retrospective. They are not actual router executions.
- Keep existing routes backward compatible where practical. Introduce typed research/observability projections rather than silently changing a generic legacy response.
- Each numbered task is a separate, reviewable commit. Tests are written first and run before the commit.

## Data-contract decisions to approve

1. **Question baseline:** use `naive` as the baseline only when that question has valid quality values in both modes and complete token accounting for the two compared observations. Otherwise return a null delta and an explanatory comparability reason.
2. **Best mode:** calculate `best_quality_mode` using highest answer correctness, then faithfulness, then lower complete total tokens, then mode name as the deterministic final tie-break. If no valid quality observations exist, return `null`.
3. **Unavailable evidence fields:** coverage and unsupported-claim ratio remain `null` until their required evidence/claim instrumentation exists. They must not be derived from “no rows returned.”
4. **Agent behavior coverage:** return every run, but attach `recorded`, `not_instrumented`, or `not_applicable` behavior status. Counts are nullable; a recorded count of zero remains a real zero.

---

### Task 1: Define typed, null-safe research comparison projections (P0)

**Files:**

- Modify `D:\flutterserver\pdftopng\evaluation\campaign_schemas.py`
- Modify `D:\flutterserver\pdftopng\evaluation\research_analytics.py`
- Modify `D:\flutterserver\pdftopng\evaluation\analytics.py`
- Modify the evaluation router that currently serves campaign analytics (locate the existing `question-comparison` route)
- Add/modify `D:\flutterserver\pdftopng\tests\test_evaluation_research_analytics.py`

**Purpose:** replace the current minimal `question-comparison` payload (`sample count`, sorted mode names, mean tokens) with a typed research projection that contains only actually derived data.

**Step 1 — write failing backend tests.** Cover all of the following before implementation:

```python
assert row.delta_correctness == pytest.approx(agentic.correctness - naive.correctness)
assert row.delta_tokens is None  # one compared run has partial accounting
assert row.ecr_correctness is None
assert row.best_quality_mode == "agentic"
assert row.evidence_coverage is None
assert row.unsupported_claim_ratio is None
assert row.comparability_reason == "incomplete_accounting"
```

Also test no-baseline, missing-RAGAS, equal-quality deterministic tie-breaks, and category/difficulty/modality values taken from the persisted question snapshot rather than a frontend default.

**Step 2 — add explicit response models.** Add models equivalent to:

```python
class QuestionModeResearchMetrics(BaseModel):
    mode: CampaignMode
    sample_count: int
    answer_correctness: float | None
    faithfulness: float | None
    answer_relevancy: float | None
    mean_latency_ms: float | None
    total_tokens: int | None
    quality_status: MetricAvailability
    accounting_status: AccountingStatus

class QuestionResearchComparisonRow(BaseModel):
    question_id: str
    category: str | None
    difficulty: str | None
    required_modalities: list[str] | None
    by_mode: list[QuestionModeResearchMetrics]
    delta_correctness: float | None
    delta_faithfulness: float | None
    delta_latency_ms: float | None
    delta_tokens: float | None
    ecr_correctness: float | None
    best_quality_mode: CampaignMode | None
    evidence_coverage: float | None
    unsupported_claim_ratio: float | None
    comparability_reason: str | None
```

Use existing status enums where available; otherwise introduce narrow string literals (`complete`, `partial`, `not_available`, `not_instrumented`) rather than boolean “valid” flags.

**Step 3 — implement the durable aggregation.** Add a research-analytics method (for example `get_question_comparison`) that joins:

- persisted campaign results for question/mode/repeat identity and question snapshot metadata;
- durable RAGAS records for correctness, faithfulness, and relevancy;
- the same strict token-accounting source used by the token-only campaign overview;
- measured run latency.

Group repeat observations by question and mode. Only calculate a delta when the exact required observations exist. Use the approved baseline and best-mode rule above. Never call `numberValue`, substitute a sorted label list, or convert an absent metric to zero.

Leave the old generic `analytics.py::_build_question_comparison` route in place only if another consumer needs it; move Evaluation Center to the new typed projection so compatibility does not force ambiguous fields into the new contract.

**Step 4 — expose and document the route.** Add a route such as:

```python
@router.get("/campaigns/{campaign_id}/research-question-comparison",
            response_model=QuestionResearchComparisonResponse)
```

It must return a successful empty/partial response for old campaigns, with explicit availability states, rather than raising or manufacturing rows.

**Step 5 — verify and commit.**

```powershell
pytest tests/test_evaluation_research_analytics.py -q
git add evaluation/campaign_schemas.py evaluation/research_analytics_service.py evaluation/analytics.py tests/evaluation/test_research_question_analytics.py
git commit -m "feat(evaluation): add trustworthy question comparisons"
```

### Task 2: Replace Question Analysis placeholder mapping (P0)

**Files:**

- Modify `D:\flutterserver\Multimodal_RAG_System\src\types\evaluation.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\services\evaluationApi.ts` (or the existing campaign API client)
- Add `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\evaluationDisplay.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\QuestionAnalysisTab.tsx`
- Add `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\evaluationDisplay.test.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.ui.test.tsx`

**Step 1 — write failing UI tests.** Test the real tab (do not mock it away) with an API payload where some metrics are absent. Required assertions:

```tsx
expect(screen.getByText("N/A")).toBeInTheDocument();
expect(screen.queryByText("+0.000")).not.toBeInTheDocument();
expect(screen.getByText("agentic")).toBeInTheDocument(); // returned best_quality_mode
expect(screen.queryByText("advanced")).not.toBeInTheDocument(); // not a sorted-label fallback
```

Also assert category, difficulty, modalities, and the backend comparability reason render from the payload.

**Step 2 — create one display policy.** In `evaluationDisplay.ts`, implement small pure helpers such as `formatOptionalNumber`, `formatOptionalPercent`, `formatOptionalTokens`, and `formatAvailability`. Their rule is `undefined | null => "N/A"`; zero is formatted as zero. Unit-test these helpers so each tab uses the same semantics.

**Step 3 — consume the new contract directly.** Replace `mapQuestionRows` in `EvaluationCenter.tsx`. Delete all hard-coded fields such as:

```ts
deltaCorrectness: 0,
deltaFaithfulness: 0,
deltaLatencyMs: 0,
ecrCorrectness: 0,
evidenceCoverage: 0,
unsupportedClaimRatio: 0,
bestMode: modes[0],
```

Render only the typed backend values. The heatmap must omit or visually mark unavailable deltas rather than colour them as zero. “Best Mode” becomes “Best Quality Mode” and has an `N/A` state.

**Step 4 — verify and commit.**

```powershell
npm test -- --run src/components/evaluation/evaluationDisplay.test.ts src/pages/EvaluationCenter.ui.test.tsx
npm run build
git add src/types/evaluation.ts src/services/evaluationApi.ts src/components/evaluation/evaluationDisplay.ts src/components/evaluation/QuestionAnalysisTab.tsx src/pages/EvaluationCenter.tsx src/components/evaluation/evaluationDisplay.test.ts src/pages/EvaluationCenter.ui.test.tsx
git commit -m "fix(evaluation): remove question analysis placeholders"
```

### Task 3: Make Run Trace select and display the actual run (P0)

**Files:**

- Modify the backend run-detail Pydantic schema and its existing route/service (locate `getRunDetail` implementation)
- Modify `D:\flutterserver\Multimodal_RAG_System\src\types\evaluation.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RunTraceTab.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RunTraceTree.tsx`
- Add backend test `D:\flutterserver\pdftopng\tests\test_evaluation_run_detail_projection.py`
- Add/modify frontend test `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RunTraceTab.test.tsx`

**Step 1 — write failing tests.** Cover selecting Q1/graph/repeat-2 after a Q9/naive initial selection. The test must prove the answer preview, trace rows, retrieval count, and token summary change to the selected run. Include an out-of-order response test: a slow prior request must not overwrite the newer selection.

**Step 2 — add a selected-run projection.** Extend the existing run-detail response with a typed summary tied to the returned `run_id`:

```python
class SelectedRunSummary(BaseModel):
    run_id: UUID
    question_id: str
    mode: CampaignMode
    repeat: int | None
    answer_preview: str | None
    latency_ms: float | None
    token_total: int | None
    accounting_status: AccountingStatus
```

Compute `token_total` from the same strict accounting source. Do not derive per-event tokens from lifecycle trace events; those events do not carry reliable token attribution.

**Step 3 — make selection real.** In `EvaluationCenter.tsx`, store `selectedRunId`, initialize it from the first available run, and refetch detail whenever it changes. Cancel/ignore stale requests and reset the selection when the campaign changes. Enable a single run selector; make question, mode, and repeat read-only metadata derived from the chosen run (or synchronized filters only if the existing API supports them).

**Step 4 — remove false event accounting.** In `RunTraceTree.tsx` remove the event-level Cost column entirely. Do not show `0` tokens for a lifecycle event with no token field. Show the selected-run token summary above the tree as `N/A` when accounting is partial/not available. Use the returned selected-run answer preview, never `results[0]`.

**Step 5 — verify and commit.**

```powershell
pytest tests/test_evaluation_run_detail_projection.py -q
npm test -- --run src/components/evaluation/RunTraceTab.test.tsx src/pages/EvaluationCenter.ui.test.tsx
git add evaluation tests/evaluation
git -C D:\flutterserver\Multimodal_RAG_System add src/types/evaluation.ts src/pages/EvaluationCenter.tsx src/components/evaluation/RunTraceTab.tsx src/components/evaluation/RunTraceTree.tsx src/components/evaluation/RunTraceTab.test.tsx src/pages/EvaluationCenter.ui.test.tsx
git commit -m "fix(evaluation): bind run trace to selected run"
```

> Commit the backend and frontend portions separately if they are separate repositories: `feat(evaluation): expose selected run projection` in `pdftopng`, then `fix(evaluation): select and render real run trace` in `Multimodal_RAG_System`.

### Task 4: Make Retrieval Evidence instrumentation-aware (P0)

**Files:**

- Modify the backend observability/result-detail projection used in Task 3
- Modify `D:\flutterserver\Multimodal_RAG_System\src\types\evaluation.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RetrievalEvidenceTab.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.tsx`
- Add backend test `D:\flutterserver\pdftopng\tests\test_evaluation_retrieval_evidence_projection.py`
- Add frontend test `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RetrievalEvidenceTab.test.tsx`

**Step 1 — write failing tests.** Seed a result-level instrumented run with null dense/BM25/rerank scores, absent expected-evidence match, and no coverage rows. Assert the API marks it `result_level` / `not_instrumented`, and the UI shows `N/A` or “Not instrumented,” never `0.00`, `no`, or an empty table that implies zero coverage.

**Step 2 — type the telemetry state.** Make retrieval-score fields and evidence flags nullable. Include `instrumentation_depth` and `evidence_coverage_status` in the selected-run projection. Keep `false` distinct from `null` for context use, answer use, and gold/evidence match.

**Step 3 — calculate coverage only when the sources exist.** Server-side, construct coverage rows from persisted expected evidence / atomic-fact snapshots and selected chunks or context packs. If the campaign did not persist the required expected evidence, return:

```json
{"status": "not_available", "rows": []}
```

not a synthetic zero-coverage result. Legacy result-level runs remain `not_instrumented` for retrieval scores.

**Step 4 — update UI semantics.** Remove `numberValue(score)` score fallbacks and `Boolean(missingFlag)` flag fallbacks. Render score cells with `formatOptionalNumber`; show an explicit coverage status message above the coverage table. Preserve measured `0.00` only when the API really sends `0`.

**Step 5 — verify and commit.**

```powershell
pytest tests/test_evaluation_retrieval_evidence_projection.py -q
npm test -- --run src/components/evaluation/RetrievalEvidenceTab.test.tsx
git commit -m "fix(evaluation): distinguish missing retrieval telemetry"
```

### Task 5: Add durable bulk Agent Behavior aggregation (P0/P1)

**Files:**

- Modify `D:\flutterserver\pdftopng\evaluation\campaign_schemas.py`
- Modify/add `D:\flutterserver\pdftopng\evaluation\agent_behavior_service.py`
- Modify the evaluation analytics router
- Modify repositories only as needed to add a bulk campaign trace lookup (do not issue one query per run)
- Add `D:\flutterserver\pdftopng\tests\test_evaluation_agent_behavior_analytics.py`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\types\evaluation.ts`
- Modify the API client and `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\AgentBehaviorTab.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.tsx`
- Add `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\AgentBehaviorTab.test.tsx`

**Step 1 — write failing backend tests.** Test three cases: a fully instrumented agentic trace; a run with no trace; and a non-agentic/not-applicable run. Assertions include real subtask/tool/visual/graph counts, null counts for missing instrumentation, status distinction, and durable RAGAS values without claim-ratio fallbacks.

**Step 2 — create a bulk response contract.** Each row must contain at least:

```python
class AgentBehaviorRow(BaseModel):
    run_id: UUID
    question_id: str
    mode: CampaignMode
    repeat: int | None
    instrumentation_status: Literal["recorded", "not_instrumented", "not_applicable"]
    subtask_count: int | None
    tool_call_count: int | None
    visual_call_count: int | None
    graph_call_count: int | None
    drilldown_depth: int | None
    answer_correctness: float | None
    faithfulness: float | None
    total_tokens: int | None
    accounting_status: AccountingStatus
```

Bulk-load agent traces/normalized observations, derive counts only from recorded trace structure, and join quality by run ID from durable RAGAS scores. Do not treat a missing `unsupported_claim_ratio` as correctness `1.00`, or a missing supported ratio as faithfulness `0.00`.

**Step 3 — provide a campaign endpoint.** Add an endpoint such as `GET /campaigns/{campaign_id}/agent-behavior`. It must return all selected campaign runs with a stable mode/question/repeat/run identity and no N+1 detail fetch pattern.

**Step 4 — replace frontend pseudo-aggregation.** Remove `mapAgentRows` fallbacks and the dependency on a single `runDetail`. Label quality columns “RAGAS Correctness” and “RAGAS Faithfulness.” Add Mode, Repeat, and Run ID (copyable/truncated) columns. Render null counts and unavailable quality as `N/A`; zero is shown only for an observed count of zero.

**Step 5 — verify and commit.**

```powershell
pytest tests/test_evaluation_agent_behavior_analytics.py -q
npm test -- --run src/components/evaluation/AgentBehaviorTab.test.tsx
git commit -m "feat(evaluation): add trustworthy agent behavior analytics"
```

### Task 6: Make Router Lab and overview warnings honest under token-only policy (P1)

**Files:**

- Modify `D:\flutterserver\pdftopng\evaluation\analytics.py`
- Modify relevant response schemas/tests
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RouterLabTab.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\CampaignOverviewTab.tsx`
- Add/modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RouterLabTab.test.tsx`
- Add/modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\CampaignOverviewTab.test.tsx`

**Step 1 — write failing tests.** With a retrospective campaign that has no actual router executions, assert Saved Tokens, Quality Loss, Quality Gain, and Router Regret display `N/A`, not zero. Assert no `$`/Cost column remains. With overview warning data, assert the partial-accounting warning and excluded-mode reason are visible.

**Step 2 — make router values nullable.** Backend router analysis must return optional metrics with a clear `analysis_type` and `availability_reason`. It may compute a retrospective metric only when the required observed values exist; it must never default a missing metric to zero.

**Step 3 — update token-only UI.** Remove monetary Cost fields from Router Lab. Rename resource comparisons to Tokens. In a retrospective-only campaign, show a concise notice: “No actual router runs were recorded; route outcome metrics are unavailable.” Preserve the selected retrospective mode/reason only as retrospective analysis.

**Step 4 — surface overview warnings.** Render `data.warnings` and per-mode exclusion/comparability reasons in Campaign Overview. Keep the existing fail-closed exclusion from Tokens vs Quality; explain it rather than trying to fill values.

**Step 5 — verify and commit.**

```powershell
pytest tests/evaluation -q
npm test -- --run src/components/evaluation/RouterLabTab.test.tsx src/components/evaluation/CampaignOverviewTab.test.tsx
git commit -m "fix(evaluation): make router and accounting warnings explicit"
```

### Task 7: Add end-to-end anti-placeholder regression coverage (P2)

**Files:**

- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.ui.test.tsx`
- Add `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.integration.test.tsx`
- Modify/add focused backend contract tests under `D:\flutterserver\pdftopng\tests\evaluation\`
- Modify `D:\flutterserver\pdftopng\docs\evaluation\README.md` or the existing Evaluation Center documentation

**Step 1 — stop blanket tab mocking.** Keep narrow service mocks, but render the actual Question Analysis, Run Trace, Retrieval Evidence, Agent Behavior, and Router Lab components in integration tests.

**Step 2 — add regression fixtures.** The fixture campaign needs:

- a valid mode delta and an unavailable delta;
- two selectable runs with distinct answer/trace/retrieval details;
- null retrieval scores and one actual zero score;
- recorded and absent agent traces;
- no actual router runs;
- complete and partial token-accounting modes.

**Step 3 — assert the invariants.** At a minimum test:

```tsx
// No unknown metric is rendered as 0/0.00/0.0%/1.00.
// Changing the selected run changes every run-specific panel.
// A null retrieval score renders N/A while numeric zero renders 0.00.
// Retrospective router metrics are N/A without actual data.
// Partial accounting excludes the affected comparison from token-derived metrics.
```

**Step 4 — document the contract.** Document the distinction among `complete`, `partial`, `not_available`, `not_instrumented`, and `not_applicable`, including the rule that legacy data is not guessed. Include a small route-to-panel data-source table.

**Step 5 — full verification and commit.**

```powershell
pytest tests/evaluation -q
npm test -- --run
npm run build
git status --short
git commit -m "test(evaluation): prevent placeholder analytics regressions"
```

## Implementation order and acceptance gate

1. Tasks 1–5 are the P0 release gate. Do not describe Question Analysis, Run Trace, Retrieval Evidence, or Agent Behavior as formal evaluation output until all five are deployed and the test fixtures pass.
2. Task 6 follows immediately as P1 because it removes remaining misleading zeroes and makes partial accounting understandable.
3. Task 7 is the regression gate before declaring the Evaluation Center trustworthy for future changes.
4. At implementation time, after the primary agent completes the code and full test suite, use one subagent only for read-only review if requested. Address validated review findings in a follow-up primary-agent commit, then rerun focused and full verification.

## Manual acceptance checklist after deployment

- Run a fresh 4-mode campaign and open Q1 through Q16 in at least two modes/repeats.
- Confirm Question Analysis differs by question where measurements differ; no blanket `0.000` heatmap and no sorted-mode “best mode.”
- Confirm changing the Run Trace selector updates its answer, trace, retrieval evidence, and token accounting together.
- Confirm result-level retrieval scores say `N/A` / `not instrumented`, while genuinely measured zero scores still show `0.00`.
- Confirm Agent Behavior lists mode/repeat/run ID and does not turn missing claims into 1.00/0.00 quality values.
- Confirm Router Lab says `N/A` for absent actual router metrics and contains no monetary fallback.
- Confirm partial Agentic/Graph accounting remains excluded from token-derived comparisons with a visible reason.
