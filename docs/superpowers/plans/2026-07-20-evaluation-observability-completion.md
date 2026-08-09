# Evaluation Center Observability Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Evaluation Center's token accounting, selected-run summaries, GraphRAG observability, router row identity, and trace lifecycle presentation without inventing unavailable measurements.

**Architecture:** Keep durable execution/RAGAS/accounting records as the source of truth. Extend the selected-run observability projection with explicit retrieval, claim, graph, and accounting diagnostics, then let the frontend render one shared selected-run context across Run Trace, Retrieval Evidence, and Claim Evidence. Preserve raw lifecycle events and retrospective router records in the backend/export while presenting a concise, identity-rich view in the UI.

**Tech Stack:** FastAPI, Pydantic, SQLite repositories, pytest; React, TypeScript, Chakra UI, Vitest, ESLint.

## Global Constraints

- This Evaluation Center is token-only: do not add monetary columns, `$0.000` fallbacks, pricing comparability, or cost-based router metrics.
- A token total is valid only when every observed call in the official execution scope has measured, balanced usage; otherwise return `total_tokens: null` and an explicit partial reason.
- Never estimate missing provider usage from text length or context size in the authoritative accounting projection.
- A phase is complete only when every measured event has a non-`unclassified` phase; keep `phase_attribution_status: partial` otherwise.
- `GRAPH_RESULT_LEVEL` identifies the selected campaign mode only; GraphRAG execution claims require persisted graph event/evidence data or an explicit fallback record.
- Preserve raw trace lifecycle rows for exports and audit; UI consolidation must be presentation-only.
- Retrospective routing is not an actual router execution. When no actual router run exists, selected-mode savings/regret values remain `N/A`.
- Old campaigns are not backfilled by inference. They remain `N/A`, `not_instrumented`, or `partial` until a new run records the required telemetry.
- Backend and frontend are separate repositories. Each task ends with a focused commit in the repository it changes; never stage the other repository's files.
- Tests are written before implementation and each task must have an independent verification command.

---

### Task 1: Diagnose and close Graph/Agentic accounting gaps (P0)

**Files:**

- Modify `D:\flutterserver\pdftopng\evaluation\accounting_schemas.py`
- Modify `D:\flutterserver\pdftopng\evaluation\research_analytics.py`
- Modify `D:\flutterserver\pdftopng\evaluation\token_normalizers.py`
- Modify `D:\flutterserver\pdftopng\core\llm_usage_callback.py`
- Modify phase wrappers in `D:\flutterserver\pdftopng\graph_rag\generic_mode.py`, `D:\flutterserver\pdftopng\graph_rag\global_search.py`, `D:\flutterserver\pdftopng\graph_rag\local_search.py`, and `D:\flutterserver\pdftopng\data_base\RAG_QA_service.py` only when the diagnostic identifies an unclassified call
- Add/modify `D:\flutterserver\pdftopng\tests\test_llm_usage_callback.py`
- Add/modify `D:\flutterserver\pdftopng\tests\test_evaluation_research_analytics.py`
- Add/modify `D:\flutterserver\pdftopng\tests\test_evaluation_token_normalizers.py`

**Interfaces:**

- Consumes: durable `AccountingScope`, `UsageEvent`, `TokenBreakdown`, provider callback payloads.
- Produces: `TokenBreakdown` diagnostics with observed/measured/missing/unbalanced/unclassified counts and stable warning reason codes.

- [ ] **Step 1: Add failing accounting diagnostics tests.**

  Seed one complete scope, one scope with a missing usage event, and one scope with an unclassified phase. Assert the resulting breakdown contains the exact counts and that `total_tokens is None` for partial accounting. Add callback fixtures for `usage_metadata`, nested `usage`, nested `token_usage`, and an empty payload; empty usage must remain `missing`.

  Run:

  ```powershell
  cd D:\flutterserver\pdftopng
  .venv\Scripts\python.exe -m pytest tests/test_llm_usage_callback.py tests/test_evaluation_token_normalizers.py tests/test_evaluation_research_analytics.py -q
  ```

  Expected: the new assertions fail because the breakdown does not yet expose the diagnostics and at least one provider shape is currently treated as missing.

- [ ] **Step 2: Expose non-authoritative diagnostics.**

  Extend `TokenBreakdown` with nullable-safe counters:

  ```python
  observed_call_count: int = 0
  measured_call_count: int = 0
  missing_usage_call_count: int = 0
  unbalanced_call_count: int = 0
  unclassified_phase_call_count: int = 0
  ```

  Update `_tokens()` to populate them from scopes/events. Keep the existing rule that `total_tokens` is `None` whenever accounting is not complete. Add warning codes such as `missing_usage`, `unbalanced_usage`, and `unclassified_phase` to the research summary, including provider/purpose/phase counts in warning metadata without exposing raw prompts or secrets.

- [ ] **Step 3: Normalize only measured provider shapes.**

  Update `extract_usage_dict()` and `EvaluationUsageCallback._extract_usage()` to inspect the already-supported terminal response locations plus the explicit nested aliases covered by the failing fixtures. Do not infer tokens from text or latency. Add a regression assertion that a provider response with no usage still produces `usage_status="missing"`, while a response with a complete nested total becomes `usage_status="measured"` and `reconciliation_status="balanced"`.

- [ ] **Step 4: Close phase boundaries identified by the diagnostics.**

  Wrap every missing graph/agentic LLM call in the phase that describes the call (`graph_reasoning`, `query_expansion`, `answer_generation`, `agent_planning`, `agent_synthesis`, or `visual_verification`). Add one integration fixture that executes a graph and agentic unit under an accounting scope and asserts no measured event is `unclassified`. If a call is genuinely outside the execution scope, leave it unaccounted and emit the explicit warning instead of attaching it to the wrong run.

- [ ] **Step 5: Verify and commit backend accounting.**

  ```powershell
  cd D:\flutterserver\pdftopng
  .venv\Scripts\python.exe -m pytest tests/test_llm_usage_callback.py tests/test_evaluation_token_normalizers.py tests/test_evaluation_accounting_runtime.py tests/test_evaluation_research_analytics.py -q
  git add evaluation/accounting_schemas.py evaluation/research_analytics.py evaluation/token_normalizers.py core/llm_usage_callback.py graph_rag data_base tests
  git commit -m "fix(evaluation): complete execution token diagnostics"
  ```

  Acceptance: a fresh Graph/Agentic smoke campaign has either complete measured token totals or a warning that identifies the exact missing provider/purpose; no total is fabricated.

### Task 2: Add a selected-run observability projection with GraphRAG details (P0)

**Files:**

- Modify `D:\flutterserver\pdftopng\evaluation\trace_schemas.py`
- Modify `D:\flutterserver\pdftopng\evaluation\router.py`
- Modify `D:\flutterserver\pdftopng\evaluation\observability_storage.py` only if a bulk run lookup is needed
- Add/modify `D:\flutterserver\pdftopng\tests\test_evaluation_run_detail_projection.py`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\types\evaluation.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.mappers.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.tsx`
- Add `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RunContextSelector.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RunTraceTab.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RetrievalEvidenceTab.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\ClaimEvidenceTab.tsx`
- Add/modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.integration.test.tsx`

**Interfaces:**

- Consumes: one `run_id`, existing normalized retrieval/chunk/claim rows, graph repositories, and `get_run_token_breakdown()`.
- Produces: `EvaluationRunObservabilityDetail` with `graph_events`, `graph_evidence_items`, `accounting_diagnostics`, and a summary tied to the returned `run_id`.

- [ ] **Step 1: Write failing backend projection tests.**

  Seed a Graph run with one `EvaluationGraphEvent`, two `EvaluationGraphEvidenceItem` rows, retrieval chunks, no claims, and partial accounting. Assert the response contains the graph route/reason, event/evidence counts, `evidence_coverage_status="not_instrumented"`, and the selected run's `total_tokens: null`. Seed a run without graph rows and assert the response says graph observability is `not_instrumented` rather than claiming graph success.

- [ ] **Step 2: Extend the typed run-detail contract.**

  Import the existing `EvaluationGraphEvent` and `EvaluationGraphEvidenceItem` models into `EvaluationRunObservabilityDetail` and add:

  ```python
  graph_events: list[EvaluationGraphEvent] = Field(default_factory=list)
  graph_evidence_items: list[EvaluationGraphEvidenceItem] = Field(default_factory=list)
  graph_observability_status: Literal["recorded", "not_instrumented", "fallback"] = "not_instrumented"
  accounting_diagnostics: TokenBreakdown
  ```

  The route must load graph rows for the selected run, derive `graph_observability_status` from persisted rows and the run's retrieval payload, and never infer a graph event from `mode == "graph"` alone.

- [ ] **Step 3: Build meaningful selected-run summaries.**

  In `EvaluationCenter.mappers.ts`, add pure mappers that produce:

  ```ts
  retrievalSummary: string;
  claimsSummary: string;
  graphSummary: string;
  ```

  Use actual counts. For example, a run with retrieval events and no graph rows renders `"1 query · 8 chunks · result-level only"`; a recorded Graph event renders its route, node/edge/path counts, and evidence-item count; an empty claim array renders `"No claim extraction recorded for this run"` rather than implying a failed extraction. Keep the summary scoped to the selected `run_id`.

- [ ] **Step 4: Add one shared run context selector.**

  Move the existing Run Trace selector into `RunContextSelector.tsx` and mount it above Run Trace, Retrieval Evidence, and Claim Evidence. It must receive `runOptions`, `selectedRunId`, and `onSelectedRunIdChange`; Question/Mode/Repeat remain read-only labels derived from the same selected option. Keep one state/request path in `EvaluationCenter.tsx` so switching tabs cannot create divergent selected runs.

- [ ] **Step 5: Render GraphRAG evidence explicitly.**

  Add a compact “Graph observability” panel to `RetrievalEvidenceTab` showing status, route, router reason, node/edge/path counts, graph-to-chunk success rate, and graph evidence-item count. Render `N/A` with the status reason for `not_instrumented` and a visible fallback reason for `fallback`. Do not turn a missing graph event into an empty successful graph summary.

- [ ] **Step 6: Verify and commit both repositories separately.**

  ```powershell
  cd D:\flutterserver\pdftopng
  .venv\Scripts\python.exe -m pytest tests/test_evaluation_run_detail_projection.py -q
  git add evaluation/trace_schemas.py evaluation/router.py tests/test_evaluation_run_detail_projection.py
  git commit -m "feat(evaluation): expose selected run graph observability"

  cd D:\flutterserver\Multimodal_RAG_System
  npm test -- --run src/pages/EvaluationCenter.integration.test.tsx src/components/evaluation/RetrievalEvidenceTab.test.tsx
  npm run lint:ci
  git add src/types/evaluation.ts src/pages/EvaluationCenter.mappers.ts src/pages/EvaluationCenter.tsx src/components/evaluation/RunContextSelector.tsx src/components/evaluation/RunTraceTab.tsx src/components/evaluation/RetrievalEvidenceTab.tsx src/components/evaluation/ClaimEvidenceTab.tsx src/pages/EvaluationCenter.integration.test.tsx
  git commit -m "feat(evaluation): show selected run graph evidence"
  ```

### Task 3: Make Run Trace lifecycle presentation unambiguous (P1)

**Files:**

- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.mappers.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RunTraceTree.tsx`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RunTraceTab.tsx`
- Add/modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RunTraceTree.test.tsx`

**Interfaces:**

- Consumes: raw `trace_events` containing the same `span_id`/`stage_name` with `running` and terminal rows.
- Produces: a presentation list with one terminal lifecycle row by default and a disclosure that the raw start event is retained.

- [ ] **Step 1: Write the failing lifecycle test.**

  Pass a `running` sequence-1 event and a `success` sequence-2 event for the same span/stage. Assert the default table has one `campaign_unit_execution` row, shows the terminal `success` and duration, and exposes a “show lifecycle events” control. Assert a failed terminal event remains visible as `failed`.

- [ ] **Step 2: Implement deterministic lifecycle grouping.**

  Group only rows with the same `span_id` and `stage_name`; choose the terminal row (`success`, `failed`, `partial`, `timeout`, or `skipped`) as the displayed row. Leave unrelated events untouched. Keep raw payload/error disclosure available through the expanded lifecycle control.

- [ ] **Step 3: Verify and commit.**

  ```powershell
  cd D:\flutterserver\Multimodal_RAG_System
  npm test -- --run src/components/evaluation/RunTraceTree.test.tsx src/pages/EvaluationCenter.integration.test.tsx
  npm run lint:ci
  git add src/pages/EvaluationCenter.mappers.ts src/components/evaluation/RunTraceTree.tsx src/components/evaluation/RunTraceTab.tsx src/components/evaluation/RunTraceTree.test.tsx
  git commit -m "fix(evaluation): clarify run trace lifecycle"
  ```

### Task 4: Make Router Lab retrospective rows identity-safe (P1)

**Files:**

- Modify `D:\flutterserver\pdftopng\evaluation\campaign_schemas.py`
- Modify `D:\flutterserver\pdftopng\evaluation\analytics.py`
- Add/modify `D:\flutterserver\pdftopng\tests\test_evaluation_router_analysis.py`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\types\evaluation.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.mappers.ts`
- Modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RouterLabTab.tsx`
- Add/modify `D:\flutterserver\Multimodal_RAG_System\src\components\evaluation\RouterLabTab.test.tsx`

**Interfaces:**

- Consumes: persisted routing decisions joined to campaign results by `run_id`.
- Produces: typed retrospective rows with `run_id`, `question_id`, `mode`, `repeat`, `analysis_type`, and nullable quality/latency/token/regret values.

- [ ] **Step 1: Write failing identity tests.**

  Seed two retrospective agentic decisions for different questions. Assert the backend rows contain distinct question/run identities, both are `analysis_type="retrospective"`, and no saved-token/regret metric is synthesized. Assert an actual-router row is the only case allowed to populate selected mode and router utility metrics.

- [ ] **Step 2: Add typed router rows and join question identity.**

  Add `RouterAnalysisRow` with `run_id`, `question_id`, `mode`, `repeat_number`, `selected_mode`, `analysis_type`, and nullable measurements. In `_routing_decisions_for_context()`, enrich each persisted decision using its owning campaign result instead of changing the database schema. Keep retrospective decisions visible but label them as per-question retrospective observations.

- [ ] **Step 3: Update the UI labels.**

  Add `Question`, `Run ID`, and `Repeat` columns. For retrospective data, label the decision card “Retrospective best-mode observation” instead of “Selected Mode”; keep `Saved Tokens`, quality deltas, and regret as `N/A` unless `hasActualRouterRuns` is true. Use a stable row key of `runId + analysisType`, not `row.label`.

- [ ] **Step 4: Verify and commit both repositories separately.**

  ```powershell
  cd D:\flutterserver\pdftopng
  .venv\Scripts\python.exe -m pytest tests/test_evaluation_router_analysis.py -q
  git add evaluation/campaign_schemas.py evaluation/analytics.py tests/test_evaluation_router_analysis.py
  git commit -m "fix(evaluation): identify retrospective router rows"

  cd D:\flutterserver\Multimodal_RAG_System
  npm test -- --run src/components/evaluation/RouterLabTab.test.tsx
  npm run lint:ci
  git add src/types/evaluation.ts src/pages/EvaluationCenter.mappers.ts src/components/evaluation/RouterLabTab.tsx src/components/evaluation/RouterLabTab.test.tsx
  git commit -m "fix(evaluation): label retrospective router observations"
  ```

### Task 5: End-to-end acceptance and documentation (P0/P1)

**Files:**

- Modify `D:\flutterserver\pdftopng\docs\design-docs\evaluation-center.md`
- Modify `D:\flutterserver\pdftopng\docs\BACKEND.md` if the selected-run/GraphRAG observability contract changes
- Modify `D:\flutterserver\Multimodal_RAG_System\src\pages\EvaluationCenter.integration.test.tsx`
- Add `D:\flutterserver\pdftopng\tests\test_evaluation_observability_contract.py` if cross-endpoint invariants are not covered by the task tests

**Interfaces:**

- Consumes: the completed backend and frontend projections from Tasks 1–4.
- Produces: a repeatable smoke/full evaluation checklist and a documented contract for complete, partial, not-available, not-instrumented, recorded, and fallback states.

- [ ] **Step 1: Add cross-contract tests.**

  Assert that a selected Graph run has consistent `run_id`, question, mode, repeat, answer preview, retrieval chunks, graph status, and token diagnostics across Run Trace, Retrieval Evidence, and Claim Evidence. Assert that an old/no-telemetry run stays `N/A` rather than receiving inferred totals. Assert that a retrospective router campaign never renders actual-router savings/regret.

- [ ] **Step 2: Run the backend regression set.**

  ```powershell
  cd D:\flutterserver\pdftopng
  .venv\Scripts\python.exe -m pytest tests/test_evaluation_analytics_api.py tests/test_evaluation_question_comparison.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_schema.py tests/test_evaluation_observability_repository.py tests/test_evaluation_run_detail_projection.py tests/test_evaluation_router_analysis.py -q --tb=short
  ```

  Expected: all tests pass; warnings may report deliberately partial legacy fixtures, but no test may accept a fabricated token/quality value.

- [ ] **Step 3: Run the frontend CI set.**

  ```powershell
  cd D:\flutterserver\Multimodal_RAG_System
  npm run lint:ci
  npm test -- --run
  npx tsc -b --pretty false
  npm run build
  ```

  Expected: lint has zero warnings/errors; all Vitest files pass; typecheck/build pass. Existing bundle-size warnings are non-blocking and must remain explicitly reported.

- [ ] **Step 4: Run a fresh smoke campaign before the full matrix.**

  Use one Graph and one Agentic question with one repeat. Confirm:

  - RAGAS quality is complete with no missing/failed samples.
  - Graph mode reports `recorded` graph observability when graph events exist, or an explicit fallback/not-instrumented status.
  - Agentic and Graph accounting either become complete or expose exact missing-usage diagnostics; no total is estimated.
  - Run Trace, Retrieval Evidence, and Claim Evidence all identify the same selected run.
  - Router Lab remains retrospective/N/A unless actual routing was enabled.

- [ ] **Step 5: Run the full campaign only after smoke acceptance.**

  Repeat the existing 64-run matrix and save the campaign ID alongside the commit hashes. Record the final quality/accounting/phase statuses in the evaluation-center documentation; do not treat old campaigns as post-fix evidence.

- [ ] **Step 6: Commit documentation and report residual limitations.**

  ```powershell
  cd D:\flutterserver\pdftopng
  git add docs/design-docs/evaluation-center.md docs/BACKEND.md tests/test_evaluation_observability_contract.py
  git commit -m "docs(evaluation): document observability acceptance"
  ```

## Self-review checklist

- Token accounting is fixed at the source where possible, but missing provider usage is still fail-closed and diagnostically visible.
- Run Trace summaries are derived from the selected run, not global `results[0]`.
- Graph mode is not treated as proof of GraphRAG execution; graph rows are loaded from the graph observability repositories.
- Router retrospective rows are per-run/per-question and cannot be mistaken for actual policy decisions.
- Raw lifecycle events remain available for audit even when the default UI collapses them.
- No task adds monetary output or turns unavailable metrics into zero.
