# V9 Agent Behavior Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Evaluation Center’s Agent Behavior page report the real v9 evidence-first execution state, execute and persist required graph/visual stages, and fail closed whenever a required stage is not satisfied.

**Architecture:** Keep the legacy v8 step projection intact, but add a versioned v9 projection built only from durable materializations, evidence packets, slot resolutions, accounting events, and graph/visual trace events. A required graph locator or visual extractor must create a success/failure event; a run without the required successful stage becomes `qualified_partial` (or configuration-incompatible before execution) and cannot pass release metrics.

**Tech Stack:** Python 3.11, FastAPI/Pydantic, SQLite/aioSQLite, pytest; React, TypeScript, Chakra UI, Vitest, Testing Library.

## Global Constraints

- `required_locator` and `visual_required` are strict: a missing, skipped, empty, or failed required stage must never yield a `complete` v9 response.
- If the required provider/capability is unavailable before execution (including an Evaluation Setup budget that cannot admit the phase), return `configuration_incompatible`; if the stage was admitted but yields no eligible evidence or fails at runtime, return `qualified_partial` with the observed failure reason.
- Unknown, not applicable, uninstrumented, and measured-zero values remain distinct in API and UI.
- Graph mode/configuration is never evidence of graph traversal; only a persisted graph event is.
- Do not synthesize v8 `steps`, subtask counts, or tool-call counts for v9 runs.
- Preserve v8 and non-agentic historical campaign rendering.
- Keep token-only evaluation UI; do not reintroduce cost fallback.
- Use batch repository reads for campaign-level behavior data; no per-run database N+1 loop.

---

## Response contract

`GET /evaluation/campaigns/{campaign_id}/research/agent-behavior` remains the endpoint, but returns a versioned response:

```python
class AgentBehaviorResponse(BaseModel):
    campaign_id: str
    behavior_schema_version: Literal["2"]
    rows: list[AgentBehaviorRow]

class AgentBehaviorRow(BaseModel):
    run_id: str
    campaign_id: str
    question_id: str
    mode: CampaignMode
    repeat_number: int
    behavior_schema: Literal["v8", "v9", "not_applicable"]
    trace_status: Literal[
        "completed", "partial", "failed", "not_applicable", "not_instrumented"
    ]
    failure_reason: str | None
    accounting_status: Literal["complete", "partial", "not_available"]
    total_tokens: int | None
    legacy: LegacyAgentBehaviorMetrics | None
    v9: V9AgentBehaviorMetrics | None
```

`V9AgentBehaviorMetrics` contains route/contract properties, durable evidence and slot counts, repair and generation metrics, token reservations, and explicit graph/visual execution states. `legacy` contains the current v8 subtasks/tool/visual/graph/drilldown counts. Exactly one is populated for a behavior-capable run.

The graph and visual execution states are:

```python
ExecutionState = Literal[
    "not_requested", "not_triggered", "executed", "failed",
    "required_but_not_satisfied", "not_instrumented",
]
```

`required_but_not_satisfied` is an execution outcome, not a zero call count.

## Task 1: Versioned behavior schema and batch v9 projection

**Files:**
- Modify: `pdftopng/evaluation/campaign_schemas.py`
- Modify: `pdftopng/evaluation/research_analytics.py`
- Modify: `pdftopng/evaluation/observability_storage.py`
- Modify: `pdftopng/tests/test_campaign_schemas.py`
- Modify: `pdftopng/tests/test_evaluation_research_analytics.py`
- Modify: `pdftopng/tests/test_evaluation_observability_repository.py`

**Consumes:** v8 `AgentTraceSummary`; `evaluation_v9_attempt_materializations`, `evaluation_evidence_packets`, `evaluation_slot_resolutions`, `evaluation_trace_events`, `evaluation_graph_events`, and accounting scopes/events.

**Produces:** `AgentBehaviorResponse.behavior_schema_version == "2"`; one v9 behavior row per v9 result, without relying on legacy `steps`.

- [ ] **Step 1: Write failing contract tests.**

  Add fixtures for: a completed v9 run with 13 persisted packets and two slots; a v8 trace; a naive run; and a failed `agentic-v9` run with a blank historical error. Assert that:

  ```python
  assert response.behavior_schema_version == "2"
  assert v9.behavior_schema == "v9"
  assert v9.v9.evidence_packet_count == 13
  assert v9.v9.slot_resolution_count == 2
  assert v9.legacy is None
  assert failed.trace_status == "failed"
  assert failed.failure_reason == "failure_reason_not_recorded"
  assert naive.behavior_schema == "not_applicable"
  ```

- [ ] **Step 2: Run targeted tests and verify the current implementation fails.**

  Run: `pytest tests/test_campaign_schemas.py tests/test_evaluation_research_analytics.py -q`

  Expected: failures because the response has no schema version or v9 projection.

- [ ] **Step 3: Define the discriminated schema.**

  Add `LegacyAgentBehaviorMetrics` and `V9AgentBehaviorMetrics` to `campaign_schemas.py`. Keep the existing flat fields only as a temporary deserialization-compatible alias if a deployed frontend still reads them; do not populate them for v9.

  V9 metric fields must include:

  ```python
  route: str | None
  graph_policy: str | None
  visual_required: bool | None
  evidence_extraction_required: bool | None
  retrieval_query_count: int | None
  provider_attempt_count: int | None
  final_generation_count: int | None
  evidence_packet_count: int | None
  packed_evidence_count: int | None
  slot_resolution_count: int | None
  required_slot_count: int | None
  supported_slot_count: int | None
  repair_count: int | None
  final_claim_count: int | None
  reserved_tokens: int | None
  reconciled_tokens: int | None
  graph_execution: ExecutionState
  visual_execution: ExecutionState
  ```

- [ ] **Step 4: Add batch repository projections.**

  Add campaign-scoped repository methods returning maps keyed by `run_id`/`attempt_id`, for materializations, evidence-packet counts, slot-resolution counts, graph events, and stage events. The SQL must group by run/attempt and return empty maps for absent rows.

  `ResearchAnalyticsService.get_agent_behavior()` loads these maps once, then constructs rows in memory. It determines v9 from materialized `agentic_execution_version == "v9"` or the persisted v9 attempt; it must not use `result.mode == "agentic"` as the only discriminator.

- [ ] **Step 5: Implement strict null/status projection.**

  - Completed v9 with a materialization: `behavior_schema="v9"`, trace status from the trace/materialization.
  - Failed v9: `trace_status="failed"`, all runtime metrics `None`, and a safe recorded failure reason; blank historical error becomes `failure_reason_not_recorded`.
  - v8: `behavior_schema="v8"`, populate only `legacy`.
  - Naive/advanced/graph non-agentic: `behavior_schema="not_applicable"` and no legacy/v9 metrics.
  - A missing v9 materialization for a completed v9 run: `trace_status="not_instrumented"`, v9 metrics `None`; never zero-fill.

- [ ] **Step 6: Run tests and commit.**

  Run: `pytest tests/test_campaign_schemas.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_repository.py -q`

  Expected: PASS.

  Commit: `feat(evaluation): project versioned v9 agent behavior`

## Task 2: Required graph locator execution and durable telemetry

**Files:**
- Modify: `pdftopng/evaluation/agentic_v9_campaign_runtime.py`
- Modify: `pdftopng/data_base/rag_graph_locator.py`
- Modify: `pdftopng/evaluation/execution_worker.py`
- Modify: `pdftopng/evaluation/observability_storage.py`
- Modify: `pdftopng/tests/test_agentic_v9_campaign_runtime.py`
- Modify: `pdftopng/tests/test_evaluation_execution_worker.py`
- Modify: `pdftopng/tests/test_evaluation_observability_repository.py`

**Consumes:** `QueryContract.graph_policy`, existing `locate_graph_sources`, source-bound graph locator results, and `EvaluationGraphEvent`/`EvaluationGraphEvidenceItem` persistence models.

**Produces:** A required graph locator call is observable and source-bound; failed/empty required graph work cannot produce a complete v9 answer.

- [ ] **Step 1: Write failing runtime tests.**

  Use injected fake graph locator outcomes and assert:

  ```python
  required_success = await runtime.answer(... graph_policy="required_locator")
  assert required_success.agent_trace["agentic_v9"]["graph_execution"]["state"] == "executed"

  required_empty = await runtime.answer(... graph_policy="required_locator")
  assert required_empty.agent_trace["response_status"] == "qualified_partial"
  assert required_empty.agent_trace["agentic_v9"]["graph_execution"]["state"] == "required_but_not_satisfied"
  ```

  Also test `never -> not_requested` and optional locator not reached -> `not_triggered`.

- [ ] **Step 2: Run tests and verify current runtime fails.**

  Run: `pytest tests/test_agentic_v9_campaign_runtime.py -q`

  Expected: failures because the runtime currently creates no graph execution payload/event.

- [ ] **Step 3: Add an explicit graph-locator adapter at the v9 boundary.**

  `agentic_v9_campaign_runtime.py` receives an injectable graph locator protocol. Its production adapter calls `data_base.rag_graph_locator.locate_graph_sources` and returns only source-authorized graph-to-chunk evidence/hints. It must not import private GraphRAG implementation details into the execution core.

- [ ] **Step 4: Enforce policy A.**

  - For `required_locator`, execute the locator before final evidence packing.
  - If graph is unavailable, errors, returns no eligible source-bound evidence, or cannot be persisted, set `graph_execution.state="required_but_not_satisfied"`, include a redacted reason, and return `qualified_partial`.
  - Do not silently substitute vector retrieval and emit `complete`.
  - For non-required policies, record only actual execution; absence remains `not_triggered`, not a synthetic fallback success.

- [ ] **Step 5: Persist graph events and evidence items.**

  Extend the v9 trace payload with `graph_execution` data. In `execution_worker.py`, materialize a graph event plus evidence items through `EvaluationObservabilityRepository`. Include graph query, route/policy, latency, result status, fallback reason, and only source-bound node/edge/chunk references.

- [ ] **Step 6: Run tests and commit.**

  Run: `pytest tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_execution_worker.py tests/test_evaluation_observability_repository.py -q`

  Expected: PASS.

  Commit: `feat(agentic-v9): execute and trace required graph locator`

## Task 3: Required visual extraction execution and durable telemetry

**Files:**
- Modify: `pdftopng/evaluation/agentic_v9_campaign_runtime.py`
- Modify: `pdftopng/data_base/agentic_v9/visual_evidence_extractor.py`
- Modify: `pdftopng/evaluation/execution_worker.py`
- Modify: `pdftopng/evaluation/observability_storage.py`
- Modify: `pdftopng/tests/test_agentic_v9_campaign_runtime.py`
- Modify: `pdftopng/tests/test_agentic_v9_visual_evidence_extractor.py`
- Modify: `pdftopng/tests/test_evaluation_execution_worker.py`

**Consumes:** `QueryContract.visual_required`, the existing `AssetLocator` and `VisualEvidenceExtractor`, and phase-policy/budget controller guarantees.

**Produces:** visual-required work has an observed success/failed state and source-bound evidence packets; visual omission cannot produce complete v9 status.

- [ ] **Step 1: Write failing visual-required tests.**

  Cover a successful located asset, no eligible asset, provider error, and disabled provider capability. Assert that success yields `executed`, while the latter three yield `qualified_partial` plus `required_but_not_satisfied` or an admission-time configuration error.

- [ ] **Step 2: Run failing tests.**

  Run: `pytest tests/test_agentic_v9_visual_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py -q`

- [ ] **Step 3: Wire the existing extractor through the runtime.**

  The production adapter must invoke `AssetLocator` first, pass only selected authorized assets to `VisualEvidenceExtractor`, and use the budgeted visual phase (`visual_extract`). It returns evidence packets that preserve `asset_id`, document identity, and page/bounding-box locator.

- [ ] **Step 4: Apply policy A and materialize stage events.**

  A visual-required run without a successful extractable source-bound visual result becomes `qualified_partial`. Persist a `visual_extract` trace event with `executed`, `failed`, or `required_but_not_satisfied`; retain a redacted reason and selected/dropped asset counts.

- [ ] **Step 5: Run tests and commit.**

  Run: `pytest tests/test_agentic_v9_visual_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_execution_worker.py -q`

  Expected: PASS.

  Commit: `feat(agentic-v9): enforce and trace required visual extraction`

## Task 4: Version-aware Agent Behavior UI

**Files:**
- Modify: `Multimodal_RAG_System/src/types/evaluation.ts`
- Modify: `Multimodal_RAG_System/src/pages/EvaluationCenter.mappers.ts`
- Modify: `Multimodal_RAG_System/src/components/evaluation/AgentBehaviorTab.tsx`
- Modify: `Multimodal_RAG_System/src/components/evaluation/AgentBehaviorTab.test.tsx`
- Modify: `Multimodal_RAG_System/src/pages/EvaluationCenter.mappers.test.ts`
- Modify: `Multimodal_RAG_System/src/pages/EvaluationCenter.integration.test.tsx`
- Modify: `Multimodal_RAG_System/src/types/evaluation.contract.test.ts`

**Consumes:** Agent Behavior schema v2.

**Produces:** A UI that distinguishes v8 from v9, actual execution from requirement, and N/A from zero.

- [ ] **Step 1: Write failing type/mapper tests.**

  Add a v9 API fixture with a required graph stage lacking an event and assert the mapped row contains `required_but_not_satisfied`, `evidencePacketCount > 0`, and `legacy === null`. Add an old-schema fixture and assert a backend redeploy warning is selected.

- [ ] **Step 2: Update TypeScript API types and mapper.**

  Define TypeScript equivalents of `LegacyAgentBehaviorMetrics`, `V9AgentBehaviorMetrics`, and schema version 2. Do not use numeric default values in mapper functions; missing values remain `null`.

- [ ] **Step 3: Render separate v9 and v8 panels.**

  - V9 summary cards: completed v9 runs, evidence packets, supported/required slots, final generations, and required-stage warnings.
  - V9 table: route, graph policy/state, visual requirement/state, retrieval queries, evidence/packed evidence, slots, repairs, provider attempts, final generation, response status, tokens.
  - Legacy table: existing subtasks/tool/visual/graph/drilldown metrics.
  - Non-agentic table rows: status and accounting only, with no behavior claims.
  - Failed v9 rows: failure badge and safe reason.

- [ ] **Step 4: Add old-backend guard.**

  If `behavior_schema_version !== "2"`, show an error alert saying the backend must be redeployed and do not present legacy zero values as v9 observations.

- [ ] **Step 5: Run frontend checks and commit.**

  Run: `npm run lint && npm test -- --run src/components/evaluation/AgentBehaviorTab.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/pages/EvaluationCenter.integration.test.tsx src/types/evaluation.contract.test.ts`

  Expected: PASS.

  Commit: `feat(evaluation-ui): render v9 behavior observability`

## Task 5: Cross-layer smoke, migration behaviour, and deployment verification

**Files:**
- Modify: `pdftopng/tests/test_evaluation_analytics_api.py`
- Modify: `Multimodal_RAG_System/src/services/evaluationApi.test.ts`
- Modify: `pdftopng/docs/superpowers/plans/2026-07-25-v9-agent-behavior-observability.md`

**Consumes:** completed Tasks 1–4.

**Produces:** a repeatable verification path proving the dashboard reflects durable runtime events, not mode labels.

- [ ] **Step 1: Add an API compatibility test.**

  Assert the research endpoint returns `behavior_schema_version="2"`, carries a v9 row for an `agentic` result with v9 materialization, and reports a failed `agentic-v9` alias as `failed` rather than `not_applicable`.

- [ ] **Step 2: Run full targeted verification.**

  Run:

  ```powershell
  pytest tests/test_campaign_schemas.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_repository.py tests/test_evaluation_execution_worker.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_visual_evidence_extractor.py tests/test_evaluation_analytics_api.py -q
  ```

  Run:

  ```powershell
  npm run lint
  npm test -- --run src/components/evaluation/AgentBehaviorTab.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/pages/EvaluationCenter.integration.test.tsx src/types/evaluation.contract.test.ts src/services/evaluationApi.test.ts
  ```

- [ ] **Step 3: Deploy in safe order.**

  1. Deploy backend schema v2 and runtime changes.
  2. Confirm `/research/agent-behavior` returns `behavior_schema_version: "2"`.
  3. Deploy frontend.
  4. Verify the backend build/version endpoint or deployment image digest matches the released commit before relying on the UI.

- [ ] **Step 4: Run production smoke before full RAGAS.**

  Execute Q14 (required graph locator) and Q15 or Q16 (visual-required). Verify durable rows directly:

  ```sql
  SELECT run_id, graph_search_mode, fallback_reason
  FROM evaluation_graph_events
  WHERE campaign_id = :campaign_id;

  SELECT run_id, stage_name, status
  FROM evaluation_trace_events
  WHERE campaign_id = :campaign_id
    AND stage_name IN ('graph_locator', 'visual_extract');
  ```

  Acceptance: required stages either have a successful event and source-bound evidence, or the result is `qualified_partial`/failed. No completed required-stage run may lack its event.

- [ ] **Step 5: Commit verification coverage.**

  Commit: `test(evaluation): verify v9 behavior contract and required stages`

## Acceptance criteria

- Latest campaign-like v9 runs display evidence/slot/route metrics rather than legacy zeros.
- A persisted v9 trace never appears as `not_instrumented` under backend schema version 2.
- Q6-style failed `agentic-v9` runs are visibly failed with a safe reason state.
- A graph or visual requirement is visibly distinct from actual execution.
- `required_locator` and `visual_required` cannot result in a `complete` v9 response unless the corresponding durable success event exists.
- Graph evidence is source-bound and persisted; a graph mode label alone is never treated as traversal evidence.
- v8 and naive historical campaigns remain readable with accurate N/A semantics.
- Backend and frontend targeted tests, lint, and two-question production smoke pass before re-running the full benchmark.
