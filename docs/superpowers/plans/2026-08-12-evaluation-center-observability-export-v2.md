# Evaluation Center Observability and Export v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every active Evaluation Center value traceable to a real backend projection and replace the campaign export with a clean Schema v2 that optionally includes sanitized observability for every run.

**Architecture:** Keep the campaign-scoped observability endpoint as the selected-run authority and move its assembly into `ResearchAnalyticsService`. Frontend panels and the new export composer consume the same typed projections; export summary sections reuse existing analytics services, while opt-in full run observability uses campaign-level bulk repository reads.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLite repositories, pytest; React 18, TypeScript 5.9, Chakra UI, Axios, Vitest, Testing Library.

## Global Constraints

- Base every implementation decision on code contracts and tests; do not use local `.db` rows as production evidence.
- `GET /api/evaluation/campaigns/{campaign_id}/runs/{run_id}/observability` is the canonical selected-run endpoint.
- Do not add a third run-detail endpoint.
- Do not expose arbitrary provider payloads, secrets, stack traces, or unrestricted claim/trace payloads through the interactive endpoint.
- Keep nullable values nullable. Missing or non-finite values must not become zero.
- Mark derived and heuristic values explicitly; never label them measured.
- Export Schema v2 replaces the old response shape. No export-response compatibility layer is required.
- `include_run_observability` defaults to `false` in backend and frontend.
- Full observability export must use campaign-level bulk loaders and must not silently truncate.
- Each task changes one Git repository and ends with exactly one focused commit.
- Update affected API/UI documentation in the same task as the contract or behavior change.
- Do not stage unrelated user changes, generated runtime data, local databases, or test caches.
- At every Wave checkpoint: report commit hashes and tests, provide the real-system checklist, then stop until the user explicitly accepts the Wave.
- Backend repository: `D:/flutterserver/pdftopng`.
- Frontend repository: `D:/flutterserver/Multimodal_RAG_System`.

---

## File Map

| Unit | Files | Responsibility |
| --- | --- | --- |
| Canonical run projection | `pdftopng/evaluation/research_analytics.py`, `pdftopng/evaluation/trace_schemas.py`, `pdftopng/evaluation/router.py` | Assemble one safe typed selected-run response, including v9 and human ratings. |
| Safe evidence projection | `pdftopng/evaluation/trace_schemas.py`, `pdftopng/evaluation/research_analytics.py`, `pdftopng/evaluation/campaign_execution.py`, `pdftopng/evaluation/observability.py` | Promote allow-listed claim fields and retrieval provenance without a database migration. |
| Backend truth semantics | `pdftopng/evaluation/campaign_schemas.py`, `pdftopng/evaluation/analytics.py`, `pdftopng/evaluation/router.py` | Preserve unknown tokens/deltas and emit the real latest result identifier. |
| Export v2 composer | `pdftopng/evaluation/export_service.py`, `pdftopng/evaluation/campaign_schemas.py`, `pdftopng/evaluation/router.py` | Compose canonical campaign sections and optional bulk run observability. |
| Frontend contract | `Multimodal_RAG_System/src/types/evaluation.ts`, `Multimodal_RAG_System/src/services/evaluationApi.ts` | Strict selected-run and export v2 types. |
| Frontend composition | `Multimodal_RAG_System/src/pages/EvaluationCenter.tsx`, `Multimodal_RAG_System/src/pages/EvaluationCenter.mappers.ts` | Lazy tab loading, campaign/run freshness, and typed projection mapping. |
| Panel truthfulness | `Multimodal_RAG_System/src/components/evaluation/*` | Remove fabricated fields, render capability states once, and preserve nulls. |
| Operations | `Multimodal_RAG_System/src/components/evaluation/EvaluationJobPanel.tsx`, `Multimodal_RAG_System/src/pages/EvaluationCenter.tsx` | Mount existing durable job state for the selected campaign. |
| Backend docs | `pdftopng/docs/product-specs/evaluation-api.md`, `pdftopng/docs/evaluation-center.md`, `pdftopng/docs/BACKEND.md` | Authoritative API and operational semantics. |
| Frontend docs | `Multimodal_RAG_System/docs/design-docs/evaluation-center.md`, `Multimodal_RAG_System/docs/FRONTEND.md`, `Multimodal_RAG_System/docs/generated/ui-surface.md` | Mounted surfaces, service calls, export action, and capability notices. |

---

# Wave 1 — Canonical Run Observability

### Task 1: Move selected-run observability into the research analytics service

**Repository:** `pdftopng`

**Files:**
- Modify: `evaluation/trace_schemas.py:356-378`
- Modify: `evaluation/research_analytics.py:79-310`
- Modify: `evaluation/router.py:792-961`
- Test: `tests/test_evaluation_research_api.py`
- Test: `tests/test_evaluation_research_analytics.py`
- Modify: `docs/product-specs/evaluation-api.md`

**Interfaces:**
- Consumes: `CampaignResultRepository`, `EvaluationObservabilityRepository`, `EvaluationAccountingStore`, and existing `V9ExecutionObservability` materialization models.
- Produces:

```python
async def ResearchAnalyticsService.get_run_observability(
    self,
    *,
    user_id: str,
    campaign_id: str,
    run_id: str,
) -> EvaluationRunObservabilityDetail:
    """Return the safe canonical projection for one owned campaign run."""
```

- Extends `EvaluationRunObservabilityDetail` with:

```python
agentic_v9: V9ExecutionObservability | None = None
```

- [ ] **Step 1: Add a failing service test for the complete canonical response**

Seed one owned v9 result, one trace event, one LLM call, one retrieval row, one
graph row, one claim, one human rating, one accounting scope, and one v9 attempt
materialization. Assert the public service response contains all normalized
families and the v9 envelope:

```python
detail = await service.get_run_observability(
    user_id="user-1", campaign_id="cmp-1", run_id="run-1"
)

assert detail.run_id == "run-1"
assert detail.run_summary is not None
assert detail.accounting_diagnostics.accounting_status == "complete"
assert [row.human_rating_id for row in detail.human_ratings] == ["rating-1"]
assert detail.agentic_v9 is not None
assert detail.agentic_v9.attempt_id == "attempt-1"
```

- [ ] **Step 2: Add failing route and ownership tests**

Override `get_research_analytics_service`, call the campaign-scoped route, and
assert it delegates exactly once. Call with a run owned by another user and
assert `404`:

```python
response = client.get(
    "/api/evaluation/campaigns/cmp-1/runs/run-1/observability",
    headers=auth_headers("user-1"),
)
assert response.status_code == 200
assert response.json()["agentic_v9"]["attempt_id"] == "attempt-1"
```

- [ ] **Step 3: Run the focused tests and confirm the contract fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py -k "run_observability" -q
```

Expected: failure because `get_run_observability` and the `agentic_v9` response
field do not exist in the canonical schema.

- [ ] **Step 4: Add the schema field and move assembly into the service**

Implement `get_run_observability` by moving the route's repository reads and
safe copies into `ResearchAnalyticsService`. Use the already loaded result's
`source_attempt_id` to load v9 materialization; do not issue the direct SQL used
by the legacy detail path.

Keep these interactive sanitizers exact:

```python
safe_trace = item.model_copy(update={"payload": {}, "error": {}})
safe_llm = item.model_copy(
    update={
        "prompt_preview": safe_plain_text_excerpt(item.prompt_preview),
        "payload": {},
        "error": {},
    }
)
```

Populate `human_ratings` from `list_human_ratings_for_run(run_id)`. Preserve
the existing graph/evidence/accounting status calculations.

- [ ] **Step 5: Reduce the route to authorization-free delegation**

The service performs ownership checks through campaign/result repositories, so
the route becomes:

```python
return await analytics.get_run_observability(
    user_id=user_id,
    campaign_id=campaign_id,
    run_id=run_id,
)
```

Remove route-local repository construction and projection code.

- [ ] **Step 6: Update the backend API document**

Document the canonical endpoint, all response families, nullable v9 behavior,
human ratings, and the interactive redaction boundary in
`docs/product-specs/evaluation-api.md`.

- [ ] **Step 7: Verify and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_observability_schema.py tests/test_evaluation_human_ratings.py -q
```

Then:

```powershell
git add evaluation/trace_schemas.py evaluation/research_analytics.py evaluation/router.py tests/test_evaluation_research_api.py tests/test_evaluation_research_analytics.py docs/product-specs/evaluation-api.md
git commit -m "fix(evaluation): unify campaign run observability"
```

### Task 2: Make the frontend consume the canonical observability contract

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/types/evaluation.ts:932-973`
- Modify: `src/services/evaluationApi.ts:400-404`
- Modify: `src/services/evaluationApi.test.ts`
- Modify: `src/pages/EvaluationCenter.tsx:43-49,209-329`
- Modify: `src/pages/EvaluationCenter.integration.test.tsx`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `docs/design-docs/evaluation-center.md`

**Interfaces:**
- Consumes: Task 1 `EvaluationRunObservabilityDetail` JSON response.
- Produces the following normalized frontend types. Arbitrary JSON is retained
  only for explicitly redacted `payload`/`error` members:

```ts
type SafeJsonObject = Record<string, unknown>;

export interface EvaluationRunSummary {
  run_id: string;
  campaign_id: string;
  question_id: string;
  mode: CampaignMode;
  repeat_number: number;
  answer_preview: string | null;
  latency_ms: number | null;
  total_tokens: number | null;
  accounting_status: 'complete' | 'partial' | 'not_available';
  created_at: string;
}

export interface EvaluationTraceEvent {
  event_id: string;
  run_id: string;
  campaign_id: string;
  span_id: string;
  parent_event_id: string | null;
  parent_span_id: string | null;
  event_type: string;
  sequence: number;
  stage_type: string;
  stage_name: string;
  started_at: string;
  ended_at: string | null;
  duration_ms: number | null;
  status: 'running' | 'success' | 'failed' | 'skipped' | 'timeout' | 'partial';
  retry_count: number;
  payload: SafeJsonObject;
  error: SafeJsonObject;
  created_at: string;
}

export interface EvaluationLlmCall {
  llm_call_id: string;
  run_id: string;
  campaign_id: string;
  provider: string | null;
  model_name: string | null;
  phase: string;
  purpose: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_preview: string | null;
  payload: SafeJsonObject;
  error: SafeJsonObject;
}

export interface EvaluationRetrievalEvent {
  retrieval_event_id: string;
  run_id: string;
  campaign_id: string;
  query: string | null;
  query_hash: string | null;
  retriever_name: string | null;
  top_k: number | null;
  result_count: number;
  latency_ms: number | null;
  payload: SafeJsonObject;
  created_at: string;
}

export interface EvaluationRetrievalChunk {
  retrieval_chunk_id: string;
  run_id: string;
  campaign_id: string;
  retrieval_event_id: string;
  chunk_id: string;
  doc_id: string | null;
  page_start: number | null;
  page_end: number | null;
  modality: string | null;
  rank_before_rerank: number | null;
  rank_after_rerank: number | null;
  dense_score: number | null;
  bm25_score: number | null;
  rerank_score: number | null;
  used_in_context: boolean;
  used_in_answer: boolean;
  expected_evidence_match: boolean;
  excerpt: string | null;
  payload: SafeJsonObject;
}

export interface EvaluationContextPack {
  context_pack_id: string;
  run_id: string;
  campaign_id: string;
  input_chunk_count: number;
  packed_chunk_count: number;
  token_count: number;
  retrieved_but_not_packed_evidence: SafeJsonObject[];
  payload: SafeJsonObject;
}

export interface EvaluationToolCall {
  tool_call_id: string;
  run_id: string;
  campaign_id: string;
  tool_name: string;
  action: string | null;
  latency_ms: number | null;
  status: string;
  payload: SafeJsonObject;
}

export interface EvaluationRoutingDecision {
  routing_decision_id: string;
  run_id: string;
  campaign_id: string;
  selected_mode: CampaignMode;
  analysis_type: 'retrospective' | 'actual';
  decision_source: 'deterministic' | 'llm_planner' | 'safe_fallback' | null;
  candidate_routes: string[];
  matched_rules: string[];
  fallback_reason: string | null;
  confidence: number | null;
  reason: string | null;
  payload: SafeJsonObject;
}

export interface EvaluationGraphEvent {
  graph_event_id: string;
  run_id: string;
  campaign_id: string | null;
  graph_query: string;
  graph_search_mode: string;
  graph_evidence_mode: string;
  graph_route: string;
  router_reason: string | null;
  matched_entity_ids: string[];
  community_ids: number[];
  node_count: number;
  edge_count: number;
  path_count: number;
  graph_latency_ms: number | null;
  graph_context_tokens: number;
  graph_to_chunk_success_rate: number | null;
  graph_noise_ratio: number | null;
  created_at: string;
}

export interface EvaluationGraphEvidenceItem {
  graph_evidence_item_id: string;
  graph_event_id: string;
  node_ids: string[];
  edge_ids: string[];
  relation_path: string[];
  source_doc_ids: string[];
  source_chunk_ids: string[];
  pages: number[];
  asset_ids: string[];
  confidence: number;
  provenance_status: 'full' | 'partial' | 'missing';
  used_as_locator: boolean;
  packed_in_context: boolean;
  used_in_answer: boolean;
  supported_claim_ids: string[];
}

export interface EvaluationClaim {
  claim_id: string;
  run_id: string;
  campaign_id: string;
  claim_text: string;
  claim_type: string | null;
  support_status: 'supported' | 'partially_supported' | 'unsupported' | 'contradicted';
  evidence: SafeJsonObject[];
  unsupported_reason: string | null;
  payload: SafeJsonObject;
}

export interface EvaluationHumanRating {
  human_rating_id: string;
  run_id: string;
  campaign_id: string;
  rater_id_hash: string;
  rubric_version: string;
  correctness_score: number;
  faithfulness_score: number;
  completeness_score: number;
  citation_quality_score: number;
  usefulness_score: number;
  comments: string | null;
  is_blinded: boolean;
  shown_mode_label: boolean;
  created_at: string;
}
```

- Produces the canonical container and client:

```ts
export interface EvaluationRunObservabilityDetail {
  run_id: string;
  campaign_id: string;
  run_summary: EvaluationRunSummary | null;
  trace_events: EvaluationTraceEvent[];
  llm_calls: EvaluationLlmCall[];
  retrieval_events: EvaluationRetrievalEvent[];
  retrieval_chunks: EvaluationRetrievalChunk[];
  context_packs: EvaluationContextPack[];
  tool_calls: EvaluationToolCall[];
  routing_decisions: EvaluationRoutingDecision[];
  graph_events: EvaluationGraphEvent[];
  graph_evidence_items: EvaluationGraphEvidenceItem[];
  claims: EvaluationClaim[];
  human_ratings: EvaluationHumanRating[];
  accounting_diagnostics: ResearchTokenBreakdown;
  agentic_v9: V9ExecutionObservability | null;
}

export async function getRunObservability(
  campaignId: string,
  runId: string,
): Promise<EvaluationRunObservabilityDetail>;
```

- [ ] **Step 1: Write failing API and contract tests**

```ts
it('loads canonical campaign run observability', async () => {
  mockedApi.get.mockResolvedValueOnce({ data: canonicalDetail });
  await expect(getRunObservability('cmp-1', 'run-1')).resolves.toEqual(canonicalDetail);
  expect(mockedApi.get).toHaveBeenCalledWith(
    '/api/evaluation/campaigns/cmp-1/runs/run-1/observability',
  );
});

it('accepts a v9 envelope on the canonical response', () => {
  expect(canonicalDetail.agentic_v9?.attempt_id).toBe('attempt-1');
});
```

- [ ] **Step 2: Write a failing direct-entry and stale-response integration test**

Render the page on Run Trace, return `run-old` after `run-new`, and assert the
new run remains displayed. Also assert v9 content renders without first opening
another tab:

```ts
expect(await screen.findByText('attempt-1')).toBeInTheDocument();
expect(screen.queryByText('old answer')).not.toBeInTheDocument();
```

- [ ] **Step 3: Run the focused frontend tests and confirm failure**

Run:

```powershell
npm test -- --run src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/pages/EvaluationCenter.integration.test.tsx
```

Expected: failure because the canonical type/function names and typed v9
response are not implemented.

- [ ] **Step 4: Replace the loose detail type and client name**

Replace `RunDetailResponse` with `EvaluationRunObservabilityDetail`, using the
existing concrete evaluation interfaces instead of
`Array<Record<string, unknown>>`. Rename `getRunDetail` to
`getRunObservability` and update every production/test import.

- [ ] **Step 5: Preserve and test request-generation guards**

Keep `runDetailRequestRef` and `requestGenerationRef` as the single freshness
mechanism. On campaign change, clear selected run detail before starting the
new request. Apply the same guard to initial tab loading so the `case 2/3/5`
Promise result cannot overwrite a later campaign selection.

- [ ] **Step 6: Update frontend design documentation**

Replace references to a generic run detail call with the canonical campaign
observability endpoint and document that all v9 panels share its envelope.

- [ ] **Step 7: Verify and commit**

Run:

```powershell
npm test -- --run src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.ui.test.tsx
npm run build
```

Then:

```powershell
git add src/types/evaluation.ts src/services/evaluationApi.ts src/services/evaluationApi.test.ts src/pages/EvaluationCenter.tsx src/pages/EvaluationCenter.integration.test.tsx src/types/evaluation.contract.test.ts docs/design-docs/evaluation-center.md
git commit -m "fix(evaluation-ui): consume canonical run observability"
```

### Task 3: Add safe claim fields and explicit retrieval provenance

**Repository:** `pdftopng`

**Files:**
- Modify: `evaluation/trace_schemas.py:126-150,232-248,356-378`
- Modify: `evaluation/research_analytics.py`
- Modify: `evaluation/campaign_execution.py:987-1211`
- Modify: `evaluation/observability.py`
- Test: `tests/test_evaluation_execution_observability.py`
- Test: `tests/test_evaluation_research_analytics.py`
- Test: `tests/test_evaluation_observability_schema.py`
- Modify: `docs/evaluation-center.md`

**Interfaces:**
- Consumes: Task 1 canonical run projection.
- Produces:

```python
AvailabilityStatus = Literal[
    "complete", "partial", "not_instrumented", "not_available", "not_applicable"
]
ObservationProvenance = Literal["measured", "persisted", "derived", "heuristic"]

class ObservationAvailability(BaseModel):
    status: AvailabilityStatus
    reasons: list[str] = Field(default_factory=list)

class EvaluationRetrievalChunkProjection(EvaluationRetrievalChunk):
    used_in_context: bool | None = None
    used_in_answer: bool | None = None
    expected_evidence_match: bool | None = None
    provenance: ObservationProvenance
    availability: ObservationAvailability
    payload: dict[str, Any] = Field(default_factory=dict)

class ClaimEvidenceReference(BaseModel):
    evidence_id: str | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None

class EvaluationClaimProjection(EvaluationClaim):
    evidence_refs: list[ClaimEvidenceReference] = Field(default_factory=list)
    repair_action: str | None = None
    post_repair_status: str | None = None
    extraction_status: Literal["recorded", "empty", "not_instrumented"]
    payload: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 1: Write failing provenance tests for both retrieval write paths**

For result-context synthesis, assert the recorder persists reserved metadata
inside the existing payload JSON and does not claim a measured attribution:

```python
chunk = recorded_chunks[0]
assert chunk.payload["observation_provenance"] == "derived"
assert chunk.payload["used_in_answer_provenance"] == "heuristic"
assert chunk.dense_score is None
assert chunk.bm25_score is None
```

For a normalized retriever event with measured attribution, assert
`observation_provenance == "measured"`.

- [ ] **Step 2: Write failing safe claim projection tests**

Seed claim evidence containing allow-listed IDs plus a secret-like extra field,
and payload repair fields plus an arbitrary provider body. Assert:

```python
claim = detail.claims[0]
assert claim.evidence_refs[0].chunk_id == "chunk-1"
assert claim.repair_action == "retry_retrieval"
assert claim.post_repair_status == "supported"
assert claim.payload == {}
assert "provider_body" not in claim.model_dump_json()
```

Add separate fixtures for an explicitly recorded empty claim list and for no
claim instrumentation; assert `empty` and `not_instrumented` remain distinct.

- [ ] **Step 3: Run focused tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_execution_observability.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_schema.py -k "provenance or claim_projection" -q
```

- [ ] **Step 4: Record provenance without a table migration**

At each write path, store these reserved payload keys:

```python
payload.update(
    {
        "observation_provenance": "derived",
        "availability_status": "partial",
        "availability_reasons": ["result_context_reconstruction"],
        "used_in_answer_provenance": "heuristic",
    }
)
```

Use `measured/complete` only where the recorder receives actual retriever
instrumentation. Historical rows without reserved keys project to
`not_available` with reason `provenance_not_recorded`; do not infer measured
status from a chunk identifier.

- [ ] **Step 5: Promote allow-listed fields in the safe projector**

Add pure helpers in `research_analytics.py`:

```python
def _project_retrieval_chunk(item: EvaluationRetrievalChunk) -> EvaluationRetrievalChunkProjection:
    metadata = item.payload if isinstance(item.payload, dict) else {}
    return EvaluationRetrievalChunkProjection(
        **item.model_dump(exclude={"payload", "used_in_context", "used_in_answer", "expected_evidence_match"}),
        used_in_context=item.used_in_context if metadata.get("availability_status") != "not_available" else None,
        used_in_answer=item.used_in_answer if metadata.get("availability_status") != "not_available" else None,
        expected_evidence_match=item.expected_evidence_match if metadata.get("availability_status") != "not_available" else None,
        provenance=metadata.get("observation_provenance", "persisted"),
        availability=_availability_from_metadata(metadata),
        payload={},
    )
```

Implement `_project_claim` with an explicit key allow-list:
`evidence_id`, `doc_id`, `chunk_id`, and integer `page`; accept scalar
`repair_action` and `post_repair_status` only. Never return the original claim
payload or evidence dictionaries.

- [ ] **Step 6: Update observability documentation**

Document the four provenance values, five availability states, historical-row
behavior, and the typed claim allow-list in `docs/evaluation-center.md`.

- [ ] **Step 7: Verify and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_execution_observability.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_schema.py tests/test_evaluation_export_redaction.py -q
```

Then:

```powershell
git add evaluation/trace_schemas.py evaluation/research_analytics.py evaluation/campaign_execution.py evaluation/observability.py tests/test_evaluation_execution_observability.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_schema.py docs/evaluation-center.md
git commit -m "fix(evaluation): project evidence provenance safely"
```

### Task 4: Map typed retrieval, claim, and graph evidence in the frontend

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/types/evaluation.ts`
- Modify: `src/pages/EvaluationCenter.mappers.ts:205-286`
- Modify: `src/pages/EvaluationCenter.mappers.test.ts`
- Modify: `src/pages/EvaluationCenter.tsx:89-101`
- Modify: `src/components/evaluation/RetrievalEvidenceTab.tsx`
- Modify: `src/components/evaluation/ClaimEvidenceTable.tsx`
- Test: `src/components/evaluation/RetrievalEvidenceTab.test.tsx`
- Test: `src/components/evaluation/ClaimEvidenceTab.test.tsx`
- Modify: `docs/design-docs/evaluation-center.md`

**Interfaces:**
- Consumes: Task 3 `EvaluationRetrievalChunkProjection`,
  `EvaluationClaimProjection`, and normalized `EvaluationGraphEvidenceItem`.
- Produces frontend view rows with `provenance` and `availabilityStatus`; graph
  sources use plural normalized arrays.

- [ ] **Step 1: Add failing mapper tests**

```ts
expect(mapped.chunks[0]).toMatchObject({
  inContext: true,
  usedInAnswer: false,
  provenance: 'derived',
  availabilityStatus: 'partial',
});

expect(mapped.graphEvidence[0]).toMatchObject({
  sourceDocIds: ['doc-1'],
  sourceChunkIds: ['chunk-1'],
  pages: [4],
  assetIds: ['asset-1'],
});

expect(mappedClaims[0]).toMatchObject({
  evidence: ['chunk-1'],
  repairAction: 'retry_retrieval',
  postRepairStatus: 'supported',
});
```

Add a historical chunk with `not_available` and assert all three boolean view
fields are null rather than false.

- [ ] **Step 2: Add failing panel tests for provenance copy**

Assert the Retrieval panel displays `Derived` or `Heuristic` next to affected
values and one availability reason. Assert typed claim references render while
the raw provider field does not.

- [ ] **Step 3: Run tests and confirm failure**

Run:

```powershell
npm test -- --run src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RetrievalEvidenceTab.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx
```

- [ ] **Step 4: Replace payload-based instrumentation guesses**

Delete the `payload.instrumentation_depth` check from `mapRetrieval`. Map
nullable booleans directly and propagate `provenance` and
`availability.status/reasons` to the view model.

Map graph evidence from:

```ts
sourceDocIds: stringArray(row.source_doc_ids),
sourceChunkIds: stringArray(row.source_chunk_ids),
pages: numberArray(row.pages),
assetIds: stringArray(row.asset_ids),
```

Map claims only from `evidence_refs`, `repair_action`, and
`post_repair_status`.

- [ ] **Step 5: Render honest provenance without adding another table**

Add a compact provenance/status label to the existing retrieval table or its
summary area. Keep missing scores as `N/A`; do not default missing modality to
`text` unless the backend provides `text`.

- [ ] **Step 6: Update frontend documentation and verify**

Document typed safe evidence and provenance rendering. Run:

```powershell
npm test -- --run src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RetrievalEvidenceTab.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/pages/EvaluationCenter.integration.test.tsx
npm run build
```

- [ ] **Step 7: Commit**

```powershell
git add src/types/evaluation.ts src/pages/EvaluationCenter.mappers.ts src/pages/EvaluationCenter.mappers.test.ts src/pages/EvaluationCenter.tsx src/components/evaluation/RetrievalEvidenceTab.tsx src/components/evaluation/ClaimEvidenceTable.tsx src/components/evaluation/RetrievalEvidenceTab.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx docs/design-docs/evaluation-center.md
git commit -m "fix(evaluation-ui): map typed evidence observability"
```

## Wave 1 Checkpoint — Mandatory Stop

- [ ] Run backend Wave 1 regression:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_api.py tests/test_evaluation_research_analytics.py tests/test_evaluation_execution_observability.py tests/test_evaluation_observability_schema.py tests/test_evaluation_human_ratings.py tests/test_evaluation_graph_events.py tests/test_evaluation_export_redaction.py -q
```

- [ ] Run frontend Wave 1 regression:

```powershell
npm test -- --run src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RunTraceTab.test.tsx src/components/evaluation/RetrievalEvidenceTab.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx
npm run build
```

- [ ] Report the four commit hashes, test totals, and contract changes.
- [ ] Ask the user to push both repositories.
- [ ] Real system: open one v9 run and verify Trace, Retrieval, Claim, graph
  evidence, human ratings, accounting, and v9 evidence all come from the same
  run selection.
- [ ] Real system: switch campaign/run rapidly and confirm no stale detail.
- [ ] Stop. Do not begin Wave 2 until the user replies that Wave 1 passed.

---

# Wave 2 — Panel Truthfulness and Operations

### Task 5: Preserve unknown tokens/deltas and emit the latest result ID

**Repository:** `pdftopng`

**Files:**
- Modify: `evaluation/campaign_schemas.py:598-615,673-686,902-914`
- Modify: `evaluation/analytics.py:1710-1729,1949-1986`
- Modify: `evaluation/router.py:1070-1125`
- Test: `tests/test_evaluation_analytics_api.py`
- Test: `tests/test_evaluation_api.py`
- Modify: `docs/product-specs/evaluation-api.md`

**Interfaces:**
- Produces `EvaluationRunListItem.total_tokens: int | None`.
- Produces `RunDiffResponse.token_delta: int | None` and
  `comparable=False` when either token total is unknown.
- Populates `CampaignProgressEvent.latest_result_id` from the newest persisted
  result visible to the authenticated campaign owner.

- [ ] **Step 1: Write failing null-semantic tests**

```python
runs = await service.list_runs(user_id="user-1", campaign_id="cmp-1")
assert runs.runs[0].total_tokens is None

diff = await service.run_diff(
    user_id="user-1", run_id="run-2", baseline_run_id="run-1"
)
assert diff.token_delta is None
assert diff.comparable is False
```

- [ ] **Step 2: Write a failing SSE test**

Seed `run-1`, advance campaign progress, consume the next `campaign_progress`
event, and assert:

```python
assert event_payload["latest_result_id"] == "run-1"
```

- [ ] **Step 3: Run focused tests and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_analytics_api.py tests/test_evaluation_api.py -k "unknown_tokens or latest_result_id or run_diff" -q
```

- [ ] **Step 4: Make schemas and builders nullable**

Replace `item.total_tokens or 0` with `item.total_tokens`. Calculate token delta
only when both operands are integers:

```python
tokens_comparable = result.total_tokens is not None and baseline.total_tokens is not None
token_delta = (
    result.total_tokens - baseline.total_tokens
    if tokens_comparable
    else None
)
```

Set `comparable=tokens_comparable and latency_comparable`; keep answer-change
status independent.

- [ ] **Step 5: Populate `latest_result_id` only when progress changes**

Add a router helper that calls the bounded campaign result list, sorts by
`created_at`, and returns the newest ID. Invoke it inside the existing
`progress_state != last_progress` branch, not on every one-second loop.

- [ ] **Step 6: Document null and SSE semantics, verify, and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_analytics_api.py tests/test_evaluation_api.py tests/test_campaign_schemas.py -q
git add evaluation/campaign_schemas.py evaluation/analytics.py evaluation/router.py tests/test_evaluation_analytics_api.py tests/test_evaluation_api.py docs/product-specs/evaluation-api.md
git commit -m "fix(evaluation): preserve unknown metrics and progress ids"
```

### Task 6: Separate Router retrospective analysis from actual execution

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/pages/EvaluationCenter.tsx:209-243,410-421`
- Modify: `src/pages/EvaluationCenter.mappers.ts:327-365`
- Modify: `src/pages/EvaluationCenter.mappers.test.ts`
- Modify: `src/components/evaluation/RouterLabTab.tsx`
- Modify: `src/components/evaluation/RouterLabTab.test.tsx`
- Modify: `src/components/evaluation/QuestionAnalysisTab.tsx`
- Modify: `src/components/evaluation/QuestionAnalysisTab.test.tsx`
- Modify: `src/types/evaluation.ts`
- Modify: `docs/design-docs/evaluation-center.md`

**Interfaces:**
- Router Lab receives `analysisType`, recorded decision rows, and an independent
  selected-run `executionRoute`.
- The view model no longer contains `oracleLabelSource`, fabricated
  `utilityFormula`, `confusionMatrix`, or unsupported KPI values.
- Question rows no longer contain `routerSelectedMode` or `ablationFlags`.

- [ ] **Step 1: Write failing direct Router-entry tests**

Select Router Lab as the first run-dependent tab. Assert the page requests, in
order-independent form, router analysis, campaign runs, and selected-run
observability. Assert the execution card uses `agentic_v9.route` while the
campaign table remains labelled `Retrospective`.

```ts
expect(getRouterAnalysis).toHaveBeenCalledWith('cmp-1');
expect(getCampaignRuns).toHaveBeenCalledWith('cmp-1');
expect(getRunObservability).toHaveBeenCalledWith('cmp-1', 'run-1');
expect(await screen.findByText('Retrospective analysis')).toBeInTheDocument();
expect(screen.getByText('Actual run route')).toBeInTheDocument();
```

- [ ] **Step 2: Write failing absence tests for fabricated data**

With the current backend router response, assert saved tokens, quality loss,
regret, oracle label, fallback formula, and confusion matrix headings are not
rendered. Assert Question Analysis has no Router Selected Mode or Ablation
Flags heading.

- [ ] **Step 3: Run focused tests and confirm failure**

```powershell
npm test -- --run src/pages/EvaluationCenter.ui.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RouterLabTab.test.tsx src/components/evaluation/QuestionAnalysisTab.test.tsx
```

- [ ] **Step 4: Load Router dependencies in one tab branch**

Change `case 6` to load router analysis and campaign runs together, select the
existing preferred run or first run, then call `getRunObservability`. Return all
three in the tab data object. Reuse the existing request-generation guard and
clear execution route when campaign changes.

- [ ] **Step 5: Shrink Router and Question view models**

Map only backend-provided values. Remove these assignments:

```ts
oracleLabelSource: 'utility_best_mode'
utilityFormula: 'Retrospective utility summary from recorded routing decisions.'
confusionMatrix: []
routerSelectedMode: 'N/A'
ablationFlags: []
```

Render optional KPI/matrix sections only when a future typed response actually
contains them; the current interface omits them entirely.

- [ ] **Step 6: Document, verify, and commit**

```powershell
npm test -- --run src/pages/EvaluationCenter.ui.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RouterLabTab.test.tsx src/components/evaluation/QuestionAnalysisTab.test.tsx
npm run build
git add src/pages/EvaluationCenter.tsx src/pages/EvaluationCenter.mappers.ts src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RouterLabTab.tsx src/components/evaluation/RouterLabTab.test.tsx src/components/evaluation/QuestionAnalysisTab.tsx src/components/evaluation/QuestionAnalysisTab.test.tsx src/types/evaluation.ts docs/design-docs/evaluation-center.md
git commit -m "fix(evaluation-ui): separate router analysis from execution"
```

### Task 7: Remove uninstrumented v9 placeholders and false zero counts

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/components/evaluation/V9EvidenceExplorer.tsx`
- Modify: `src/components/evaluation/V9EvidenceExplorer.test.tsx`
- Modify: `src/components/evaluation/ClaimEvidenceTab.tsx`
- Modify: `src/components/evaluation/ClaimEvidenceTab.test.tsx`
- Modify: `src/components/evaluation/AblationDashboardTab.tsx:59-65,323-441`
- Modify: `src/components/evaluation/AblationDashboardTab.test.tsx`
- Modify: `src/components/evaluation/RunTraceTree.tsx:104`
- Modify: `src/components/evaluation/RunTraceTab.test.tsx`
- Modify: `docs/FRONTEND.md`

**Interfaces:**
- Produces one reusable `CapabilityNotice` copy per affected section.
- `formatCount(value)` returns `N/A` for missing/non-finite input and preserves
  a real numeric zero.
- A duration of `0` renders `0 ms`, not `n/a`.

- [ ] **Step 1: Write failing component tests**

```ts
expect(screen.queryByRole('columnheader', { name: 'Cited' })).not.toBeInTheDocument();
expect(screen.queryByText('N/A — not instrumented per slot')).not.toBeInTheDocument();
expect(screen.getAllByText(/not instrumented/i)).toHaveLength(1);

expect(screen.getByText('Samples: N/A')).toBeInTheDocument();
expect(screen.getByText('Failed: 0')).toBeInTheDocument();
expect(screen.getByText('0 ms')).toBeInTheDocument();
```

- [ ] **Step 2: Run focused tests and confirm failure**

```powershell
npm test -- --run src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/RunTraceTab.test.tsx
```

- [ ] **Step 3: Remove repeated unsupported fields**

Delete the Cited column, per-slot/per-source token rows, and per-slot Graph
column when their value is hard-coded. Add one Alert or Text block per section:

```text
Per-evidence citation attribution and per-slot/source token accounting are not instrumented for this run.
```

Render the notice only when the corresponding typed capability is absent.

- [ ] **Step 4: Fix null and zero formatters**

```ts
function formatCount(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString()
    : 'N/A';
}
```

Change duration rendering to an explicit null check:

```ts
typeof event.durationMs === 'number' ? `${event.durationMs} ms` : 'n/a'
```

- [ ] **Step 5: Update docs, verify, and commit**

```powershell
npm test -- --run src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/RunTraceTab.test.tsx
npm run build
git add src/components/evaluation/V9EvidenceExplorer.tsx src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/ClaimEvidenceTab.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/AblationDashboardTab.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/RunTraceTree.tsx src/components/evaluation/RunTraceTab.test.tsx docs/FRONTEND.md
git commit -m "fix(evaluation-ui): remove uninstrumented panel placeholders"
```

### Task 8: Mount durable job state for the selected campaign

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/pages/EvaluationCenter.tsx`
- Modify: `src/pages/EvaluationCenter.ui.test.tsx`
- Modify: `src/components/evaluation/EvaluationJobPanel.tsx`
- Modify: `src/components/evaluation/EvaluationJobPanel.test.tsx`
- Modify: `docs/FRONTEND.md`
- Modify: `docs/generated/ui-surface.md`

**Interfaces:**
- Consumes existing `EvaluationJobPanelProps` and selected campaign ID.
- Produces one mounted durable-job panel in the selected campaign operation
  area; it is absent when no campaign is selected.

- [ ] **Step 1: Write a failing mount/lifecycle test**

```ts
expect(screen.queryByText('Durable evaluation job')).not.toBeInTheDocument();
await user.selectOptions(screen.getByLabelText('Campaign'), 'cmp-1');
expect(await screen.findByText('Durable evaluation job')).toBeInTheDocument();
expect(listCampaignJobs).toHaveBeenCalledWith('cmp-1');
```

Switch to `cmp-2` and assert the old job ID disappears and the panel loads
`cmp-2`. Add a terminal callback test that refreshes campaign inventory once.

- [ ] **Step 2: Run focused tests and confirm failure**

```powershell
npm test -- --run src/pages/EvaluationCenter.ui.test.tsx src/components/evaluation/EvaluationJobPanel.test.tsx
```

- [ ] **Step 3: Mount the existing component without a new tab**

Lazy-import `EvaluationJobPanel` and render it below the selected campaign
controls/operation area:

```tsx
{selectedCampaignId ? (
  <EvaluationJobPanel
    key={selectedCampaignId}
    campaignId={selectedCampaignId}
    onJobTerminal={() => void refreshCampaignInventory()}
  />
) : null}
```

Use the campaign ID as the React key so local polling state cannot leak across
campaigns. Reuse existing toast/error behavior.

- [ ] **Step 4: Refresh docs and generated inventory**

Update `docs/FRONTEND.md`, then run:

```powershell
npm run docs:sync
npm run docs:check
```

- [ ] **Step 5: Verify and commit**

```powershell
npm test -- --run src/pages/EvaluationCenter.ui.test.tsx src/components/evaluation/EvaluationJobPanel.test.tsx
npm run build
git add src/pages/EvaluationCenter.tsx src/pages/EvaluationCenter.ui.test.tsx src/components/evaluation/EvaluationJobPanel.tsx src/components/evaluation/EvaluationJobPanel.test.tsx docs/FRONTEND.md docs/generated/ui-surface.md
git commit -m "feat(evaluation-ui): surface durable campaign jobs"
```

## Wave 2 Checkpoint — Mandatory Stop

- [ ] Run backend Wave 2 regression:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_analytics_api.py tests/test_evaluation_api.py tests/test_campaign_schemas.py tests/test_evaluation_job_store.py tests/test_evaluation_job_worker.py -q
```

- [ ] Run frontend Wave 2 regression:

```powershell
npm test -- --run src/pages/EvaluationCenter.ui.test.tsx src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RouterLabTab.test.tsx src/components/evaluation/QuestionAnalysisTab.test.tsx src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/EvaluationJobPanel.test.tsx
npm run build
```

- [ ] Report four commit hashes and test totals.
- [ ] Real system: enter Router Lab first, confirm retrospective and actual-run
  sections stay separate, then switch campaigns.
- [ ] Real system: verify missing instrumentation appears once and unknown
  counts do not display zero.
- [ ] Real system: verify selected campaign durable job polling, retry, terminal
  refresh, and latest result refresh.
- [ ] Stop. Do not begin Wave 3 until the user replies that Wave 2 passed.

---

# Wave 3 — Export Schema v2

### Task 9: Replace the backend export response with Schema v2 summaries

**Repository:** `pdftopng`

**Files:**
- Create: `evaluation/export_service.py`
- Modify: `evaluation/campaign_schemas.py:874-899`
- Modify: `evaluation/router.py:150-193,636-646`
- Modify: `evaluation/analytics.py:1399-1646`
- Test: `tests/test_evaluation_export_redaction.py`
- Test: `tests/test_evaluation_analytics_api.py`
- Modify: `docs/product-specs/evaluation-api.md`
- Modify: `docs/BACKEND.md`

**Interfaces:**
- Produces:

```python
class ExportAvailability(BaseModel):
    status: AvailabilityStatus
    reasons: list[str] = Field(default_factory=list)

class ExportSection(BaseModel):
    availability: ExportAvailability
    data: dict[str, Any] | list[Any] | None

class ExportRunObservability(BaseModel):
    included: bool
    availability: ExportAvailability
    data: EvaluationRunObservabilityDetail | None = None

class ExportRunV2(BaseModel):
    result: dict[str, Any]
    ragas_metrics: dict[str, float] = Field(default_factory=dict)
    observability: ExportRunObservability

class ExportCampaignResponse(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    export_metadata: dict[str, Any]
    campaign: dict[str, Any]
    sections: dict[str, ExportSection]
    runs: list[ExportRunV2] = Field(default_factory=list)

class EvaluationExportService:
    async def export_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        request: ExportCampaignRequest,
    ) -> ExportCampaignResponse:
        """Compose Schema v2 from canonical analytics projections."""
```

- [ ] **Step 1: Write a failing Schema v2 contract test**

Call export with default options and assert exact top-level keys and default
run observability state:

```python
payload = response.json()
assert set(payload) == {
    "schema_version", "export_metadata", "campaign", "sections", "runs"
}
assert payload["schema_version"] == "2.0"
assert payload["export_metadata"]["options"]["include_run_observability"] is False
assert payload["runs"][0]["observability"]["included"] is False
assert payload["runs"][0]["observability"]["data"] is None
```

Assert the old top-level `metrics`, `trace_events`, `llm_calls`,
`retrieval_summary`, and `claim_summary` keys are absent.

- [ ] **Step 2: Write failing panel-projection composition tests**

Inject fakes for `ResearchAnalyticsService`, `EvaluationAnalyticsService`, and
`ReleaseMetricsService`. Assert each canonical method is called once and its
exact serialized response appears under:

```text
overview
question_analysis
agent_behavior
router_analysis
ablation
human_evaluation
diagnostics
```

For a campaign without a compatible release benchmark, assert the overview's
release subsection has `not_applicable` rather than an empty complete object.

- [ ] **Step 3: Run tests and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_api.py -k "export_v2" -q
```

- [ ] **Step 4: Add v2 request/response models**

Add `include_run_observability: bool = False` to `ExportCampaignRequest` and
replace the old response fields with the exact models above. Keep the five
existing content/redaction flags and `format="json"`.

- [ ] **Step 5: Implement the bounded export composer**

Construct `EvaluationExportService` with injected canonical services. Use
`asyncio.gather` for independent campaign-level projections. Wrap each result
in `ExportSection` using its own status fields; convert a legitimate missing
optional projection to `not_applicable` or `not_available`, not a fabricated
empty success.

Every run includes the redacted result, finite official RAGAS values, and:

```python
ExportRunObservability(
    included=False,
    availability=ExportAvailability(
        status="not_applicable",
        reasons=["run_observability_not_requested"],
    ),
    data=None,
)
```

- [ ] **Step 6: Switch the route and remove the old assembler**

Create one module-level `EvaluationExportService` after the three canonical
services are initialized, add a dependency factory, and make `/export` call
it. Remove `EvaluationAnalyticsService.export_campaign` and its now-unused
private export-only helpers after `rg "export_campaign"` confirms no remaining
callers.

- [ ] **Step 7: Document, verify, and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_api.py tests/test_evaluation_research_api.py -q
git add evaluation/export_service.py evaluation/campaign_schemas.py evaluation/router.py evaluation/analytics.py tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_api.py docs/product-specs/evaluation-api.md docs/BACKEND.md
git commit -m "feat(evaluation): replace campaign export with schema v2"
```

### Task 10: Add opt-in bulk observability to Export v2

**Repository:** `pdftopng`

**Files:**
- Modify: `evaluation/research_analytics.py:265-309`
- Modify: `evaluation/export_service.py`
- Modify: `evaluation/observability_storage.py`
- Test: `tests/test_evaluation_export_redaction.py`
- Test: `tests/test_evaluation_research_analytics.py`
- Test: `tests/test_evaluation_observability_repository.py`
- Modify: `docs/product-specs/evaluation-api.md`

**Interfaces:**
- Consumes Task 1 private canonical row projector and Task 9 Schema v2.
- Produces:

```python
async def ResearchAnalyticsService.get_campaign_run_observability(
    self,
    *,
    user_id: str,
    campaign_id: str,
    results: Sequence[CampaignResult],
    request: ExportCampaignRequest,
) -> dict[str, EvaluationRunObservabilityDetail]:
    """Build every run from campaign-level bulk records and one accounting load."""
```

- [ ] **Step 1: Write a failing bulk-loader test**

Use repository spies for a three-run campaign. Assert each campaign-level
loader is called once and every `list_*_for_run` method is called zero times:

```python
details = await service.get_campaign_run_observability(
    user_id="user-1",
    campaign_id="cmp-1",
    results=results,
    request=ExportCampaignRequest(include_run_observability=True),
)
assert set(details) == {"run-1", "run-2", "run-3"}
assert observability.list_trace_events_for_campaign.await_count == 1
assert observability.list_trace_events_for_run.await_count == 0
```

Repeat assertions for LLM, retrieval, context, tools, routing, graph, claims,
human ratings, and v9 materializations. Assert accounting scopes/events load
once for the campaign.

- [ ] **Step 2: Write failing redaction matrix tests**

Parameterize safe defaults, raw trace, full prompts, answers off, and excerpts
off. Assert safe defaults exclude raw payloads and secrets, while explicitly
enabled fields include only the requested stored material:

```python
assert exported_run["observability"]["included"] is True
assert exported_run["observability"]["data"]["trace_events"][0]["payload"] == {}
assert "sk-secret" not in response.text
```

Add a test proving all result IDs appear and event arrays are not truncated.

- [ ] **Step 3: Run focused tests and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_redaction.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_repository.py -k "bulk or full_observability" -q
```

- [ ] **Step 4: Extract a pure per-run accounting builder**

Refactor `get_run_token_breakdown` so both selected-run and bulk paths call:

```python
def _token_breakdown_for_run(
    *,
    run_id: str,
    scopes: Sequence[AccountingScope],
    events: Sequence[UsageEvent],
    agentic_execution_version: str,
    observability_partial_reasons: Sequence[str],
) -> TokenBreakdown:
    """Calculate one strict token projection from already loaded campaign rows."""
```

The public single-run method loads campaign scopes/events once and delegates;
the bulk method loads them once total and delegates for each result.

- [ ] **Step 5: Implement campaign-level observability grouping**

Load every normalized family through `list_*_for_campaign`, producing
`dict[run_id, list[row]]`. Use one private `_build_run_observability` for both
single and bulk paths. Apply request redaction after canonical assembly so the
data formula does not differ between interactive and export paths.

If a repository lacks a campaign-level loader, add it in
`observability_storage.py` with one bounded query ordered by run/sequence. Do
not fall back to a per-run loop.

- [ ] **Step 6: Attach complete details to every Schema v2 run**

When requested, require one detail entry per campaign result:

```python
if set(details) != {result.id for result in results}:
    raise AppError(
        code=ErrorCode.INTERNAL_ERROR,
        message="Complete run observability could not be assembled",
        status_code=500,
    )
```

Legitimately empty entity arrays remain empty with explicit availability;
missing run containers fail the export rather than truncate it.

- [ ] **Step 7: Document, verify, and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_redaction.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_repository.py tests/test_evaluation_accounting_store.py -q
git add evaluation/research_analytics.py evaluation/export_service.py evaluation/observability_storage.py tests/test_evaluation_export_redaction.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_repository.py docs/product-specs/evaluation-api.md
git commit -m "feat(evaluation): add bulk run observability export"
```

### Task 11: Add the frontend Export v2 option and strict response contract

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/types/evaluation.ts:975-1002`
- Modify: `src/services/evaluationApi.ts:358-364`
- Modify: `src/services/evaluationApi.test.ts`
- Modify: `src/components/evaluation/AblationDashboardTab.tsx:243-310,469-503`
- Modify: `src/components/evaluation/AblationDashboardTab.test.tsx`
- Modify: `docs/FRONTEND.md`
- Modify: `docs/generated/ui-surface.md`

**Interfaces:**
- Consumes Tasks 9-10 Schema v2.
- Produces strict `ExportCampaignV2`, `ExportSection<T>`, and
  `ExportRunV2` TypeScript interfaces; no `extends Record<string, unknown>`.
- Adds `include_run_observability?: boolean` to `ExportCampaignRequest`.

- [ ] **Step 1: Write failing type/API tests**

```ts
await exportCampaignAnalysis('cmp-1', {
  include_run_observability: true,
  include_raw_trace_payloads: false,
});

expect(mockedApi.post).toHaveBeenCalledWith(
  '/api/evaluation/campaigns/cmp-1/export',
  expect.objectContaining({ include_run_observability: true }),
);
expect(response.schema_version).toBe('2.0');
```

- [ ] **Step 2: Write failing checkbox and filename tests**

Assert the checkbox is off initially and the default request contains false.
Cover all four filenames:

```text
cmp-1-summary-redacted-v2.json
cmp-1-observability-redacted-v2.json
cmp-1-summary-custom-v2.json
cmp-1-observability-custom-v2.json
```

`custom` is selected only by full prompts or raw trace payloads. Assert an
export rejection leaves the previous preview unchanged.

- [ ] **Step 3: Run focused tests and confirm failure**

```powershell
npm test -- --run src/services/evaluationApi.test.ts src/components/evaluation/AblationDashboardTab.test.tsx
```

- [ ] **Step 4: Replace loose export types**

Model every top-level v2 key as required. Type `sections` with named fields and
use the canonical response interfaces for their `data`. Type each run as:

```ts
export interface ExportRunV2 {
  result: CampaignResult;
  ragas_metrics: Record<string, number>;
  observability: {
    included: boolean;
    availability: ExportAvailability;
    data: EvaluationRunObservabilityDetail | null;
  };
}
```

- [ ] **Step 5: Add the default-off UI option and deterministic filename**

Add a checkbox labelled `Include all run observability` with helper text
`Larger file`. Pass the option unchanged. Calculate the filename from two
booleans:

```ts
const scope = options.include_run_observability ? 'observability' : 'summary';
const content = options.include_full_prompts || options.include_raw_trace_payloads
  ? 'custom'
  : 'redacted';
return `${campaignId}-${scope}-${content}-v2.json`;
```

Update preview counts from `response.runs.length` and the v2 sections/runs;
do not read the removed legacy `summary` object.

- [ ] **Step 6: Update generated docs, verify, and commit**

```powershell
npm test -- --run src/services/evaluationApi.test.ts src/components/evaluation/AblationDashboardTab.test.tsx src/pages/EvaluationCenter.ui.test.tsx
npm run build
npm run docs:sync
npm run docs:check
git add src/types/evaluation.ts src/services/evaluationApi.ts src/services/evaluationApi.test.ts src/components/evaluation/AblationDashboardTab.tsx src/components/evaluation/AblationDashboardTab.test.tsx docs/FRONTEND.md docs/generated/ui-surface.md
git commit -m "feat(evaluation-ui): add export v2 observability option"
```

## Wave 3 Checkpoint — Mandatory Stop

- [ ] Run backend Wave 3 regression:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_api.py tests/test_evaluation_research_api.py tests/test_evaluation_research_analytics.py tests/test_evaluation_observability_repository.py tests/test_evaluation_accounting_store.py -q
```

- [ ] Run frontend Wave 3 regression:

```powershell
npm test -- --run src/services/evaluationApi.test.ts src/components/evaluation/AblationDashboardTab.test.tsx src/pages/EvaluationCenter.ui.test.tsx src/types/evaluation.contract.test.ts
npm run build
npm run docs:check
```

- [ ] Report three commit hashes, response schema, and bulk loader call counts.
- [ ] Real system: download default summary export and confirm every run has
  `included=false` and `data=null`.
- [ ] Real system: enable full observability, download again, confirm every
  campaign run exists and selected sample values match the panels.
- [ ] Real system: test the four filename classes and inspect redaction metadata.
- [ ] Stop. Do not begin Wave 4 until the user replies that Wave 3 passed.

---

# Wave 4 — Parity, Documentation, and Release Gate

### Task 12: Enforce backend panel/export parity

**Repository:** `pdftopng`

**Files:**
- Modify: `tests/test_evaluation_research_end_to_end.py`
- Modify: `tests/test_evaluation_export_redaction.py`
- Modify: `tests/test_evaluation_research_api.py`
- Modify: `docs/evaluation-center.md`
- Modify: `docs/BACKEND.md`

**Interfaces:**
- Consumes completed Schema v2 and canonical services.
- Produces an end-to-end parity fixture proving export sections equal the
  separately requested panel projections.

- [ ] **Step 1: Add an end-to-end parity test**

Create one campaign fixture containing two modes, official RAGAS, accounting,
v9 evidence, routing, ablation, a human rating, an error, and a stage warning.
Call every active campaign projection and export summary. Assert exact serialized
equality for each section:

```python
assert export.sections["question_analysis"].data == question.model_dump(mode="json")
assert export.sections["agent_behavior"].data == behavior.model_dump(mode="json")
assert export.sections["router_analysis"].data == router.model_dump(mode="json")
assert export.sections["ablation"].data == ablation.model_dump(mode="json")
```

Compare overview, human evaluation, errors, and stage warnings the same way.

- [ ] **Step 2: Add a full-run parity assertion**

Request interactive selected-run observability and full export with safe
defaults. Assert canonical fields match after applying the documented export
content options:

```python
assert exported.run_summary == interactive.run_summary
assert exported.accounting_diagnostics == interactive.accounting_diagnostics
assert exported.agentic_v9 == interactive.agentic_v9
assert exported.graph_evidence_items == interactive.graph_evidence_items
```

- [ ] **Step 3: Run the parity tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_end_to_end.py tests/test_evaluation_export_redaction.py tests/test_evaluation_research_api.py -q
```

Expected: PASS. If equality fails, fix the shared projection/composer in this
task and keep the corrective assertion in the same commit.

- [ ] **Step 4: Finalize backend documentation**

Add the real-environment checklist, availability/provenance matrix, Schema v2
example, and no-N+1 guarantee to `docs/evaluation-center.md`; ensure
`docs/BACKEND.md` links to it and removes obsolete export wording.

- [ ] **Step 5: Run backend release suite and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_end_to_end.py tests/test_evaluation_research_api.py tests/test_evaluation_research_analytics.py tests/test_evaluation_analytics_api.py tests/test_evaluation_export_redaction.py tests/test_evaluation_execution_observability.py tests/test_evaluation_observability_repository.py tests/test_evaluation_graph_events.py tests/test_evaluation_human_ratings.py tests/test_evaluation_job_worker.py -q
git add tests/test_evaluation_research_end_to_end.py tests/test_evaluation_export_redaction.py tests/test_evaluation_research_api.py docs/evaluation-center.md docs/BACKEND.md
git commit -m "test(evaluation): enforce panel export parity"
```

### Task 13: Lock frontend panel/export parity and generated documentation

**Repository:** `Multimodal_RAG_System`

**Files:**
- Modify: `src/pages/EvaluationCenter.integration.test.tsx`
- Modify: `src/pages/EvaluationCenter.ui.test.tsx`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `docs/FRONTEND.md`
- Modify: `docs/design-docs/evaluation-center.md`
- Modify: `docs/generated/ui-surface.md`

**Interfaces:**
- Consumes completed backend/OpenAPI contract and all frontend panel changes.
- Produces a single integration scenario that covers direct tab entry,
  campaign switching, observability truth states, durable jobs, and both export
  scopes.

- [ ] **Step 1: Add the final integration scenario**

Mock one complete v9 campaign and one partial legacy campaign. Assert:

```ts
expect(screen.getByText('Measured')).toBeInTheDocument();
expect(screen.getByText('Heuristic')).toBeInTheDocument();
expect(screen.queryByText('utility_best_mode')).not.toBeInTheDocument();
expect(screen.queryByText('Router Selected Mode')).not.toBeInTheDocument();
```

Switch campaigns and assert no v9 values leak into the legacy campaign. Mount
the job panel, then export summary and full scopes and assert the exact request
flags and v2 preview counts.

- [ ] **Step 2: Run the integration/contract tests**

```powershell
npm test -- --run src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.ui.test.tsx src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/EvaluationJobPanel.test.tsx
```

- [ ] **Step 3: Synchronize and verify documentation**

Update service-call lists, eight-tab behavior, durable jobs, capability notices,
and Export v2. Then run:

```powershell
npm run docs:sync
npm run docs:check
npm run docs:links
npm run contract:check
```

If `contract:check` reports only intentional changes from this plan, regenerate
`src/test/fixtures/agenticV9ApiContract.ts` with `npm run contract:pin`, then
rerun `contract:check`.

- [ ] **Step 4: Run the frontend release suite**

```powershell
npm test -- --run src/services/evaluationApi.test.ts src/types/evaluation.contract.test.ts src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.ui.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/RunTraceTab.test.tsx src/components/evaluation/RetrievalEvidenceTab.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/RouterLabTab.test.tsx src/components/evaluation/QuestionAnalysisTab.test.tsx src/components/evaluation/V9EvidenceExplorer.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/components/evaluation/EvaluationJobPanel.test.tsx
npm run lint:ci
npm run build
```

- [ ] **Step 5: Commit**

Review `git status --short`, add only files changed by this task, including the
contract pin only if Step 3 regenerated it:

```powershell
git add src/pages/EvaluationCenter.integration.test.tsx src/pages/EvaluationCenter.ui.test.tsx src/types/evaluation.contract.test.ts src/test/fixtures/agenticV9ApiContract.ts docs/FRONTEND.md docs/design-docs/evaluation-center.md docs/generated/ui-surface.md
git commit -m "test(evaluation-ui): lock evaluation center contract parity"
```

## Wave 4 Checkpoint — Final Mandatory Stop

- [ ] Report both Wave 4 commit hashes and the complete Wave 1-4 commit ledger.
- [ ] Report backend/frontend release suite totals, lint, build, docs, and
  contract-check results.
- [ ] Confirm both repositories contain no task-related uncommitted files.
- [ ] Ask the user to push both repositories.
- [ ] Real system: execute the complete checklist from
  `pdftopng/docs/evaluation-center.md` using a v9 campaign, a legacy/partial
  campaign, a failed/partial-stage run, and a multi-run export campaign.
- [ ] Stop for release acceptance. Do not merge, tag, deploy, or begin unrelated
  cleanup without a new user instruction.

---

## Corrective Task Protocol

If a real-system checkpoint fails:

1. stay in the current Wave;
2. record the observed request, response status, safe response shape, and UI
   symptom without copying secrets or raw provider payloads;
3. add one failing regression test that reproduces the production contract;
4. implement the smallest fix;
5. run the current Wave regression suite;
6. create a separate commit named
   `fix(evaluation): correct wave N <specific behavior>` or
   `fix(evaluation-ui): correct wave N <specific behavior>`; and
7. repeat the same Wave checkpoint.

The next Wave begins only after the user explicitly accepts the corrected
checkpoint.
