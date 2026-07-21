# Agentic RAG v9 Evidence-First Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` task-by-task. Every task uses TDD, focused verification, review, and a scoped commit.
> **Status:** Revised — ready for implementation
> **Behavioral baseline:** `2a4e4e9e1dcdd2ff269695edb007de7e9e3da79c`
> **Runtime:** Python 3.11
> **Design:** `docs/superpowers/specs/2026-07-21-agentic-rag-v9-evidence-first-design.md`

Implement in order with TDD, focused verification, a review checkpoint, and one scoped commit per task. Line numbers are advisory; Task 0 validates paths against the pinned baseline. Backend owns API/OpenAPI; frontend work has a separate plan.

## Invariants

- Setup owns model, thinking, and input/output ceilings; phase policy owns sampling and phase output cap.
- Every provider attempt is atomically reserved before `.ainvoke()`; callbacks only reconcile actual usage.
- Retrieval, Graph, and visual tasks produce evidence, never user-facing answers.
- Positive evidence has its own provenance and scope. Absence is `SlotResolution`, not an EvidencePacket.
- Graph is locator-only until resolved to source chunks/assets.
- Final prompt contains only packed evidence; final generation count is `<=1`.
- Request/campaign execution version is authoritative; environment supplies only the default.
- v9 policy stays under `data_base/agentic_v9`; v8 planner/evaluator/synthesizer remain rollback-safe.
- v1 datasets and historical campaigns are immutable.

## Canonical types

```python
AgenticV9Route = Literal[
    "single_lookup", "bounded_compare", "exact_structured",
    "multi_document_exact", "multi_hop", "graph_relational",
]
GraphPolicy = Literal["never", "locator_fallback", "required_locator"]
ResponseStatus = Literal["complete", "qualified_partial", "insufficient"]

RagRetrievalResult   # generic RAG boundary
TaskRetrievalResult  # one v9 task
V9ExecutionResult    # whole run
```

Task 1 must define every shared type before consumers: `QueryContract`, `RequiredSlot`, `RetrievalTask`, `EvidenceSource`, `EvidenceScope`, `EvidencePacket`, `SlotResolution`, `SufficiencyReport`, `ConflictCandidate`, `FinalClaim`, `FinalAnswerResult`, `RetrievalPolicy`, `AnswerPolicy`, `GeneratedRagAnswer`, `EffectivePhasePolicy`, `BudgetReservation`, `BudgetExceededError`, `ExecutionPolicy`, `V9ExecutionRequest`, `V9ExecutionMetrics`, and `V9ExecutionEvent`. All LLM functions are async.

Evidence and claim support are intentionally different:

```python
EvidenceSupportType = Literal["direct", "calculated", "scope_constraint", "contradictory"]
ClaimSupportType = Literal["direct", "calculated", "comparative_inference", "qualified"]
ScopeMatch = Literal["same", "different", "unknown"]
```

Cross-document comparison is a `FinalClaim` with premise evidence IDs, never a synthetic source packet. Calculations use `Decimal` and persist raw value, normalized value, display precision, and rounding mode. Source locators distinguish `pdf_page_index` from `printed_page_label`; citation format is versioned.

## Runtime contract addendum

### Phase policy and call allocation

| Phase | Temperature | Top-p | Top-k | Output cap | Per-run maximum |
|---|---:|---:|---:|---:|---:|
| `route_plan` | 0.10 | 0.80 | 20 | 384 | 1 |
| `query_rewrite` | 0.10 | 0.80 | 20 | 192 | 1 |
| `retrieval_judge` | 0.10 | 0.70 | 10 | 96 | 1 |
| `graph_route` | 0.10 | 0.70 | 10 | 128 | 1 |
| `graph_traversal` | N/A | N/A | N/A | N/A | 1 |
| `visual_extract` | 0.10 | 0.80 | 20 | 768 | 1 batched call |
| `evidence_extract` | 0.10 | 0.80 | 20 | 768 | 1 batched call |
| `conflict_arbitration` | 0.10 | 0.80 | 20 | 256 | 1 |
| `claim_verifier` | 0.10 | 0.80 | 20 | 384 | 1 batched call |
| `final_answer` | 0.25 | 0.90 | 40 | 1,536 | 1 |

Phase limits are ceilings, not entitlements, and remain subordinate to route call/token/deadline budgets. Admission priority is final response reserve → required route planning → required visual/Graph locator → evidence extraction → retrieval judge/rewrite → conflict arbitration → optional claim verifier. Completely unsupported runs skip final LLM and use a deterministic insufficiency response.

### Transport/runtime separation

`V9ExecutionRequest` is serializable and contains question, user/requested sources, history, Setup snapshot, execution version, and trace identity. Non-serializable dependencies live in `V9RuntimeContext(cancellation_token, event_sink, budget_controller, deadline, clock, llm_invoker)`. The core signature is `await core.execute(request=request, runtime=runtime)`.

### Unified provider boundary

Every query-time model user receives an `LlmInvoker` protocol dependency. v9 route planning, rewrite, retrieval judging, Graph routing, visual extraction, prose curation, conflict arbitration, claim verification, and final answering cannot call `.ainvoke()` directly. Retrieved content is untrusted evidence: it cannot alter policy, authorization, tools, or output schema; the v9 final model has no tool permissions.

## Task 0 — Freeze baseline and build Golden Dataset v2

**Create:** `evaluation/golden/agentic_v9_questions_v2.json`, `evaluation/golden/agentic_v9_baseline_manifest.json`, `tests/test_agentic_v9_golden_dataset.py`.

- [ ] Record backend behavioral commit; frontend URL/branch/commit; model/thinking, corpus/index, prompt and evaluator snapshots; source campaign IDs; artifact SHA-256 values.
- [ ] Add a path-validation test for every file named by this plan.
- [ ] Copy v1. Repair Q14 contradictory ground truths/source scope and Q16 formula splitting.
- [ ] Fill all 16 questions with required/optional atomic facts, claim importance, expected evidence/locators, source docs, and expected route.
- [ ] Assert unique IDs, valid references, non-empty required facts/evidence, stable hash, and six-route coverage.
- [ ] Verify: `python -m pytest tests/test_agentic_v9_golden_dataset.py tests/test_evaluation_test_case_schema.py -q`
- [ ] Commit: `test(agentic-v9): freeze golden dataset and baselines`

The manifest initially records frontend `https://github.com/DeadMark70/Multimodal_RAG_System_Web.git`, branch `master`, commit `1ab15449af756886039614fab6b6cc64781d1d23`; refresh only if execution starts from a newer approved baseline. Document commits are not runtime baselines.

## Task 1 — Define coherent v9 schemas

**Create:** `data_base/agentic_v9/{__init__,schemas}.py`, `tests/test_agentic_v9_schemas.py`. **Modify:** `evaluation/trace_schemas.py`.

- [ ] `EvidencePacket` requires version, task/round/query IDs, statement, support type, source, scope/locator, and optional normalized value/unit/calculation premises; it has no `missing` support type.
- [ ] `SlotResolution.status`: `supported|conflicted|explicitly_unavailable|not_found`.
- [ ] `SufficiencyReport` separates `evidence_complete`, `answerable`, `response_status`, slot groups, and stop reason.
- [ ] Graph defaults by route: never, never, locator fallback, locator fallback, locator fallback, required locator.
- [ ] `V9ExecutionRequest` includes authorized doc IDs, requested source names, bounded history, Setup snapshot, execution version, and trace identity; `V9RuntimeContext` carries cancellation/event/budget/clock dependencies and is never serialized.
- [ ] Verify schema and observability schema tests.
- [ ] Commit: `feat(agentic-v9): define coherent execution contracts`

## Task 2 — Enforce A-type phase policy

**Create:** `data_base/agentic_v9/phase_policy.py`, tests. **Modify:** `core/llm_factory.py`, `evaluation/model_capabilities.py`.

- [ ] Implement the complete runtime-contract phase matrix, including query rewrite, Graph route, and claim verifier.
- [ ] Effective output is `min(setup ceiling, phase cap)`; nested scope cannot change Setup model/thinking.
- [ ] Preserve v8 behavior outside v9 scope.
- [ ] Verify phase, factory override, and capability tests.
- [ ] Commit: `feat(agentic-v9): enforce setup-authoritative phase policy`

### Task 2C — Allocate calls and preflight Setup feasibility

**Create:** `data_base/agentic_v9/budget_feasibility.py`, tests. **Modify:** phase policy and QueryContract validation.

- [ ] Define `max_calls_by_phase` and admission priority exactly as the runtime contract above.
- [ ] Implement `BudgetFeasibilityReport` and validate after contract resolution but before the first provider call.
- [ ] Reservation includes Setup-authoritative reasoning/thought-token reserve, not only visible output.
- [ ] Incompatible thinking/route budget returns `configuration_incompatible`; never disable thinking or inflate budget silently.
- [ ] Verify high-thinking-budget fixtures for single lookup and bounded compare.
- [ ] Commit: `feat(agentic-v9): preflight route budget feasibility`

## Task 3 — Add pre-invoke budgets and accounting reconciliation

**Create:** `data_base/agentic_v9/budget_controller.py`, `budgeted_llm.py`, corresponding tests. **Modify:** usage context/callback/factory.

- [ ] `RunBudgetController` uses `asyncio.Lock`; atomically reserves call plus estimated input and maximum output.
- [ ] Protect one final-call input/output envelope from optional phases.
- [ ] `invoke_budgeted_llm` reserves before provider invocation and reconciles actual usage idempotently; missing usage gets a conservative estimate.
- [ ] Every provider retry, direct/Graph/visual/verifier/final attempt counts. Deterministic parser recovery does not call an LLM.
- [ ] Reservation failure prevents provider invocation. Final failure/unavailability returns deterministic qualified partial with generation count zero.
- [ ] Verify budget, callback, and usage aggregation tests; use existing tests, not nonexistent `test_evaluation_accounting.py`.
- [ ] Commit: `feat(agentic-v9): enforce atomic provider budgets`

### Task 3C — Inject the BudgetedLlmInvoker into every v9 model path

**Modify:** `data_base/query_transformer.py`, v9 CRAG adapters, `graph_rag/generic_mode.py` or its isolated v9 adapter, visual/evidence/conflict/final modules. **Create:** architecture test.

- [ ] Define `LlmInvoker.invoke(phase, purpose, messages)` and inject it through every v9 query-time component.
- [ ] Ensure Multi-Query rewrite, CRAG judge, generic Graph fallback router, visual helpers, and verifier cannot bypass the pre-invoke gate.
- [ ] Add AST/monkeypatch architecture test that fails on direct `.ainvoke()` inside `data_base/agentic_v9` and proves the Graph fallback uses the invoker.
- [ ] Legacy v8 direct paths may remain; v9 adapters must not enter them.
- [ ] Commit: `refactor(agentic-v9): centralize provider invocation`

## Task 4 — Split generic retrieval from generation in six reviewable commits

### 4A Schemas

Create `rag_pipeline_schemas.py` and tests; define `RagRetrievalResult`/`GeneratedRagAnswer`; preserve legacy `RAGResult`. Commit `refactor(rag): define retrieval generation boundaries`.

### 4B Dense/BM25/Multi-Query/RRF

Create `rag_retrieval.py` and tests; extract retrieval without prompt/generation; preserve query origin, ranks, metadata, and expansion usage. Commit `refactor(rag): extract hybrid retrieval pipeline`.

### 4C Filter/reranker

Preserve pre/post ranks, thresholds, rejected candidates, and unavailable scores as N/A. Commit `refactor(rag): isolate filtering and reranking`.

### 4D CRAG

Create `rag_crag.py`; separate deterministic classification from optional LLM judge/rewrite while preserving v8 parity. Commit `refactor(rag): isolate corrective retrieval`.

### 4E Graph locator

Create `rag_graph_locator.py`; return route/fallback/path and resolved sources; test raw graph content cannot support claims. Commit `refactor(rag): isolate graph source locator`.

### 4F Legacy generation parity

Create `rag_generation.py`; legacy wrapper delegates to retrieval + legacy generation. Legacy visual synthesis stays here and is inaccessible to v9. Test naive/advanced/graph, empty/error, and visual parity. Commit `refactor(rag): preserve legacy answer generation parity`.

## Task 5 — Implement token estimator and context packer

**Create:** `data_base/agentic_v9/token_estimator.py`, `context_packer.py`, `tests/test_agentic_v9_context_packer.py`.

`PackedEvidenceContext` contains packets, rendered text, estimated input tokens, dropped packet IDs, and per-slot/per-source token usage.

- [ ] Enforce `min(Setup input ceiling, remaining runtime)` after final-output reserve.
- [ ] Keep best evidence for every answerable required slot, then quality/source diversity; deduplicate chunks/spans.
- [ ] Never split atomic numbers, formulae, theorem conditions, table headers/rows, or citations.
- [ ] Fail closed if mandatory evidence cannot fit; persist dropped IDs and distributions.
- [ ] Commit: `feat(agentic-v9): pack bounded evidence context`

### Task 5B — Estimate the complete final prompt and preserve premise closure

**Modify:** `token_estimator.py`, `context_packer.py`, tests.

- [ ] Return `PromptTokenEstimate` with instruction, question, contract, history, evidence, image, schema, safety margin, and total tokens.
- [ ] Evidence budget subtracts fixed prompt/history/image/safety overhead plus final output and Setup thinking reserve.
- [ ] Selecting a calculated packet or derived-claim premise includes the transitive closure of all direct premise packets.
- [ ] Use conservative language/content-specific estimates for Chinese, English, LaTeX, JSON/table, and images; do not rely on `len(text)//4`.
- [ ] Persist estimated-vs-provider input error and increase safety margin or fail closed after calibrated error thresholds are exceeded.
- [ ] Commit: `feat(agentic-v9): estimate complete prompt budgets`

## Task 6 — Build shared evidence pool

**Create:** `evidence_pool.py` and tests.

- [ ] Identity uses doc/chunk/parent/page/asset; normalized hash is fallback only.
- [ ] Preserve each item’s metadata and retrieval scores; never borrow `source_doc_ids[0]`.
- [ ] Concurrent add is deterministic/idempotent; distinguish retrieved, accepted, packed, used, and rejected sets.
- [ ] Commit: `feat(agentic-v9): add provenance-safe evidence pool`

## Task 7 — Resolve scope, route, and compile tasks

### 7A Source scope resolver

Create `source_scope_resolver.py` and tests. Resolve source names to internal IDs; result is resolved names ∩ user authorization ∩ request `doc_ids`. Ambiguous/unauthorized scope fails closed. Commit `feat(agentic-v9): resolve authorized document scopes`.

### 7B QueryContract router

Create `route_planner.py`, prompt JSON, tests. Deterministic-first routing; one budgeted planner call only for ambiguity. Emit slots, graph policy, budgets, locators, and resolved scope. Test six routes including Q1/Q2 and a fixed graph path. Commit `feat(agentic-v9): plan slot-driven retrieval contracts`.

### 7C Retrieval task compiler

Create `retrieval_tasks.py` and tests. Q9 bounded A/B, Q15 asset locator, Q16 source groups, Q1/Q2 dependencies. Every task has target slots, scope, locator, round/query ID, graph/visual policy; none has an answer field. Commit `feat(agentic-v9): compile bounded retrieval tasks`.

## Task 8 — Make visual processing evidence-only

**Create:** `asset_locator.py`, `visual_evidence_extractor.py`, tests.

- [ ] Locate page/figure/table/formula before visual invocation.
- [ ] Visual model receives target slot/question fragment/asset/source and returns EvidencePacket JSON only.
- [ ] Use budget, phase policy, semaphore, timeout, and cancellation; never call legacy visual answer synthesis.
- [ ] Commit: `feat(agentic-v9): extract visual evidence only`

## Task 9 — Extract evidence after retrieval stabilizes

**Create:** `evidence_extractor.py`, prompt JSON, tests.

- [ ] Deterministically extract exact values, units, formulae, theorem ranges, table rows, enumerations, locators, and calculation premises each round.
- [ ] Complete repair first, then make at most one batched prose-curator call for remaining unresolved prose slots.
- [ ] Drop invalid packet output without an LLM repair call.
- [ ] Commit: `feat(agentic-v9): curate final evidence packets`

### Task 9B — Validate source-bound prose packets

**Create:** `data_base/agentic_v9/evidence_validator.py`, tests. **Modify:** evidence schema/extractor.

- [ ] Persist `source_span_hash` and `validation_status=deterministic_valid|quote_bound|invalid`.
- [ ] Require verbatim span membership; reject new numbers and unsourced model/dataset/metric entities.
- [ ] Reject added negation, comparison, first/SOTA, cause, outperforms, safe/robust, best/highest language unless present in the source span.
- [ ] Curator cannot rewrite source scope; prose packets remain extractive or minimally normalized.
- [ ] High-risk abstractions become FinalClaims with premise IDs, not EvidencePackets.
- [ ] Commit: `feat(agentic-v9): validate source-bound evidence`

## Task 10 — Sufficiency and bounded repair

### 10A Sufficiency

Create `sufficiency_gate.py` and tests. Explicitly unavailable may stop repair but cannot set evidence complete. Persist supported/unavailable/missing/conflicted slots and response status. Commit `feat(agentic-v9): evaluate slot sufficiency`.

### 10B Repair/selective CRAG

Create `repair.py`, `selective_crag.py`, tests. Queries derive only from missing slot/entity/locator, preserve scope, use deterministic CRAG first, obey route caps, and protect final budget. Commit `feat(agentic-v9): add bounded evidence repair`.

## Task 11 — Conflict and isolated final answer

### 11A Scope-aware conflict

Create `conflict_gate.py` and tests. Conflict requires same slot/metric/dataset/split/model variant/protocol/prompt setting and incompatible values. Scope matching is `same|different|unknown`; unknown becomes `scope_ambiguous` and may require arbitration or qualification, never silently “no conflict.” Only persisted unresolved candidates may invoke one arbitration call. Commit `feat(agentic-v9): gate scope-aware conflicts`.

### 11B Final answer/verifier/citations

Create `final_answer.py`, `claim_verifier.py`, `citation_renderer.py`, tests. Final input is question + contract + packed packets + slot resolutions + optional arbitration. Claims list evidence IDs; deterministic verification handles exact facts; at most one batched verifier handles unresolved high-risk prose. Remove/qualify failed claims without regeneration. Do not add v9 policy to v8 `agents/*`. Commit `feat(agentic-v9): generate one verified final answer`.

## Task 12 — Bounded execution core

### 12A State machine

Create `execution_core.py` and tests. Sequence: scope/contract → retrieval → deterministic candidates → sufficiency/repair → final prose batch → final sufficiency/conflict → pack → final/deterministic partial. Assert subtask answers `0`, prose curator `<=1`, arbitration `<=1`, final generation `<=1`. Commit `feat(agentic-v9): orchestrate evidence-first execution`.

### 12B Runtime bounds

Create `execution_policy.py` and tests. Initial limits: retrieval concurrency 3, LLM 2, visual 1; timeouts route/judge 2s, extract 8s, final 15s. Test provider-attempt retries, TaskGroup cancellation, campaign cancel, SSE disconnect, and reservation reconciliation. Commit `feat(agentic-v9): enforce runtime bounds and cancellation`.

### Task 12C — Propagate a whole-run deadline

**Modify:** execution policy/core/runtime context; add deadline tests.

- [ ] Add `total_deadline_s=24.0` plus retrieval, rerank, Graph, visual, and existing phase timeouts.
- [ ] Each operation uses `min(phase_timeout, deadline.remaining_seconds())`.
- [ ] When time is insufficient for the final reserve, skip optional repair/arbitration/verifier and enter final or deterministic partial.
- [ ] `insufficient` with zero supported slots must never invoke final LLM.
- [ ] Commit: `feat(agentic-v9): enforce whole-run deadline`

## Task 13 — Persistence and API contract

### 13A Storage/migrations

Modify `evaluation/db.py`, `trace_schemas.py`, `observability_storage.py`; add repository/migration tests.

| Artifact | Authoritative storage |
|---|---|
| QueryContract | versioned trace payload + agent detail |
| Sufficiency/reservations/repairs | versioned trace payload |
| Actual provider usage | existing `evaluation_usage_events` only |
| EvidencePacket | new `evaluation_evidence_packets` |
| SlotResolution | new `evaluation_slot_resolutions` |
| Final claims | existing `evaluation_claims` |
| Packed/dropped evidence | existing `evaluation_context_packs` |

Add indexes/FKs/schema version/payload-size guard and historical empty defaults. Commit `feat(evaluation): persist agentic v9 evidence state`.

### Task 13C — Add attempt idempotency, authorization, and redaction

**Modify:** DB/storage/router/export/security tests.

- [ ] New rows include attempt, run, campaign, condition, schema version, and created time; unique keys are `(attempt_id,evidence_id)` and `(attempt_id,slot_id,resolution_stage)`.
- [ ] Materialize contract/packets/slots/claims/completion trace atomically for an attempt; cancelled attempts retain trace but never complete evidence state.
- [ ] Official benchmark uses latest official successful attempt; operational diagnostics include every retry attempt. Keep these token totals separate.
- [ ] Enforce campaign ownership on new tables; unauthorized source resolution cannot fall back to all documents or leak names into trace.
- [ ] Default API returns bounded plain-text excerpts, not raw HTML/Markdown/full prompts; authorized export may return full spans. Apply secret redaction and payload size/version guards.
- [ ] Add cross-user denial, prompt-injection-as-evidence, retry-idempotency, and cancelled-materialization tests.
- [ ] Commit: `feat(evaluation): secure idempotent v9 evidence persistence`

### 13B Analytics/API/OpenAPI

Modify `analytics.py`, `router.py`, `openapi.json`; add API tests. Expose contract, slots, packets, dropped context, graph outcome, budgets, repairs, conflicts, response/cancel status, generation count. Unknown remains N/A; no money fields. Commit `feat(evaluation): expose v9 evidence observability api`.

## Task 14 — Evaluation adapter and RAGAS integrity

Modify campaign schemas/engine/worker, agentic evaluation service, and rag modes; add tests.

- [ ] Persist `agentic_execution_version: Literal["v8","v9"]`; env is default only.
- [ ] Formal identities are `naive/naive-baseline`, `agentic/agentic-v8/v8`, and `agentic/agentic-v9/v9`; shadow alone uses `agentic-v9-shadow`.
- [ ] Adapter maps campaign ↔ core without containing policy.
- [ ] Result separates `used_evidence_documents` from all retrieved; only used/cited documents populate `RAGResult.documents` and RAGAS contexts.
- [ ] Test context-to-claim evidence mapping, unpacked exclusion, and deduplication.
- [ ] `/runs` returns condition ID, execution profile/version, and response status so UI cannot merge v8, v9, and shadow.
- [ ] Commit: `feat(evaluation): adapt campaigns to agentic v9`

## Task 15 — Chat SSE adapter

Modify `sse_events.py`, `agentic_chat_service.py`, `router.py`; add tests.

- [ ] Keep existing `plan_ready`, `plan_confirmed`, `task_start`, `task_done`, `synthesis_start`, `complete`; never expect nonexistent `plan_complete`.
- [ ] Make `TaskDoneData.answer: str|None=None`; add optional evidence count, target slots, sources. v7/v8 still send answers.
- [ ] Keep current maximum ten history messages plus a history-token cap. History aids query resolution but cannot be evidence/cited.
- [ ] Propagate client disconnect cancellation.
- [ ] Commit: `feat(chat): stream agentic v9 evidence progress`

## Task 16 — Execute separate frontend plan

Backend stops after verified API/OpenAPI. UI work follows `docs/superpowers/plans/2026-07-21-agentic-rag-v9-frontend-observability.md`; never mix repository commits.

## Task 17 — Metrics, benchmark, promotion

### 17A Derived metrics

Add required-slot coverage, important-claim unsupported rate, provenance failure rate, pack efficiency, Graph locator success/fallback, generation count, and Agentic/Naive token ratio. Fail closed on partial accounting, missing golden data, incompatible snapshots, or missing used-evidence mapping. Add deterministic paired bootstrap. Commit `feat(evaluation): derive v9 evidence release metrics`.

### 17B Gates

- Smoke only: Q9/Q15/Q16 × naive/v8/v9 × 1.
- Formal: all 16 × naive/v8/v9 conditions × 8 paired repeats with identical golden hash, corpus/index, prompt, model/thinking, phase, evaluator, and code snapshots.
- Mandatory routes: Q10/fixed single lookup, Q9, Q15, Q16, Q1, Q2, fixed graph path.
- Report paired mean delta, 95% paired-bootstrap CI, category/per-question regressions, tokens/calls/latency, evidence metrics, accounting completeness.
- Engineering correctness delta vs v8 `>=0`; default statistical lower bound `>=-0.01` (strict `>=0` requires explicit approval).
- First gates: token ratio `<=4`, relevancy `>=0.70`, P95 `<25s`, provenance failures `0`, subtask answers `0`, final generation `<=1`, complete accounting/phase attribution.
- Promote default only if all gates pass; rollback remains config-only.

### Task 17C — Enforce scientific condition identity and clustered bootstrap

- [ ] Pair by `(question_id, repeat_number)` and identify arms by `(mode, condition_id, execution_profile)`; never merge formal and shadow v9.
- [ ] Aggregate eight repeats per question, then bootstrap 16 question clusters; never treat 128 executions as independent questions.
- [ ] Randomize arm order within each question/repeat block and blind the RAGAS judge to labels.
- [ ] Official token ratio is `sum(v9 official runtime tokens)/sum(naive official runtime tokens)`, not mean per-run ratios.
- [ ] P95 uses successful accounting-complete official v9 runs, but excessive failure/timeout fails the gate rather than being excluded silently.
- [ ] Add a non-blocking ablation arm for v8 + A-type phase policy to separate policy from architecture effects.
- [ ] Commit: `feat(evaluation): enforce clustered v9 benchmark statistics`

## Review checkpoints

- **R0 Task 1:** types, scope, absence/provenance, Graph policy.
- **R1 Task 3:** Setup authority, atomic hard gate, retry/final reserve, accounting.
- **R2 Task 4F:** legacy parity and v9/legacy visual separation.
- **R3 Task 7C:** context ceiling, evidence pool, authorized scope, six routes.
- **R4 Task 10B:** evidence extraction, truthful sufficiency, bounded repair.
- **R5 Task 12B:** isolated final, at-most-one generation, concurrency/timeout/cancel.
- **R6 Task 15:** persistence/API/Evaluation/Chat compatibility and RAGAS contexts.
- **Release Task 17B:** independent benchmark comparability review.

## Definition of done

All provider attempts are pre-budgeted; Setup input ceiling is demonstrably enforced; every positive packet has provenance/scope; Graph/visual/retrieval create no answers; final generation never exceeds one; v8 rollback is intact; cancellation stops work; v9 evidence is queryable; RAGAS sees only cited packed evidence; Golden v2 and the eight-repeat paired release gate pass.

## Non-goals

No model/embedding/retriever/Graph DB/Python replacement, multi-model routing, pricing UI, unbounded agent loop, whole-answer regeneration, or frontend implementation in this repository.
