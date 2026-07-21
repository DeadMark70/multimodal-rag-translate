# Agentic RAG v9 Evidence-First Design

**Status:** Approved and frozen

**Date:** 2026-07-21

## Objective

Upgrade the current Agentic v8 execution flow to a budgeted, evidence-first v9 flow. Retrieval subtasks must stop producing full answers. They produce provenance-preserving evidence packets in a shared run-level evidence pool, and the run performs at most one formal final-answer generation.

## Fixed Decisions

- Evaluation Setup remains authoritative for `model_name` and thinking controls.
- Agentic phase policy controls `temperature`, `top_p`, and `top_k`.
- Setup `max_output_tokens` is a ceiling; the effective phase value is `min(setup ceiling, phase cap)`.
- Setup `max_input_tokens` is enforced by the v9 context packer and run budget. The model context window is not treated as the usable run budget.
- v9 does not change the model, embedding model, FAISS, BM25, or reranker and does not introduce multi-model routing.
- Evaluation v8 remains available behind an execution-version flag until v9 passes shadow and benchmark gates.
- Graph remains locator-only evidence. Raw graph text cannot directly support final claims.
- Backend behavioral baseline is commit `2a4e4e9e1dcdd2ff269695edb007de7e9e3da79c`; later document-only commits do not change that runtime baseline.
- Backend runtime remains Python 3.11 for this upgrade. Python-version migration is out of scope.
- Execution version is persisted per request/campaign. `AGENTIC_EXECUTION_VERSION` supplies only the default.
- Evaluation contexts contain only evidence actually packed for and cited by the final answer.

## Core Invariant

```text
retrieval task -> RetrievalResult -> Evidence Pool -> Evidence Packets
                                                       |
                                                       v
                                              one final generation
```

The following v8 pattern is forbidden in v9:

```text
subtask retrieval -> full subtask answer -> fact extraction -> arbitration -> synthesis
```

## Query Contract

`QueryContract` replaces complexity tier as the execution authority. `strategy_tier` remains trace-only compatibility metadata. Graph behavior is represented by `GraphPolicy = Literal["never", "locator_fallback", "required_locator"]`, not a boolean.

Required route values:

- `single_lookup`
- `bounded_compare`
- `exact_structured`
- `multi_document_exact`
- `multi_hop`
- `graph_relational`

The contract records intent, required answer slots, entities, locator hints, graph/visual requirements, repair limit, LLM-call limit, runtime token budget, and resolved authorized document scope. Requests contain only requested IDs/names; authenticated identity comes from the adapter, and only the resolver may produce authorized IDs. Deterministic routing handles clear lookup, locator, numeric, formula, explicit comparison, exclusion, and already-decomposed questions. One RoutePlan LLM call is permitted only for ambiguous slot structure, hidden dependencies, unknown document scope, uncertain graph need, or mixed comparison/numeric/visual/verdict questions. Source names are resolved to internal document IDs and intersected with both user authorization and request-supplied `doc_ids`; routing can never expand document authorization.

## Retrieval and Generation Boundaries

The monolithic `rag_answer_question()` is split behind two public interfaces:

```python
async def retrieve_rag_evidence(
    *,
    question: str,
    user_id: str,
    doc_ids: list[str] | None,
    retrieval_policy: RetrievalPolicy,
) -> RagRetrievalResult: ...

async def generate_answer_from_evidence(
    *,
    question: str,
    evidence_packets: list[EvidencePacket],
    answer_policy: AnswerPolicy,
) -> FinalAnswerResult: ...
```

The existing `rag_answer_question()` remains a compatibility wrapper for Naive, Advanced, Graph, and v8 callers. Generic retrieval returns `RagRetrievalResult`; an individual v9 task returns `TaskRetrievalResult`; the run returns `V9ExecutionResult`. v9 calls retrieval for all retrieval tasks and calls answer generation only after sufficiency and conflict handling. v9 policy code remains in `data_base.agentic_v9`; v8 planner, evaluator, and synthesizer receive only backward-compatible utility fixes.

Visual routes are evidence-only:

```text
visual retrieval -> page/figure/table locator -> visual extraction -> EvidencePacket
```

They never produce an initial answer or run the legacy visual answer-synthesis loop.

## Evidence Pool and Provenance

Run-level deduplication occurs before generation. The stable identity is composed from `doc_id`, `chunk_id`, `parent_id`, `page`, and `asset_id`; a SHA-256 hash of normalized text is fallback-only.

Every evidence item derives source identity from its own `Document.metadata`. The current behavior that assigns every context to `source_doc_ids[0]` is removed. Missing canonical `doc_id` is a traceable provenance failure and cannot silently borrow another document's identity.

## Evidence Packets and Slot Resolution

Evidence packets are the v9 positive fact state. Required `support_type` values are:

- `direct`
- `calculated`
- `scope_constraint`
- `contradictory`

Every packet includes a versioned `EvidenceScope` covering applicable dataset, split, metric, model/variant, training protocol, prompt setting, noise level, and publication year, plus extractor/prompt versions, task/round/query IDs, normalized value/unit, calculation operation and premise IDs where applicable.

Absence is modeled separately as `SlotResolution(status="explicitly_unavailable"|"not_found")`; it is not a source-less EvidencePacket. Provenance missing rate counts only positive evidence packets.

Numeric values, formulae, locators, theorem ranges, table rows, and explicit enumerations use deterministic extraction first. Retrieval and repair finish before one batched prose-curation call processes the remaining unresolved prose slots. `FinalClaim.support_type` is `direct`, `calculated`, `comparative_inference`, or `qualified`. Cross-document comparative inference is therefore a FinalClaim, never an EvidencePacket, and must list its premise evidence IDs.

## Context Packing and Runtime Budget

The final input is built by a dedicated context packer. It enforces the minimum of the Setup input ceiling and remaining runtime-token budget, reserves final output capacity, preserves at least the best evidence for every answerable required slot, maintains source diversity, deduplicates chunks, and never truncates an atomic number, formula, table header/row, or theorem condition. Dropped packets and per-slot/per-source token usage are observable.

All provider invocations pass through an async pre-invoke budget wrapper. Reservation is atomic, includes estimated input plus maximum output, and reserves one final-call envelope. Every provider attempt, including retry, Graph, visual, verifier, and direct calls, consumes call/token budget. Callback accounting records actual usage and reconciles idempotently but is not the hard gate. Parser failure uses deterministic extraction/fallback and never makes an unbudgeted repair call. If final-call budget is unavailable, a deterministic qualified-partial renderer is used and final-generation count is zero.

## Sufficiency and Repair

Sufficiency is evaluated across the collection against required slots. Answer length and loose lexical overlap are not completion criteria. `SufficiencyReport` separates `evidence_complete`, `answerable`, and `response_status` (`complete`, `qualified_partial`, `insufficient`). A slot that is explicitly unavailable can stop repair without falsely marking evidence complete.

Repair queries are generated only from missing slot, missing entity, and locator. Repair limits are:

| Route | Maximum repair rounds |
|---|---:|
| `single_lookup` | 0 |
| `bounded_compare` | 1 |
| `exact_structured` | 1 |
| `multi_document_exact` | 2 |
| `multi_hop` | 1 |
| `graph_relational` | 1 |

Budget exhaustion ends repair and produces a qualified partial answer. It does not start an unbounded agent loop.

## Selective CRAG and Graph

CRAG uses deterministic pass/correct decisions first. The LLM retrieval grader is reserved for semantically relevant but ambiguous, partial, close-scored, conflicting, or scope-ambiguous evidence. Corrective rewrite produces one targeted query by default and at most two.

Graph is independent of complexity. `single_lookup` and `bounded_compare` default to `never`; exact structured, multi-document exact, and multi-hop routes default to `locator_fallback`; graph-relational uses `required_locator`. Graph is enabled for unknown-scope relations, lineage, relation/path queries, graph-to-source localization, or vector/BM25 failure to discover cross-document links. A graph hit must resolve to source chunks/assets before it can support a final claim.

## Conflict and Final Answer

Conflict candidates require the same slot, metric, dataset, and experimental scope with incompatible values or verdicts. Dataset, protocol, model-size, prompt-setting, year, or data-volume differences are scope differences rather than conflicts. Only unresolved conflict candidates invoke one arbitration call.

Final generation consumes the original question, Query Contract, supported evidence packets, calculated facts, optional arbitration result, missing slots, and output schema. It does not consume full subtask answers, planner reasoning, duplicate chunks, or rejected retrieval noise.

The final model returns structured claims with `evidence_ids`. A deterministic renderer adds citations. High-risk claims are verified selectively. Deterministic verification covers numbers, calculations, formula normalization, source identity, and comparative premises. One batched verifier LLM call is allowed only for unresolved abstractive causal claims. Failed claims are removed or qualified; the whole answer is not regenerated.

## Route Budgets

| Route | Maximum LLM calls | Runtime token budget | Graph policy | Repair |
|---|---:|---:|---|---:|
| `single_lookup` | 2 | 3,500 | `never` | 0 |
| `bounded_compare` | 4 | 6,500 | `never` | 1 |
| `exact_structured` | 4 | 7,500 | `locator_fallback` | 1 |
| `multi_document_exact` | 5 | 10,000 | `locator_fallback` | 2 |
| `multi_hop` | 5 | 10,000 | `locator_fallback` | 1 |
| `graph_relational` | 5 | 12,000 | `required_locator`, at most one traversal | 1 |

## Phase Policies

All v9 runtime phases continue to use the Setup-selected model.

| Phase | Temperature | Top-p | Top-k | Output cap |
|---|---:|---:|---:|---:|
| `route_plan` | 0.10 | 0.80 | 20 | 384 |
| `query_rewrite` | 0.10 | 0.80 | 20 | 192 |
| `retrieval_judge` | 0.10 | 0.70 | 10 | 96 |
| `graph_route` | 0.10 | 0.70 | 10 | 128 |
| `evidence_extract` | 0.10 | 0.80 | 20 | 768 |
| `conflict_arbitration` | 0.10 | 0.80 | 20 | 256 |
| `claim_verifier` | 0.10 | 0.80 | 20 | 384 |
| `final_answer` | 0.25 | 0.90 | 40 | 1,536 |
| `visual_extract` | 0.10 | 0.80 | 20 | 768 |

## Deployment

- `CampaignConfig` and `V9ExecutionRequest` persist `agentic_execution_version: Literal["v8", "v9"]`; `AGENTIC_EXECUTION_VERSION` supplies only the default.
- Default remains `v8` until shadow comparison passes.
- Shadow results use `condition_id="agentic-v9-shadow"` so they cannot collide with authoritative campaign rows.
- Shadow explicitly persists an `operational` or `research` evaluation policy. Operational shadow cannot affect authoritative answers/gates; research shadow is independently evaluated and required for its declared comparison.
- Evaluation and streaming chat become thin adapters around the shared v9 core.
- Persist Query Contract and SufficiencyReport in versioned trace payloads, actual provider usage only in `evaluation_usage_events`, EvidencePackets and SlotResolutions in normalized v9 tables, final claims in `evaluation_claims`, and the final pack in `evaluation_context_packs`.
- `RAGResult.documents` contains only `used_evidence_documents`; all retrieved/rejected/unpacked evidence remains observability-only.
- Attempt-scoped evidence state is idempotent and authorized through campaign ownership. Formal arm identity is `(mode, condition_id, execution_profile)`; shadow identity is distinct from formal v9.
- Run detail exposes v9 data only through a versioned nested `agentic_v9` envelope. OpenAPI is the transport source of truth.
- Historical campaigns remain readable and retain their original execution profile.

## Concurrency, Cancellation, and Chat History

`ExecutionPolicy` explicitly bounds retrieval, LLM, and visual concurrency, gives every phase a timeout, and establishes a 24-second whole-run deadline at attempt/chat start before source resolution. The budget controller and evidence pool are concurrency-safe. Campaign cancellation, task cancellation, timeout, and SSE disconnect propagate through the execution core and reconcile reservations before exit.

`V9ExecutionRequest.history` preserves at most the existing ten-message chat limit subject to a separate history token cap. History may help conversational query resolution but cannot be stored as academic evidence or cited by a final claim.

## Evaluation Protocol

Golden Dataset v2 is immutable and separately hashed; v1 is never overwritten. It repairs Q14 reference/source contradictions and Q16 formula tokenization, and supplies required/optional atomic facts, claim importance, expected evidence, and expected route for all 16 formal questions. A separate route-regression fixture supplies deterministic/synthetic cases, including graph-relational coverage, and does not participate in formal quality averages.

Q9/Q15/Q16 × one repeat is a smoke gate only. Promotion uses 16 questions × `naive`, v8, and v9 × eight paired repeats with an identical corpus, index, prompt, model, evaluator, and dataset snapshot. The bootstrap unit is the question cluster: aggregate repeats per question, then resample 16 questions. Report mean paired delta, 95% paired-bootstrap confidence interval, category deltas, and per-question regressions. Engineering correctness delta must be non-negative and the default statistical safety bound is lower CI ≥ -0.01.

Document authority is fixed: this design is the semantic source of truth, the backend plan is the implementation source of truth, and generated OpenAPI is the transport source of truth.

## Acceptance Criteria

First release gate:

- Agentic/Naive runtime-token ratio is at most 4.
- Correctness does not fall below the current Agentic baseline.
- Relevancy is at least 0.70.
- P95 latency is below 25 seconds.
- Low-complexity routes make zero planner calls, at most one final-generation call, and zero Graph calls.
- Every route produces zero full subtask answers and at most one final answer.
- Evidence provenance missing rate is zero.
- Conflict arbitration occurs only for a persisted conflict candidate.
- Every repair query references a missing slot.
- Accounting and phase attribution are complete.

Second target:

- Faithfulness is at least 0.77.
- Correctness is at least 0.49.
- Relevancy is at least 0.72.
- Important-claim unsupported rate decreases.
- Agentic/Naive runtime-token ratio is at most 3.

Mandatory route regressions cover all six routes: Q10 (or a fixed single lookup), Q9, Q15, Q16, Q1/Q2, and a fixed graph relation/path case. Q1 and Q2 are always included because of their known faithfulness risk.
