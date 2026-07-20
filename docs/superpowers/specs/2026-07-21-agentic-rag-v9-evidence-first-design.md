# Agentic RAG v9 Evidence-First Design

**Status:** Approved design baseline

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

`QueryContract` replaces complexity tier as the execution authority. `strategy_tier` remains trace-only compatibility metadata.

Required route values:

- `single_lookup`
- `bounded_compare`
- `exact_structured`
- `multi_document_exact`
- `multi_hop`
- `graph_relational`

The contract records intent, required answer slots, entities, locator hints, graph/visual requirements, repair limit, LLM-call limit, and runtime token budget. Deterministic routing handles clear lookup, locator, numeric, formula, explicit comparison, exclusion, and already-decomposed questions. One RoutePlan LLM call is permitted only for ambiguous slot structure, hidden dependencies, unknown document scope, uncertain graph need, or mixed comparison/numeric/visual/verdict questions.

## Retrieval and Generation Boundaries

The monolithic `rag_answer_question()` is split behind two public interfaces:

```python
async def retrieve_rag_evidence(
    *,
    question: str,
    user_id: str,
    doc_ids: list[str] | None,
    retrieval_policy: RetrievalPolicy,
) -> RetrievalResult: ...

async def generate_answer_from_evidence(
    *,
    question: str,
    evidence_packets: list[EvidencePacket],
    answer_policy: AnswerPolicy,
) -> FinalAnswerResult: ...
```

The existing `rag_answer_question()` remains a compatibility wrapper for Naive, Advanced, Graph, and v8 callers. v9 calls retrieval for all retrieval tasks and calls answer generation only after sufficiency and conflict handling.

## Evidence Pool and Provenance

Run-level deduplication occurs before generation. The stable identity is composed from `doc_id`, `chunk_id`, `parent_id`, `page`, and `asset_id`; a SHA-256 hash of normalized text is fallback-only.

Every evidence item derives source identity from its own `Document.metadata`. The current behavior that assigns every context to `source_doc_ids[0]` is removed. Missing canonical `doc_id` is a traceable provenance failure and cannot silently borrow another document's identity.

## Evidence Packets

Evidence packets are the v9 fact state. Required `support_type` values are:

- `direct`
- `calculated`
- `comparative_inference`
- `scope_constraint`
- `contradictory`
- `missing`

Numeric values, formulae, locators, theorem ranges, table rows, and explicit enumerations use deterministic extraction first. General prose is curated in one batched evidence-extraction LLM call. Cross-document comparative inference is deferred to final generation and must list its premise evidence IDs.

## Sufficiency and Repair

Sufficiency is evaluated across the collection against required slots. Answer length and loose lexical overlap are not completion criteria. Each required slot must be supported, have traceable calculation/inference premises, or be explicitly marked insufficient.

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

Graph is independent of complexity. Explicit A-vs-B comparison, named model comparison, Figure/Table/Appendix/Theorem lookup, and multi-document exact extraction default to Graph off. Graph is enabled only for unknown-scope relations, lineage, relation/path queries, graph-to-source localization, or vector/BM25 failure to discover cross-document links.

## Conflict and Final Answer

Conflict candidates require the same slot, metric, dataset, and experimental scope with incompatible values or verdicts. Dataset, protocol, model-size, prompt-setting, year, or data-volume differences are scope differences rather than conflicts. Only unresolved conflict candidates invoke one arbitration call.

Final generation consumes the original question, Query Contract, supported evidence packets, calculated facts, optional arbitration result, missing slots, and output schema. It does not consume full subtask answers, planner reasoning, duplicate chunks, or rejected retrieval noise.

The final model returns structured claims with `evidence_ids`. A deterministic renderer adds citations. High-risk claims are verified selectively. Deterministic verification covers numbers, calculations, formula normalization, source identity, and comparative premises. One batched verifier LLM call is allowed only for unresolved abstractive causal claims. Failed claims are removed or qualified; the whole answer is not regenerated.

## Route Budgets

| Route | Maximum LLM calls | Runtime token budget | Graph | Repair |
|---|---:|---:|---:|---:|
| `single_lookup` | 2 | 3,500 | 0 | 0 |
| `bounded_compare` | 4 | 6,500 | 0 | 1 |
| `exact_structured` | 4 | 7,500 | 0 | 1 |
| `multi_document_exact` | 5 | 10,000 | 0 | 2 |
| `multi_hop` | 5 | 10,000 | conditional | 1 |
| `graph_relational` | 5 | 12,000 | at most one traversal | 1 |

## Phase Policies

All v9 runtime phases continue to use the Setup-selected model.

| Phase | Temperature | Top-p | Top-k | Output cap |
|---|---:|---:|---:|---:|
| `route_plan` | 0.10 | 0.80 | 20 | 384 |
| `retrieval_judge` | 0.10 | 0.70 | 10 | 96 |
| `evidence_extract` | 0.10 | 0.80 | 20 | 768 |
| `conflict_arbitration` | 0.10 | 0.80 | 20 | 256 |
| `final_answer` | 0.25 | 0.90 | 40 | 1,536 |
| `visual_extract` | 0.10 | 0.80 | 20 | 768 |

## Deployment

- `AGENTIC_EXECUTION_VERSION=v8|v9` selects the authoritative execution core.
- Default remains `v8` until shadow comparison passes.
- Evaluation and streaming chat become thin adapters around the shared v9 core.
- Persist `execution_profile`, Query Contract, effective phase configs, budget ledger, stop reason, evidence counts, repairs, Graph decision/fallback, conflict decision, and final-generation count.
- Historical campaigns remain readable and retain their original execution profile.

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

Q9, Q15, and Q16 are mandatory representative cases for bounded comparison, structured extraction, and multi-document exact retrieval.
