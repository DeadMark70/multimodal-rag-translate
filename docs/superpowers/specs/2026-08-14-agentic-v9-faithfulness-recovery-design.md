# Agentic RAG v9 Faithfulness Recovery Design

**Status:** Approved

**Date:** 2026-08-14

## Objective

Restore the evidence-first behavior already required by the Agentic RAG v9 design. The evaluation campaign runtime must qualify retrieved evidence before treating a required slot as supported, and final synthesis must consume the Query Contract and emit source-bound structured claims instead of wrapping an unconstrained prose answer in one synthetic claim.

This is a focused runtime correction. It does not change the model, retriever, reranker, route budgets, Native evaluation behavior, or the current general chat Agentic implementation.

## Motivation and Evidence

Campaign export `75dba5d0-1b06-4a0f-a588-e1761e2e8105-summary-custom-v2.json` contains 96 completed Agentic runs and 96 completed Naive runs. Agentic improved answer relevancy but lost 11.81 percentage points of faithfulness. Correctness was close to Naive and the observed correctness difference was not the primary regression.

The loss grows with the number of required answer slots:

| Required slots | Questions | Mean Agentic minus Naive faithfulness |
|---:|---:|---:|
| 1 | 9 | +0.1 percentage points |
| 2 | 16 | -13.6 percentage points |
| 3 | 6 | -26.7 percentage points |

The persisted v9 observations also show that all 96 Agentic runs produced one final claim, all were marked complete, and none performed repair. That behavior is inconsistent with multi-slot evidence-grounded synthesis.

## Root Cause

Three runtime shortcuts bypass the intended v9 contract:

1. Retrieved chunks are promoted to `deterministic_valid` evidence before their contents have been shown to support the assigned slot. Sufficiency therefore measures packet presence rather than semantic support.
2. The campaign adapter supplies a no-op prose-curation stage, so the existing source-bound `EvidenceExtractor` is not part of the executed evaluation path.
3. Final generation receives only the question and rendered evidence text. Its free-text response is wrapped as one `direct` claim referencing every packed evidence ID and is returned as `complete`, regardless of required-slot coverage.

## Scope

### In scope

- Evaluation Agentic v9 execution through `AgenticV9CampaignRuntime` and `V9ExecutionCore`.
- Shared evidence qualification and final-answer behavior under `data_base.agentic_v9`.
- Sufficiency, repair, claim validation, response-status derivation, observability, and focused documentation/tests needed by this correction.

### Out of scope

- Naive, Advanced, Graph, and v8 evaluation behavior.
- The current general `/rag/agentic/stream` implementation, which does not yet execute `V9ExecutionCore`.
- Model, embedding, FAISS, BM25, reranker, and dataset changes.
- Automatic enforcement or new instrumentation for the Agentic/Naive token ratio.
- Tuning RAGAS prompts or evaluators to improve the reported score.

Shared v9 components must remain reusable so a later chat migration can adopt the same semantics without copying campaign-specific code.

## Fixed Decisions

- A raw retrieved candidate is not positive evidence merely because it carries a `slot_id`.
- Sufficiency consumes qualified evidence packets, never the unqualified retrieval pool.
- Existing `EvidenceExtractor`, evidence validators, `FinalAnswerRenderer`, and `ClaimVerifier` are extended or wired in place; no parallel qualification framework is introduced.
- Qualification is batched across the unresolved slots and eligible candidates in a retrieval round. No per-chunk or per-slot provider calls are introduced.
- Existing route `max_llm_calls`, runtime-token budgets, phase policies, deadline, and `RunBudgetController` remain authoritative. Route budgets are not automatically increased.
- The final-answer reserve remains protected. If a qualification, repair, conflict, verifier, or final call cannot be admitted, the run fails closed to a qualified partial or insufficient response.
- The existing v9 target of Agentic/Naive runtime-token ratio at most 3.0 remains a manual campaign acceptance measure. This work does not add a release gate for it.
- Complete answers remain natural user-facing prose. Only partial answers explicitly separate confirmed findings from unresolved requirements.

## Corrected Execution Flow

```text
resolve authorized scope
  -> plan Query Contract
  -> retrieve candidate pool
  -> qualify candidate evidence as a batch
  -> evaluate sufficiency from qualified packets
  -> optionally repair missing/unsupported slots
  -> qualify newly retrieved candidates as a batch
  -> re-evaluate sufficiency
  -> resolve genuine conflicts
  -> pack qualified evidence
  -> one structured final synthesis
  -> deterministic claim/slot validation
  -> natural complete answer or explicit partial/insufficient answer
```

Qualification occurs before the sufficiency decision it informs. This supersedes the earlier v9 wording that placed all prose curation after repair. Each completed retrieval round may contribute one batched qualification attempt when prose evidence remains unresolved and the existing budget controller admits the call. Deterministic extraction is always attempted first and does not consume provider budget.

Repair remains bounded by the Query Contract. A repair result cannot support a slot until its new candidates pass the same qualification boundary. If the controller cannot admit required qualification while retaining final-answer reserve, repair stops and the unresolved slot remains explicit.

## Evidence Qualification

The raw retrieval pool remains observable but is not passed directly to the sufficiency gate. `EvidenceExtractor` produces the positive evidence view used by sufficiency and packing:

1. Deterministic extractors handle exact numeric values, formulae, table rows, explicit enumerations, and structured locators.
2. Remaining prose candidates are curated in one batch for the currently unresolved slots.
3. Returned packets must reference an eligible source evidence ID and authorized slot ID.
4. Source text, locator state, and quote/span binding are validated by the existing evidence validators.
5. Unknown source IDs, unauthorized slot IDs, contradicted locators, invalid spans, malformed output, and unsupported statements are discarded.
6. High-risk comparative, causal, superlative, state-of-the-art, best, or outperform statements are FinalClaim candidates rather than positive EvidencePackets.

`validation_status="deterministic_valid"` is assigned only by deterministic validation. The campaign adapter must not stamp that status on an arbitrary retrieved chunk.

The sufficiency gate retains its simple responsibility: a slot is supported when at least one accepted packet authorizes that slot. Its input boundary, rather than increasingly complex logic inside the gate, guarantees that packet presence now means validated support.

## Repair Semantics

Repair planning uses only unresolved or unsupported required slots. It does not repeat already satisfied retrieval work.

After a repair retrieval:

- deterministic qualification runs over new candidates;
- prose qualification is attempted as one batch if required and admitted by the existing controller;
- the qualified evidence view is merged by stable evidence identity;
- sufficiency is recomputed from that qualified view;
- candidates that cannot be qualified remain retrieval diagnostics, not supporting evidence.

Budget or deadline exhaustion is a normal terminal condition. It produces `qualified_partial` when at least one verified finding exists and `insufficient` when none exists. It never promotes unqualified evidence to avoid a partial answer.

## Structured Final Synthesis

The campaign runtime uses the shared `FinalAnswerRenderer` rather than constructing a free-text prompt and a synthetic claim. The final provider input contains:

- original question;
- complete Query Contract and required-slot definitions;
- packed, qualified EvidencePackets only;
- SlotResolutions, including unresolved and explicitly unavailable slots;
- sufficiency state;
- optional conflict arbitration result;
- strict structured-output instructions.

The provider returns structured claims. Every positive claim must include a `slot_id`, a non-empty statement, a permitted support type, and either `evidence_ids` or `premise_evidence_ids`.

Backend validation is authoritative:

- referenced evidence must exist in the final packed context;
- the evidence must be authorized for the claim's slot;
- direct evidence must have a usable validation status;
- calculated claims must have complete and valid premises;
- a required slot is answer-complete only when at least one accepted claim covers it;
- high-risk prose claims use the existing batched verifier when the controller admits it;
- a failed, unavailable, or omitted verifier never upgrades a claim to supported.

The provider cannot set the final response status. The backend derives it from accepted claims and required-slot coverage.

## Response Status and Rendering

`complete` requires every required slot to have both a supported SlotResolution and at least one accepted final claim. A single claim may cover multiple slots only when it explicitly lists those slot bindings and each binding is independently authorized by its evidence.

`qualified_partial` is returned when at least one required slot has an accepted finding and at least one remains unresolved, unavailable, rejected, or uncovered by the final claims.

`insufficient` is returned when no accepted required-slot finding remains.

Rendering is deterministic after claim validation:

- Complete responses are rendered as a normal cohesive answer with deterministic citations. Internal JSON and status labels are not exposed as the primary answer format.
- Partial responses contain a natural-language confirmed section followed by a concise unresolved section derived from required-slot descriptions.
- Insufficient responses state that the supplied evidence cannot support the requested answer.
- No-evidence runs skip final generation and return the deterministic insufficient response.
- Invalid final JSON, unknown evidence references, missing slot bindings, or an unavailable final provider fail closed; they do not restore the current unconstrained prose answer.

## Cost and Budget Behavior

This correction introduces no new public cost policy and no automatic token-ratio gate.

All provider work continues through `BudgetedLlmInvoker` and `RunBudgetController`. Deterministic evidence extraction and deterministic claim validation are preferred. Prose qualification is batched. Final synthesis remains at most one provider call. The existing claim verifier remains batched and selective rather than running for every claim.

If the existing route budget cannot admit an optional stage, the stage is skipped or terminates fail-closed according to its contract. The implementation must not silently increase `max_llm_calls`, runtime-token budgets, or Setup output ceilings to make the new path succeed.

The existing target `Agentic/Naive runtime-token ratio <= 3.0` is evaluated from normal campaign accounting after deployment.

## Observability

Existing normalized v9 observability remains authoritative. The corrected runtime must persist enough state to distinguish:

- retrieved candidates from qualified EvidencePackets;
- deterministic versus prose qualification and their validation statuses;
- initial versus repair-round evidence;
- supported, explicitly unavailable, and not-found SlotResolutions;
- accepted, qualified, and rejected final claims;
- actual used evidence IDs rather than all packed evidence IDs;
- the backend-derived response status;
- zero or one actual final-generation call;
- any actual verifier call through normal accounting.

The runtime must not fabricate a successful qualification, repair, claim, or completion event when the corresponding stage was skipped or failed.

## Compatibility and Rollback Boundary

The correction is intentionally limited to the evaluation v9 adapter and shared v9 components. Existing v8 and non-Agentic paths retain their current prompts and response contracts.

This design does not reintroduce the previously rolled-back visual behavior. Visual evidence still enters the same typed packet and qualification boundary, and final output remains natural prose rather than a UI-specific structured envelope.

Persisted historical campaigns remain readable. No migration rewrites prior one-claim results or retroactively changes their status.

## Testing Strategy

Implementation follows test-driven development and must cover these behaviors before production wiring:

1. A retrieved packet with a slot ID but no semantic/source validation cannot satisfy the slot.
2. Deterministic numeric and structured evidence can satisfy a slot without an evidence-extraction provider call.
3. Prose candidates for multiple slots are sent in one batch and only valid source/slot bindings survive.
4. Qualification failure or malformed output fails closed.
5. Missing qualified slots trigger only the existing bounded repair path.
6. Repair candidates cannot satisfy a slot until qualified.
7. Budget exhaustion preserves final reserve and returns a partial or insufficient result.
8. Final input contains the Query Contract, packed evidence, SlotResolutions, unresolved requirements, and arbitration.
9. Unknown evidence IDs, missing slot bindings, invalid premises, and high-risk verifier failures cannot become supported claims.
10. Multi-slot complete output has accepted claim coverage for every required slot; one synthetic whole-answer claim is rejected.
11. No-evidence execution makes no final provider call.
12. Complete output remains natural; partial output separates confirmed and unresolved content.
13. Persisted used evidence IDs equal the accepted claims' actual references rather than every packed packet.
14. Naive, v8, and non-evaluation chat regression suites remain unchanged.

Focused unit tests cover the extractor, validator, sufficiency gate, repair loop, renderer, claim verifier, budget controller, and execution core. Campaign-runtime integration tests cover the fully wired path and its normalized observability. The existing evaluation suite provides cross-mode regression coverage.

## Acceptance Criteria

- Campaign Agentic v9 no longer stamps raw retrieved chunks as semantically valid evidence.
- Sufficiency and repair operate only on qualified evidence.
- Final synthesis receives the full Query Contract and resolution state.
- Response status is backend-derived; no runtime adapter returns unconditional `complete`.
- Every accepted positive claim is bound to a required slot and packed evidence.
- Multi-slot answers cannot be represented by one synthetic claim unless that claim has explicit, independently valid bindings for every covered slot.
- Missing or invalid evidence yields qualified partial or insufficient output rather than unsupported completion.
- Final generation remains at most one call and no-evidence runs use zero final calls.
- Existing route and runtime budgets are not increased by this change.
- Native, v8, and current general chat behavior are unchanged.
- The follow-up campaign is evaluated against the existing Agentic/Naive runtime-token ratio target of at most 3.0; no new automatic release gate is introduced.

## Authority and Amendment

This document is the semantic source of truth for the faithfulness correction. It narrowly amends the evidence-qualification timing and final-answer wiring in `2026-07-21-agentic-rag-v9-evidence-first-design.md`. All other frozen v9 decisions remain authoritative.
