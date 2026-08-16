# Agentic RAG v9 Grounded Completion Design

**Status:** Review requested

**Date:** 2026-08-16

## Objective

Complete the existing Wave 2 evidence-qualification direction without adding a
new agent architecture. The runtime must become stable across user-selected
models, distinguish direct evidence from synthesis work, generate hidden typed
claims, and derive the terminal response status from the verified final result.

The user-facing answer remains natural language. The model selected in the
Evaluation Setup remains authoritative; this design does not force Gemini 3.1,
Gemini 2.5, thinking mode, or any other provider setting.

## Evidence Behind This Design

The paired Q5/Q23 observability exports provide a useful model comparison:

- Gemini 2.5 Flash-Lite: campaign `f3fefec3-94e6-48f1-a869-8b8bac62f2f7`.
- Gemini 3.1 Flash-Lite no-think: campaign
  `0ec9af78-9eef-4ad8-8561-62b6353a6626`.

For Q23, both models received the same candidate statements and identical
qualification prompt hashes. Gemini 2.5 qualified two packets and produced
three non-verbatim rejections; Gemini 3.1 qualified three packets and produced
one non-verbatim rejection. Gemini 2.5 then generated numbers absent from the
packed evidence. Gemini 3.1 used source-present values, but preferred the
coarse prose values `4.5` and `17` over Table 1 values `4.51` and `17.5`, and
asserted an unsupported rounding rule.

For Q5, the central Algorithm 1 candidate was present in both runs. Gemini 2.5
qualified no packet and returned insufficient. Gemini 3.1 qualified three of
four slots and produced a useful partial answer, but omitted the original
branch and final averaging required by the unresolved parent slot.

These results establish four remaining gaps:

1. Qualification asks the provider to copy evidence text that the backend
   already owns, creating model-dependent non-verbatim failures.
2. Direct source requirements and derived comparison/calculation work are mixed
   into the same required-slot set.
3. The campaign runtime bypasses the existing typed final renderer and sends
   only the question plus rendered evidence.
4. `complete` can still be inferred from qualified packet-to-slot bindings
   without verifying the final claims and synthesis obligations.

## Fixed Decisions

- Preserve Wave 2. Do not restore raw-packet promotion or bypass qualification.
- Qualification provider output contains identifiers only; it never supplies
  authoritative evidence prose.
- The backend copies the original candidate statement and provenance after it
  validates the returned source ID and slot bindings.
- A `RequiredSlot` represents a fact that can be supported directly by source
  evidence. Comparison, arithmetic, rounding interpretation, causal
  explanation, and other cross-slot conclusions are `SynthesisObligation`s.
- Reuse the existing `EvidenceExtractor`, `FinalAnswerRenderer`,
  `ClaimVerifier`, Query Contract v2, and budget controller. Do not introduce a
  parallel pipeline.
- One batched qualification call per admitted round remains the limit. Final
  synthesis remains one provider call, and the existing batched high-risk
  verifier remains at most one call.
- User-selected provider/model settings remain unchanged.
- Five behavioral requirements are implemented as four tasks. Terminal status
  correctness is a mandatory invariant, not a separate architecture.

## Relationship To The Existing Recovery Plan

This design preserves the completed Wave 1 and Wave 2 implementation in
`docs/superpowers/plans/2026-08-15-agentic-v9-grounded-recovery.md`. It
supersedes that plan's pending Wave 3 Tasks 11–14 only. A revised implementation
plan must replace those pending tasks rather than append a second final-answer
pipeline.

## Corrected Runtime Flow

```text
Query Contract v2
  -> retrieve source-owned candidates
  -> provider selects source IDs and permitted slot IDs
  -> backend copies exact statements and provenance
  -> sufficiency resolves direct evidence slots only
  -> optional bounded repair and qualification
  -> pack qualified exact evidence
  -> build compact synthesis context
       question
       direct slots and resolutions
       synthesis obligations
       response constraints
       unresolved requirements
       exact packed evidence
       arbitration results
  -> one typed final-synthesis call
  -> deterministic claim validation
  -> at most one batched high-risk claim-verifier call
  -> deterministic terminal-status reduction
  -> natural-language answer
```

## Task 1: Identifier-Only Evidence Qualification

### Provider contract

The qualification response becomes:

```json
{
  "packets": [
    {
      "source_evidence_id": "E1",
      "slot_ids": ["S1", "S2"]
    }
  ]
}
```

The provider selects only from aliases and slot IDs supplied in the request.
Unknown aliases, empty slot lists, unknown slots, and source-slot bindings that
were not eligible for that slot are rejected row by row. One bad row does not
discard other valid rows.

For every accepted row, backend code creates the curated packet from the
canonical `EvidencePoolItem`:

- `statement` is copied from the source candidate;
- source, locator, scope, and raw numeric metadata are copied unchanged;
- the provider-returned slot IDs are deduplicated in order;
- the existing prose validator still confirms source/provenance integrity;
- the provider cannot inject or rewrite evidence text.

The old `statement_not_verbatim` rejection counter remains readable for
historical exports but must be zero for new identifier-only profiles. New code
must not infer success from that historical counter.

This reduces response size and removes the main Gemini 2.5 versus Gemini 3.1
copying variance while preserving fail-closed source and slot authorization.

## Task 2: Separate Direct Slots From Synthesis Obligations

The deterministic decomposition and optional atomic overlay must apply this
classification:

- Direct fact, locator, value, quoted method step, or named source assertion:
  `RequiredSlot`.
- Comparison across facts, arithmetic result, ranking, best/SOTA judgment,
  rounding interpretation, causal conclusion, or cross-source reconciliation:
  `SynthesisObligation` with explicit `depends_on_slot_ids`.
- Output prohibition or presentation instruction: `ResponseConstraint`.

For Q23, Table 1 values and the Abstract/contribution statements are direct
slots. Recomputing ratios and deciding which rounding behavior can or cannot be
confirmed are synthesis obligations. A Table 1 packet alone must never resolve
the rounding obligation.

For Q5, avoid a parent slot that duplicates all child slots. The overall CSS
reconstruction is a synthesis obligation depending on direct slots for the
original branch, flip branches, SiamSSM transformation, accumulation, and final
averaging. If any required direct fact is absent, the overall reconstruction
remains unresolved.

Both deterministic and provider-planned contracts must pass the same final
normalization and validation. The optional atomic planner may improve the
classification, but deterministic fallback must preserve the original question
and produce a valid evidence/synthesis split.

## Task 3: Compact Typed Final Synthesis And Exact Claims

Reconnect the campaign runtime to the existing `FinalAnswerRenderer`. Its input
is a compact projection containing only the information required to answer:

- original question;
- required direct slots and their resolutions;
- synthesis obligations and response constraints;
- unresolved direct requirements;
- packed qualified evidence with exact statement, source, locator, support
  type, and premise IDs;
- arbitration results when present.

Do not serialize route budgets, repeated authorized-document lists, rankings,
raw trace payloads, or unselected candidate text into the final prompt.

The provider output remains hidden JSON. Preserve `supported_findings` for
direct slot claims and add a separate typed collection for synthesized findings:

```python
class SynthesizedFinding(BaseModel):
    obligation_id: str
    statement: str
    premise_evidence_ids: list[str]

class UnresolvedObligation(BaseModel):
    obligation_id: str
    reason: str
```

Each synthesized finding must name a real obligation, include all evidence
premises needed by that obligation's dependent slots, and survive deterministic
validation. Extend `FinalClaim` with `obligation_id`; exactly one of `slot_id`
or `obligation_id` must be present on every accepted claim. Arithmetic and
rounding claims are derived claims, never relabeled as direct source claims.
High-risk comparative, causal, SOTA, best, or outperform claims continue through
the existing single batched verifier.

`used_evidence_ids` is the stable ordered union of evidence and premise IDs from
accepted claims only. Packed but unreferenced evidence is not marked used. The
renderer produces natural prose and citations from accepted typed claims; it
never exposes provider JSON to the user.

Malformed provider output, unknown IDs, missing premise closure, or provider
failure causes a claim-free partial/insufficient result. There is no regeneration
loop.

## Task 4: Deterministic Terminal-Status Invariant

The final status is computed after claim validation and obligation resolution.
Sufficiency remains useful for deciding whether to repair or synthesize, but it
does not own the final `complete` decision.

### `complete`

All of the following must hold:

1. Every required direct slot has a `supported` resolution backed by qualified
   evidence.
2. Every required direct slot is represented by at least one accepted direct
   claim.
3. Every synthesis obligation has an accepted synthesized claim.
4. Every accepted claim references only packed qualified evidence and has
   complete premise closure.
5. No required direct slot or synthesis obligation remains unresolved.

### `qualified_partial`

At least one accepted claim can be rendered, but one or more required direct
slots, synthesis obligations, or claim validations remain unresolved.

### `insufficient`

No accepted claim can be rendered, or final generation fails before producing
any valid claim.

This reducer is the explicit enforcement of requirement five. It is kept as a
small pure function and covered independently so later runtime changes cannot
restore packet-presence-based completion.

## Observability And Compatibility

New runs must expose enough typed data to explain the result:

- identifier-only qualification profile/version;
- candidate and qualified counts;
- used-evidence count;
- unresolved-requirement count across slots and obligations;
- final generation and verifier call counts;
- deterministic terminal status and reasons.

Do not add redundant aggregate fields when the same distinction is already
available from typed final claims (`slot_id` versus `obligation_id`) or typed
unresolved rows.

Historical exports remain readable. New profile rows use the new fields and
must not silently default missing values to zero. The backend export schema,
OpenAPI artifacts, frontend TypeScript/Zod decoder, and observability panels are
updated in the same contract checkpoint.

## Error Handling

- Qualification is row-tolerant and batch-fail-closed: keep valid authorized
  rows, reject malformed/unknown/unauthorized rows, and never promote a raw
  candidate because the provider failed.
- If qualification cannot be admitted while preserving final-answer reserve,
  stop repair and retain unresolved slots.
- If typed final synthesis is malformed, do not fall back to unconstrained prose.
- If claim verification cannot be admitted or fails, reject the affected
  high-risk claims and return partial/insufficient according to the reducer.
- No exception text, provider payload, prompt, or evidence outside the
  authorized packed set is exposed in interactive or exported diagnostics.

## Test Strategy

Focused tests must prove behavior rather than add broad fixture volume:

1. Identifier-only rows copy the exact canonical statement and provenance.
2. Mixed valid/invalid qualification rows preserve valid rows.
3. Unknown alias and unauthorized source-slot bindings remain rejected.
4. Gemini-style extra prose/content blocks cannot become evidence.
5. Q5 produces direct child slots plus one reconstruction obligation and cannot
   become complete without the original branch and averaging evidence.
6. Q23 produces direct Table/Abstract/contribution slots plus ratio/rounding
   obligations; Table 1 alone cannot resolve the rounding obligation.
7. The compact final payload contains the contract semantics and exact packed
   evidence but excludes unrelated runtime data.
8. Direct and synthesized findings produce separate typed claims.
9. Used evidence contains only IDs referenced by accepted claims.
10. Packet-to-slot coverage without accepted claims cannot produce `complete`.
11. An unresolved required obligation produces `qualified_partial`.
12. All direct slots, obligations, claims, and premises resolved produces
    `complete`.
13. Provider failure or wholly rejected claims produces `insufficient`.
14. Existing provider-call and token-budget bounds remain unchanged.

Checkpoint testing should repeat Q5 and Q23 with the user-selected Gemini 2.5
Flash-Lite first. Gemini 3.1 Flash-Lite no-think remains a useful comparison,
not a runtime requirement. The full Q1-Q32 campaign follows only after focused
checkpoint behavior is correct.

## Non-Goals

- No new router or query-complexity classifier.
- No model pinning or automatic provider substitution.
- No per-slot or per-chunk LLM calls.
- No new retrieval engine, reranker, graph strategy, or context-replacement
  architecture.
- No RAGAS prompt tuning.
- No automatic Agentic/Naive token-ratio release gate.
- No changes to Naive, Advanced, Graph, v8, or general chat execution.

## Acceptance Criteria

- Qualification provider output cannot modify evidence prose or provenance.
- Direct evidence slots and synthesis obligations are semantically distinct in
  every new Query Contract v2.
- Campaign final generation uses the compact typed synthesis path and never the
  question-plus-evidence free-text adapter.
- Every final claim maps to exactly one direct slot or one synthesis obligation
  and uses exact evidence/premise IDs.
- `complete` is impossible from slot bindings alone.
- Q5 and Q23 expose honest partial/complete statuses consistent with accepted
  claims, not merely higher aggregate scores.
- Provider-call budgets and user-selected model configuration remain intact.
