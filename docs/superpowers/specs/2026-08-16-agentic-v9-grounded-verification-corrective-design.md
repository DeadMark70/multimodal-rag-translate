# Agentic v9 Grounded Verification Corrective Design

**Date:** 2026-08-16

**Status:** Approved for implementation planning

**Scope:** Backend Agentic RAG v9 final-claim validation, verifier admission, feasibility accounting, and generic qualification anchors. Q5 and Q23 are regression fixtures only.

## Problem

Wave 3B connected structured final synthesis, claim/evidence mapping, obligation closure, and honest terminal status. The first real checkpoint exposed four general faults:

1. obligation-bound findings can have complete qualified direct premises but are rejected because the current deterministic checker requires a pre-existing `calculated` packet;
2. every obligation claim bypasses the existing batched verifier, including arithmetic, rounding, comparison, and qualification claims;
3. the numeric parser drops values followed by `x` or `×`, allowing ratio and rounding claims to evade deterministic checks;
4. a valid direct paraphrase is rejected when it is not textually identical to the complete evidence packet, even when provenance and slot authorization are correct.

The checkpoint also exposed a retrieval-side false positive: a provider can bind topically related evidence to a slot even when an explicit question/slot locator or numeric anchor is absent. This can make sufficiency look complete before the final layer rejects the claims.

## Goals

- Preserve deterministic provenance, qualification, packed-context, slot-authorization, and premise-closure checks.
- Route semantically unresolved claims into the existing verifier in one batch per run.
- Let obligation findings derive from complete direct premises without inventing a calculated evidence packet.
- Parse scalar, percentage, `x`, `×`, and `fold` numeric forms without silent loss.
- Make the final provider distinguish source-stated facts, derived conclusions, and unresolved items.
- Reserve at most one claim-verifier provider call during feasibility admission.
- Reject generic qualification bindings that miss explicit structured locators or numeric anchors.
- Keep the user-visible answer as natural language.

## Non-goals

- No model pinning or model-family branch.
- No second verifier pipeline, NLI dependency, classifier, repair loop, or per-claim call.
- No question IDs, golden answers, paper names, Algorithm 1 special case, or hard-coded Q5/Q23 values in production code.
- No frontend or export schema change; existing claim and metric shapes are sufficient.
- No change to the release token-ratio policy.

## Decision 1: Deterministic gate returns a disposition

The current boolean-like contract conflates structural validity, semantic support, and rejection. Replace it with an explicit result:

```python
ClaimGateStatus = Literal["accepted", "verify", "rejected"]

class ClaimGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    status: ClaimGateStatus
    reason: str | None = None
```

The gate owns only facts the application can prove:

1. referenced IDs exist in the packed context;
2. every referenced packet is qualified;
3. direct evidence is authorized for the target slot;
4. every obligation dependency contributes at least one packed direct premise;
5. explicit direct-claim numeric tokens are present in cited evidence with the same semantic suffix;
6. a direct statement that is a normalized verbatim span of cited evidence can be accepted without a model.

Disposition rules:

- structural/provenance/premise failure: `rejected`;
- obligation claim with complete premises: `verify`;
- direct normalized verbatim span with supported numeric tokens: `accepted`;
- structurally valid direct paraphrase: `verify`;
- direct claim with an unsupported explicit number/ratio/percentage: `rejected`.

The gate never decides that a calculation, comparison, rounding explanation, qualification, or non-verbatim paraphrase is semantically supported.

## Decision 2: One verifier batch owns semantic entailment

All `verify` claims are sent to the existing `ClaimVerifier.verify()` in one call. The batch includes:

- direct paraphrases;
- arithmetic and ratio derivations;
- rounding statements;
- comparison, selection, causal, and qualification obligations;
- narrative aggregation obligations that are not a verbatim direct fact.

The verifier receives each claim, its target requirement/obligation description, and only its cited packed evidence. It must return one verdict per claim. Missing, malformed, or unavailable verdicts reject the corresponding claim. There is no retry and no regeneration.

## Decision 3: Obligation premises are legal evidence

An obligation-bound claim is legal when:

- the obligation exists;
- all `depends_on_slot_ids` are covered by its cited packed, qualified direct premises;
- every premise ID exists and is included in the final context pack.

It does not require an evidence packet with `support_type="calculated"`. `support_type` remains useful observability metadata but does not manufacture a new source object and does not bypass the verifier.

## Decision 4: Numeric tokens preserve semantics

Normalize numeric tokens into `(decimal_value, semantic_kind)`:

- `33`, `33.0` -> `("33", "scalar")`;
- `33x`, `33×`, `33-fold` -> `("33", "ratio")`;
- `12.5%`, `12.5 percent` -> `("12.5", "percent")`.

The parser must not discard a number because it is adjacent to `x` or `×`. A direct ratio/percentage claim cannot be accepted from a scalar-only source token. Derived obligation numbers are not required to appear literally in the premises, but the verifier must validate the derivation.

## Decision 5: Prompt is prevention, not authorization

The final-synthesis prompt must say:

- evidence insufficiency belongs in an unresolved collection, never a supported finding;
- source-stated facts and derived conclusions must be distinguished;
- rounding method must not be inferred when the source does not state it;
- every synthesized finding must cite all direct premises required by its obligation;
- uncertainty or inability to derive a result must be emitted as unresolved.

A stronger model should reduce rejected candidates, but structured output is still candidate data. Only the gate and verifier authorize claims.

## Decision 6: Verifier budget is explicit and bounded

Post-contract feasibility reserves one optional `claim_verifier` provider call in addition to final synthesis and qualification. This is a capacity reservation, not a mandatory invocation. Runtime invokes the phase only when the deterministic gate emits at least one `verify` claim. `FinalAnswerRenderer` remains the sole owner and can call it at most once.

If the verifier call is unavailable or rejected by the controller, all pending claims become unresolved/qualified-rejected. The system must not accept them to preserve apparent completeness.

## Decision 7: Qualification uses generic hard anchors

Before accepting a provider-selected source/slot row, derive conservative anchors from the original question and slot definition:

- structured locators such as `Algorithm 1`, `Table 1`, `Figure 2`, or named regions such as `Abstract` and `contribution`;
- explicit numeric ratio/percentage tokens;
- mixed-case or acronym-like technical identifiers.

Candidate text plus canonical metadata must satisfy the applicable anchors. Explicit slot-local anchors take precedence; question-level structured locators are inherited when a slot describes a requested subpart but has no locator of its own. Existing source authorization and row-tolerant parsing remain unchanged.

No common prose keywords are treated as hard anchors. When no hard anchor can be derived, current provider qualification behavior remains available.

## Terminal semantics

- Only accepted deterministic claims and verifier-supported claims contribute to `used_evidence_ids`.
- Rejected or omitted claims generate unresolved rows through the existing reducer.
- `complete` still requires every direct slot and synthesis obligation to have an accepted claim and no unresolved rows.
- A provider-bound packet alone never proves final completion.

## Regression strategy

Unit fixtures describe generic behavior, not benchmark IDs. Two replay regressions use sanitized structures derived from real Q5/Q23 exports:

- a multi-step mechanism with missing explicit locator evidence cannot appear fully qualified/sufficient;
- a ratio calculation may be accepted from complete direct premises, while an unstated rounding method is rejected or rendered unresolved.

Production code must not inspect `question_id` or match fixture-specific prose.

## Rollback boundary

Implement as two independently revertible backend commits:

1. claim gate, numeric parsing, verifier routing, prompt, and budget admission;
2. generic qualification anchors and real-export-derived regressions.

No Wave 3B schema rollback and no restoration of the old free-text final adapter are permitted.
