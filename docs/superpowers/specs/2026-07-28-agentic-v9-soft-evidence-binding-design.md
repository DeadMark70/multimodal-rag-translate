# Agentic v9 Soft Evidence Binding Design

Date: 2026-07-28

## Problem

The current v9 runtime replaces authorized retrieval candidates with the output
of one strict evidence-extraction call. The output must be exact JSON and every
statement must be a contiguous verbatim source span. A timeout, malformed
response, paraphrase, or empty result therefore removes all candidate evidence.

The campaign `7d4560e6-99bd-480a-bd47-b89ccdd07022` demonstrated the failure
mode: all 16 runs completed as `insufficient`, all final contexts were empty,
and every answer was a deterministic unresolved-evidence template. The same
model configuration previously produced non-empty contexts in 14 of 16 runs.

## Decision

Evidence extraction and quote binding will no longer be evidence-admission hard
gates for ordinary prose.

The production flow will be:

1. Decompose the question into atomic slots.
2. Enforce source authorization as a hard gate.
3. Run hybrid retrieval within authorized sources.
4. Rerank candidates, falling back to hybrid order on timeout or error.
5. Bind authorized candidates to slots semantically in one batch.
6. Apply structured locator validation only when a slot explicitly requests a
   table, figure, formula, theorem, section, or page.
7. Run missing-slot corrective retrieval when required.
8. Evaluate sufficiency.
9. Generate an evidence-bound answer.

## Admission Rules

Source authorization remains fail-closed. Evidence from an unauthorized
document cannot enter a slot, repair query, packed context, or final answer.

For ordinary prose, a candidate may support a slot without being an exact
contiguous quotation. Evidence extraction may improve the binding and attach
diagnostics, but an extraction timeout, malformed JSON, or empty model result
must not erase authorized candidates.

Structured claims retain stricter behavior:

- When matching locator metadata exists, it can validate and strengthen the
  evidence.
- When metadata explicitly contradicts the requested locator, that candidate is
  rejected for the affected slot.
- When ingestion did not provide locator metadata, the state is
  `locator_unavailable`; semantic evidence remains available.

## Runtime Behavior

The runtime keeps two separate collections:

- `candidate_packets`: authorized, reranked, slot-compatible retrieval evidence.
- `curated_packets`: optional evidence-extraction output that passed validation.

Curated packets augment or replace the corresponding candidate only when valid.
They never replace the whole candidate collection with an empty collection.

If evidence extraction fails, the runtime records a bounded diagnostic and
continues with authorized candidate packets. The final sufficiency and synthesis
stages receive the effective union of valid curated packets and remaining
candidate packets.

Quote-binding status is research telemetry:

- `quote_bound`
- `semantic_bound`
- `locator_validated`
- `locator_unavailable`
- `rejected`

It is not a global answer/no-answer switch.

## Observability

Per run, record:

- raw retrieval count;
- authorized candidate count;
- reranked candidate count and fallback state;
- candidate packet count;
- curated packet count;
- extraction status and safe failure reason;
- effective evidence count;
- slot coverage before and after repair.

Token accounting must attribute the optional model call to
`evidence_extract`. Missing phase telemetry must not change answer behavior.

## Tests

The change is accepted only when automated tests prove:

1. Malformed evidence-extraction JSON preserves authorized candidate packets.
2. Extraction timeout preserves authorized candidate packets.
3. Empty extraction output preserves authorized candidate packets.
4. Valid curated packets are used without duplicating their candidates.
5. Unauthorized candidates are never restored by fallback.
6. Explicit structured-locator mismatch remains rejected.
7. Missing structured metadata remains semantically usable.
8. Final contexts are non-empty when authorized retrieval produced usable
   evidence.

After unit and integration tests, run the Q5/Q7/Q11/Q14/Q16 smoke set. Do not
run another full 16-question evaluation until all five produce evidence packets
and no unauthorized source enters the effective evidence set.

## Non-goals

- Do not redesign the router.
- Do not add per-slot LLM calls.
- Do not make visual extraction a new hard gate.
- Do not change RAGAS scoring.
- Do not perform a repository-wide revert to the pre-v9 implementation.

## Rollback Boundary

The behavioral rollback is limited to the curated-only admission introduced in
the recent evidence-extraction integration. Atomic slots, source authorization,
reranking, conditional structured locators, repair, telemetry, and current UI
contracts remain in place.
