# Evidence Qualification Rejection Counts Design

## Goal

Expose why a structurally valid evidence-qualification response produces zero qualified packets, without persisting provider responses, prompts, questions, or evidence text.

## Contract

`V9ExecutionMetrics` gains three non-negative integer fields, each defaulting to `0` for historical materializations:

- `qualification_unknown_source_id_count`
- `qualification_unauthorized_source_slot_count`
- `qualification_statement_not_verbatim_count`

Counts aggregate across all qualification rounds in one run. They are serialized in interactive run observability and Export Schema v2. The frontend TypeScript contract and strict Zod decoder accept the same required export fields. No UI panel change is required.

## Classification

- A returned `source_evidence_id` absent from the current candidate map increments `qualification_unknown_source_id_count`.
- Unknown slot IDs or a source/slot pairing outside `eligible_ids_by_slot` increment `qualification_unauthorized_source_slot_count` once per rejected row.
- A candidate rejected specifically because its statement is not a normalized contiguous source span increments `qualification_statement_not_verbatim_count`.
- Malformed JSON/container errors remain `invalid_provider_response`; other malformed row fields remain skipped and are not assigned to these three categories.

Existing source authorization, slot authorization, exact quote binding, Structured Outputs, model selection, budgets, prompts, and sufficiency behavior remain unchanged.

## Verification

Reuse existing backend extractor/runtime/export tests and existing frontend contract/decoder tests. Add no new test module. Verify OpenAPI artifacts and the pinned frontend contract after the backend schema changes.
