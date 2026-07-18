# Evaluation Center P0 Final Review Corrections Design

**Date:** 2026-07-18

**Scope:** Corrections required by the final cross-repository review of the Evaluation Center P0 research metrics and accounting implementation.

## 1. Objective

Close the remaining strictness gaps without weakening the approved version-2 accounting rules. Missing measurements must not appear as measured zero, cross-mode cost-quality comparisons must use one evaluator cohort per metric, RAGAS retry metadata must come from durable state, and the frontend must render the backend's explicit unclassified phase value.

## 2. Token Values for Missing and Partial Usage

If an included scope has no measured usage events, `TokenBreakdown` returns `null` for every token category and total, with an empty `by_phase` map. It never returns category zeroes created only by schema defaults.

If a scope contains both measured and missing usage events, known category and phase values remain visible as measured subtotals while `accounting_status="partial"`. Documentation and tests treat those values as incomplete subtotals, not complete totals. No new numerical estimate is introduced.

## 3. Campaign-Level Evaluator Cohorts

For each requested quality metric, research analytics selects one deterministic campaign-level canonical evaluator identity from official, current scores. The identity consists of evaluator model, metric version, and compatibility signature. Selection is ordered by descending valid official sample count and then by lexical identity for deterministic ties.

Every mode aggregates only scores from that metric's canonical identity. A mode whose current scores belong only to a different identity remains visible but is not comparable and receives `evaluator_metadata_mismatch`. Campaign aggregates use the same canonical identities, so they never mix evaluator policies across modes.

Per-result `evaluation_signature` continues to protect input currentness and idempotency. It is not used as a cross-result cohort identity. Legacy scores without a compatibility signature retain the existing strict raw-signature fallback and are never treated as compatible by assumption.

## 4. Durable RAGAS Retry Accounting

`ragas_batch` accounting scopes persist a nullable, non-negative `retry_count`. Fresh scopes start at zero. Existing databases receive an additive migration whose historical rows remain `null`; absence of historical retry metadata is never interpreted as zero.

The RAGAS retry loop increments the durable counter immediately before each retry attempt. A first-attempt success records zero; one failed attempt followed by success records one; exhausted retries record the number of retries actually started. Provider callback failures that trigger the same retry loop follow the same rule.

Research analytics sums the persisted scope counters only when every included RAGAS scope has durable retry metadata. If any included scope has an unknown historical counter, `EvaluationOverheadSummary.retry_count` is `null` and the response carries the existing partial/legacy warning semantics. The API and frontend types therefore make this field nullable. Analytics does not infer retries from `scope_key`, invocation UUIDs, usage-event count, or provider cost.

## 5. Frontend Unclassified Phase Rendering

The token breakdown reads `by_phase.unclassified` directly. When the key exists, its measured value is displayed, including a real zero. When the key is absent:

- `phase_attribution_status="complete"` displays zero;
- `partial` or `not_available` displays `N/A`.

The frontend does not derive unclassified tokens by subtracting every phase from total tokens.

## 6. Partial Quality Sample Accounting

For each result and metric, analytics classifies the sample as valid, evaluating, failed, or missing from its current score and RAGAS work state. `valid_samples`, `missing_samples`, and `failed_samples` are computed per result. A terminal failure is counted as failed even when other results for the same metric have valid scores.

## 7. Compatibility and Migration

All changes are additive. Existing research-summary URLs and response fields remain available; `evaluation_overhead.retry_count` becomes nullable so unknown history cannot masquerade as zero. New durable retry metadata is internal to accounting scopes. Old accounting rows are not backfilled or estimated. Legacy and unresolved observations remain visible with partial or unavailable statuses.

Normal chat remains outside evaluation accounting. Existing legacy analytics endpoints remain available but are not used as the Campaign Overview research source.

## 8. Test and Review Requirements

Backend tests must cover:

- zero measured events returning nullable categories;
- mixed measured/missing events returning status-labelled subtotals;
- two modes with different evaluator identities not entering one comparison;
- deterministic canonical cohort selection and a fully compatible control;
- successful RAGAS retry, exhausted retry, and callback-triggered retry counters;
- historical RAGAS scopes without retry metadata returning `retry_count=null`;
- per-result failed sample counts in a mixed valid/failed metric;
- migration and store round-trip for durable retry metadata.

Frontend tests must cover:

- a non-zero explicit `by_phase.unclassified` value;
- absent unclassified with complete attribution displaying zero;
- absent unclassified with partial or unavailable attribution displaying `N/A`;
- missing token categories rendering `N/A`, never synthetic zero.

Completion requires focused tests, full backend pytest, full frontend Vitest, frontend lint/build, changed-file Ruff checks, independent backend/frontend correction reviews, and a renewed cross-repository final review.
