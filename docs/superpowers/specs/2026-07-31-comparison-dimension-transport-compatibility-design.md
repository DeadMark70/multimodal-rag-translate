# Comparison Dimension Transport Compatibility Design

## Problem

The production Q3/Q4/Q14 smoke shows that Gemini returns every
`dimensions` item as a non-string value. The planner currently requires
`list[str]`, so Pydantic rejects the complete response at the
`transport_schema` boundary and the comparison overlay never runs.

## Approved design

Normalize provider-only dimension values before `_PlannerPayload` validation:

- Preserve non-empty string items unchanged.
- Accept an object only when one preferred textual label can be selected from
  `dimension`, `name`, `label`, or `value`, in that order.
- Ignore descriptive companion fields after a preferred label is selected.
- Reject arrays, numeric scalars, empty strings, objects without a supported
  textual label, and conflicting objects whose supported labels disagree.
- Keep the trusted `ComparisonPlan.dimensions` type as `list[str]`.
- Keep subject validation, length limits, deduplication, numeric invention
  guard, and fail-soft fallback unchanged.

## Security and observability

Raw provider values are not persisted. Invalid values continue to produce a
bounded `transport_schema` validation issue. The normalizer does not accept
arbitrary object serialization as evidence or as a retrieval query.

## Verification

TDD must cover the production-shaped object response, companion metadata,
conflicting labels, unsupported objects, numeric values, and existing string
responses. The Q3/Q4/Q14 smoke remains the production acceptance checkpoint.
