# Agentic v9 Provider Schema Compaction Design

## Problem

The Agentic v9 contract planner's complete Pydantic JSON Schema is valid
locally, but Gemini 3.1 Flash-Lite rejects it at provider invocation with HTTP
400 `INVALID_ARGUMENT`. Live canary isolation established that:

- the minimal schema succeeds;
- `evidence_requirements + confidence` succeeds;
- adding `synthesis_obligations` alone succeeds;
- adding `response_constraints` alone succeeds;
- the combined planner schema fails;
- the same complete schema succeeds after removing provider-irrelevant
  validation metadata, reducing its serialized size from 3894 to 2331 bytes.

The failure is therefore the provider's structured-output schema complexity
boundary, not the planner response parser, API credentials, model availability,
or an invalid planner domain contract.

## Selected Design

Keep `atomic_contract_planner_response_schema()` as the canonical strict schema.
Add one pure provider projection that recursively removes only:

- `additionalProperties`
- `title`
- `default`
- `minLength` / `maxLength`
- `minItems` / `maxItems`
- `minimum` / `maximum`

The projection preserves field names, object and array structure, `$defs` and
`$ref`, `required`, `enum`, `anyOf`, and nullability. The contract-planning
provider and the `current` canary bind this projected schema.

## Validation Boundary

Provider compaction does not weaken what the application accepts. After the
provider returns JSON, `_PlannerDecision` remains the authoritative transport
validator and continues enforcing:

- `extra="forbid"`;
- string, list, and numeric bounds;
- required fields and enum values;
- the existing planner semantic validation and immutable-route overlay rules.

The provider schema is only a generation aid. The canonical Pydantic schema is
the acceptance boundary.

## Scope

In scope:

- one pure schema projection owned by the contract-planner provider boundary;
- production contract-planner binding;
- `current` contract-planner canary parity;
- focused regression tests and a short documentation note.

Out of scope:

- changes to `QueryContract`, `_PlannerDecision`, or planner prompts;
- dynamic per-question schemas;
- additional provider calls or retries;
- evidence-qualification changes;
- relaxed post-response or semantic validation.

## Verification

Tests must prove that:

1. the projection removes exactly the approved keywords recursively;
2. all structural planner fields remain in the projected schema;
3. the canonical schema retains all original strict constraints;
4. production and the `current` canary bind the same projected schema;
5. malformed, extra, or out-of-range responses remain rejected by the
   canonical Pydantic and semantic validation path;
6. affected planner, provider-boundary, canary, campaign-runtime, and evidence
   qualification tests pass, followed by scoped Ruff.

