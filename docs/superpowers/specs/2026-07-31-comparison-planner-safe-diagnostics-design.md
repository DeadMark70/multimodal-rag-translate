# Agentic v9 Comparison Planner Safe Diagnostics

## Goal

Identify the exact boundary that turns production Gemini comparison-planner
responses into `schema_violation`, without changing planner decisions, retrieval,
reranking, repair, synthesis, prompts, or provider-call counts.

## Scope

The planner keeps its existing public outcome:

- `status`: `planned` or `fallback`
- `fallback_reason`: the existing stable reason such as `schema_violation`

Fallback outcomes additionally carry optional diagnostics:

- `fallback_stage`: one of `response_decode`, `transport_schema`,
  `subject_validation`, `trusted_plan_validation`, or `numeric_guard`
- `validation_issues`: a bounded list of objects containing only:
  - `path`: normalized field path, for example `subjects.0.subject_role`
  - `type`: stable validation error type, for example `missing`

No raw response text, invalid input value, prompt content, source name,
document identifier, exception message, Pydantic URL, or validation context is
persisted.

## Data flow

1. `ComparisonPlanner` catches failure at the precise boundary.
2. Pydantic errors are reduced to `path` and `type`, deduplicated, sorted, and
   capped at eight entries.
3. The v9 campaign runtime places the diagnostics beside the existing planner
   status in `agent_trace.agentic_v9.comparison_planner`.
4. Existing observability projection and redacted export expose the two new
   fields in `comparison_summary`.
5. Existing consumers that do not know the fields remain compatible because
   the original status and fallback reason do not change.

## Safety and failure behavior

- Diagnostics are metadata only and cannot promote or reject a plan.
- Unexpected exceptions do not expose messages; they produce a stage with an
  empty issue list.
- Non-schema fallbacks such as timeout, provider error, not-comparison, and
  invalid subjects preserve their current behavior.
- `validation_issues` is empty when no Pydantic validation error exists.

## Verification

Automated tests must prove:

1. Transport validation records only safe field paths and error types.
2. Trusted-plan validation and numeric guard are distinguishable.
3. Raw invalid values and exception text do not appear in serialized traces or
   export projections.
4. A valid comparison still produces the same plan.
5. Existing fallback reasons and provider-call counts remain unchanged.

After deployment, run only Q3, Q4, and Q14 once. The resulting
`comparison_summary` will identify the exact production incompatibility. A
separate minimal behavior fix will be designed from that evidence; this change
does not attempt to guess or repair the provider payload.
