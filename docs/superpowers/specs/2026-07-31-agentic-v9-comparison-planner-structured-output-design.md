# Agentic v9 Comparison Planner Structured Output Design

**Date:** 2026-07-31

**Status:** Approved design (Option A)

## Goal

Make the Agentic v9 comparison planner reliably produce a valid, compact
comparison plan without requiring thinking mode, adding an LLM call, or changing
retrieval, reranking, source authorization, context packing, or final synthesis.

## Problem

The comparison planner currently asks Gemini to return JSON through prompt text,
then decodes and validates the free-form response. The provider call does not
carry a native response schema. Gemini can therefore return syntactically valid
JSON whose field types do not match the planner contract, such as object-valued
entries in the string-only `dimensions` array.

Thinking mode improves this behavior only intermittently and adds reasoning
tokens and latency. Parser compatibility rules reduce individual failures but do
not make the transport contract deterministic.

## Scope

This change is limited to the comparison-planning phase of evaluation Agentic
v9. It includes:

- a smaller provider-facing transport schema;
- a shorter semantic-only planner prompt;
- Gemini native structured JSON output for `agentic_v9_comparison_plan`;
- deterministic promotion from untrusted transport data into the existing
  trusted `ComparisonPlan` domain model;
- fail-soft fallback and existing safety validation;
- a versioned execution profile and focused regression coverage.

It does not include:

- additional planner calls or schema-repair retries;
- changes to Hybrid retrieval, reranking, per-task top-k, or final context
  packing;
- changes to source authorization, graph or visual execution, or final
  synthesis;
- OpenRouter support or a new cross-provider structured-output abstraction;
- frontend, API response, database, or historical-result migrations.

## Architecture

The comparison path remains one optional, budgeted provider call:

```text
comparison eligibility
  -> budget admission
  -> Gemini native structured output
  -> compact transport validation
  -> deterministic subject promotion
  -> existing semantic and numeric guards
  -> comparison overlay or fail-soft base retrieval
```

Only the provider returned for purpose `agentic_v9_comparison_plan` is bound to
the JSON schema. Other Agentic v9 phases keep their existing provider behavior.

The binding must return the normal LangChain response object so
`usage_metadata` remains visible to `BudgetedLlmInvoker`. The implementation
must not replace the response with a parsed-only dictionary that loses token
accounting metadata.

## Compact Transport Contract

The provider-facing schema is intentionally smaller than the trusted domain
model:

```json
{
  "is_comparison": true,
  "subjects": [
    {
      "name": "nnMamba",
      "query": "nnMamba Params FLOPs"
    },
    {
      "name": "EfficientMedNeXt-L",
      "query": "EfficientMedNeXt-L Params FLOPs"
    }
  ],
  "dimensions": ["Params", "FLOPs"],
  "qualification": "cross-paper comparison"
}
```

Contract rules:

- `is_comparison` is a required boolean.
- `subjects` contains zero to four objects.
- Each subject contains only a non-empty `name` and `query`.
- `dimensions` contains at most twelve strings and no nested objects.
- `qualification` is a bounded optional string.
- Unknown fields do not become trusted domain data.

The LLM no longer generates `subject_id`, `aliases`, `subject_role`, or
`question_span`.

## Deterministic Promotion and Safety

The backend converts validated transport subjects into the existing
`ComparisonSubject` model:

- `display_name` comes from transport `name`;
- `retrieval_query` comes from transport `query`;
- `aliases` is an empty list;
- `subject_id` is a stable opaque identifier derived from normalized subject
  text, not copied from provider output.

Before promotion, each subject name must appear explicitly in the original
question with the existing Unicode-aware boundary behavior. Duplicate
normalized subjects are rejected. A valid comparison requires two to four
unique accepted subjects.

The existing invented-number guard remains authoritative: numeric terms in a
retrieval query or dimension must already be present in the question. The
trusted `ComparisonPlan` validation remains in place even though the provider
uses native structured output, because structured output guarantees shape, not
semantic correctness.

## Prompt Design

The prompt moves into the standard Agentic RAG prompt registry and follows the
repository prompt-definition contract (`version`, `description`,
`required_variables`, and `template`). It contains semantic instructions only:

- classify a comparison only when at least two independent named entities are
  compared, related, or jointly judged;
- do not treat two claims, metrics, capabilities, conditions, or prompt types
  about one entity as independent subjects;
- create one subject-specific retrieval query per entity;
- list only dimensions explicitly requested by the question;
- do not answer, choose a winner, or invent values.

The prompt does not repeat a JSON example or schema. The user message contains
only the question. Runtime-authorized source names are not sent to the planner;
source authorization remains enforced at the retrieval boundary and is not a
planner classification input.

## Provider Binding

The production Google/LangChain provider for
`agentic_v9_comparison_plan` is bound with:

```python
response_mime_type="application/json"
response_schema=<compact planner JSON schema>
```

The binding is purpose-specific. `final_answer`, graph locator, visual
extraction, claim verification, and all non-v9 callers remain unchanged.

This version intentionally does not add a provider-neutral structured-output
interface. When OpenRouter becomes an active runtime provider, the same compact
JSON schema can be mapped in a separate provider-adapter change.

## Error Handling

The planner remains fail-soft:

- timeout -> existing `timeout` fallback;
- provider or schema-binding rejection -> existing `provider_error` fallback;
- invalid JSON or response envelope -> existing `invalid_response` fallback;
- schema mismatch -> existing `schema_violation` diagnostic;
- fewer than two valid explicit subjects -> existing `invalid_subjects`
  fallback;
- invented numeric terms -> existing numeric-guard fallback.

No failure starts a second LLM call. Base retrieval continues when the
comparison overlay is unavailable. No planner failure may clear already
available evidence or fail the whole campaign unit.

## Observability and Versioning

Existing `comparison_plan` phase accounting, terminal-attempt persistence, safe
validation diagnostics, and planner latency fields remain authoritative.

The Agentic v9 execution profile is bumped with a structured-comparison suffix
so campaigns produced before and after this behavior change are not considered
the same benchmark condition. No historical rows are rewritten.

## Test Strategy

### Unit tests

- The generated provider schema defines `dimensions.items.type` as `string`.
- A two-subject payload promotes into the existing trusted plan.
- A three-subject payload promotes without changing the four-subject cap.
- Single-entity claim arbitration does not produce a comparison overlay.
- Duplicate subjects, absent question spans, and invented numbers fail-soft.
- A malformed transport payload produces sanitized validation diagnostics.
- Prompt registry entries declare and format only the required question
  variable.

### Provider-boundary tests

- Only `agentic_v9_comparison_plan` receives JSON MIME type and response schema.
- Other purposes receive the unchanged provider.
- The bound response retains raw usage metadata through
  `BudgetedLlmInvoker`.
- A binding/provider error consumes no retry and returns the existing fallback.

### Runtime integration tests

- A valid comparison plan overlays subject tasks and preserves the per-subject
  chunk cap.
- A non-comparison or invalid plan preserves base retrieval.
- Comparison planner phase accounting and safe diagnostics remain complete.
- The new execution profile is persisted in success and failure projections.

### Server smoke

Run Q3, Q4, and Q14 with thinking disabled and three repeats each:

- zero `dimensions/value_error` failures;
- Q4 produces two independent subjects;
- Q14 produces three independent subjects;
- Q3 does not promote two claims about one entity into an entity comparison;
- at most one comparison planner provider call per eligible run;
- token accounting and phase attribution remain complete;
- fallback runs still produce usable base-retrieval results.

Only after this nine-run smoke passes should the full sixteen-question paired
evaluation run.

## Rollback

The change is split into focused commits for transport/prompt, provider binding,
and integration/versioning. Reverting those commits restores prompt-only planner
behavior. The existing `comparison_specialization_enabled` runtime switch
remains the emergency capability-level disable path; no new feature flag is
introduced.

## Acceptance Criteria

- Gemini structured output is used only for comparison planning.
- The provider-facing schema contains no provider-generated identifiers, roles,
  aliases, or exact-span fields.
- The planner works with thinking disabled.
- No additional LLM call or retry is introduced.
- Existing source authorization and retrieval behavior are unchanged.
- Native structured output does not break token accounting.
- Q3/Q4/Q14 smoke satisfies the conditions above before full evaluation.
