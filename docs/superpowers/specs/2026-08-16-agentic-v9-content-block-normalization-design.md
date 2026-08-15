# Agentic v9 Content-Block Normalization Design

## Problem

Gemini structured output now reaches the application as a LangChain `AIMessage`
whose `content` is a list of typed content blocks. The returned JSON is valid,
but the contract-planning and evidence-qualification parsers accept only string
content, bytes, or already-decoded mappings. Evidence qualification therefore
reports `invalid_provider_response` before applying its strict packet checks.

The real canary demonstrated the concrete valid response:

```json
{"packets":[{"source_evidence_id":"E1","slot_ids":["S1"],"statement":"a two-stage decoder"}]}
```

wrapped in a `type="text"` content block with separate Gemini signature metadata.

## Scope

- Normalize raw Agentic v9 provider responses at the shared provider boundary.
- Use LangChain's `AIMessage.text` property when available. It joins only text
  blocks and excludes block metadata such as Gemini signatures.
- Preserve compatibility with raw strings and existing mapping-based test doubles.
- Route contract planning, evidence qualification, and their canaries through the
  same normalization behavior.
- Keep the raw `AIMessage` provider contract, accounting callbacks, budgets,
  schemas, and downstream semantic validation unchanged.

The current contract-planner provider-invocation failure is a separate issue. This
change makes content-block responses parseable after a response is received; it
does not weaken or mask provider invocation errors.

## Approaches Considered

### 1. Shared Agentic v9 response-text boundary — selected

Add one small helper beside the shared provider builders. Consumers use it before
JSON decoding. This keeps provider-specific response representation out of domain
parsers while retaining their existing strict validation.

### 2. Parse content blocks independently in every consumer — rejected

This is locally small but duplicates subtle block filtering in the planner,
extractor, smoke canaries, and future structured phases. The implementations could
drift and accidentally include signature metadata.

### 3. Replace raw binding with `with_structured_output()` — rejected

This changes the provider return type and would require broader changes to raw
response parsing, usage accounting, budget instrumentation, and canaries. It is
unnecessary for the confirmed representation mismatch.

## Data Flow

1. The bound Gemini provider returns a raw `AIMessage`.
2. The shared normalizer obtains `AIMessage.text` when it is a string-compatible
   value; raw string responses remain supported.
3. Contract planning JSON-decodes the normalized text and applies the existing
   Pydantic and semantic checks.
4. Evidence qualification JSON-decodes the normalized text and applies the
   existing exact envelope, evidence ID, slot ID, eligibility, and span checks.
5. Missing text, malformed JSON, unexpected keys, or invalid evidence continue to
   fail closed with their current outcomes.

## Error Handling and Security

- Do not concatenate arbitrary block metadata.
- Do not serialize the complete content-block list with `str()`.
- Do not accept non-text blocks as JSON.
- Do not expose provider exception text, signatures, prompts, or credentials.
- Keep current sanitized canary failure payloads.

## Verification

- RED: an `AIMessage` containing the observed text block is rejected by the
  current evidence parser and contract response validator.
- GREEN: the same response qualifies `E1` for `S1`, while malformed/non-text
  content still fails closed.
- Run focused provider/planner/extractor/canary tests, then the impacted Agentic v9
  runtime suite and scoped Ruff.
- After deployment, rerun both real-server canaries. Contract provider invocation
  is evaluated independently from response normalization.

