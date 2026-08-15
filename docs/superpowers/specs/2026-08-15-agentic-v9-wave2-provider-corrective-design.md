# Agentic RAG v9 Wave 2 Provider Corrective Design

**Date:** 2026-08-15  
**Status:** Approved for specification  
**Scope:** Repair Wave 2 evidence qualification without starting Wave 3

## Problem Statement

The real Q1-Q32 Wave 2 checkpoint campaign
`63c7f821-857a-441b-bd05-d0743c5b185b` produced 234 candidate evidence
packets but zero qualified packets. All 46 persisted `evidence_extraction`
provider calls failed, final generation never ran, and all 32 runs returned the
same `insufficient` fallback answer.

The failure is not evidence-validator strictness. The production
`EvidenceExtractor` sends a system-only message. The installed Gemini adapter
separates that message into `system_instruction` and creates a request with an
empty `contents` list. The extractor then catches every provider exception and
returns an empty packet list, making a provider failure indistinguishable from
a valid no-match qualification result.

The campaign also recorded 14 failed `atomic_contract_planning` calls. That is
a separate provider-health signal whose exact server-side cause is not present
in the redacted export. It must be diagnosed through the existing real-server
planner canary rather than by changing planner behavior speculatively.

## Goals

1. Make evidence qualification use a valid, schema-bound production provider
   request.
2. Preserve fail-closed evidence semantics: unqualified candidates must never
   satisfy a slot.
3. Distinguish provider failure, invalid response, valid no-match, deterministic
   qualification, and provider-qualified evidence in durable observability.
4. Keep the existing route, retrieval, repair, sufficiency, and provider-call
   budgets unchanged.
5. Add production-equivalent checks that fail before deployment when the real
   provider boundary is incompatible.
6. Keep backend OpenAPI and frontend TypeScript/Zod contracts synchronized.

## Non-Goals

- Do not begin Wave 3 structured final synthesis.
- Do not relax `is_qualified_evidence()` or restore blanket
  `deterministic_valid` promotion.
- Do not add per-slot or per-chunk LLM calls.
- Do not increase route token or provider-call limits.
- Do not change the deterministic route, retrieval policy, repair policy, or
  final-answer prompt.
- Do not guess at or alter Atomic Planner behavior until its real-server canary
  identifies the separate provider failure.

## Selected Design

### 1. Shared schema-bound evidence provider

Extend `data_base/agentic_v9/provider_boundary.py` with one evidence
qualification provider factory. It reuses `get_llm("synthesizer")` and
`bind_json_schema()` and binds a fixed response schema:

```text
object
  packets: array
    source_evidence_id: string
    slot_ids: non-empty string array
    statement: non-empty string
```

The object and packet rows reject additional properties. The campaign provider
factory selects this boundary only for `purpose="evidence_extraction"`.

`EvidenceExtractor` sends the formatted qualification request as a non-empty
user message. This matches the working contract-planner request shape and
guarantees that the Gemini adapter creates at least one request `contents`
entry. The response remains subject to the existing local allowlist checks for
source IDs, slot IDs, statements, and source-derived packet construction.

### 2. Typed internal qualification outcome

Add an internal immutable result owned by the evidence extractor:

```text
EvidenceQualificationOutcome
  packets
  status
  failure_code
  provider_call_attempted
  provider_response_received
```

Allowed statuses are:

- `not_attempted`
- `deterministic`
- `provider_qualified`
- `no_match`
- `provider_failed`
- `invalid_response`

Allowed safe failure codes are:

- `provider_attempt_failed`
- `invalid_provider_response`
- `budget_not_admitted`
- `null`

The existing list-returning extractor interface may remain as a compatibility
wrapper, but the campaign adapter must use the diagnostic outcome. This keeps
the execution-core `prose_curate` interface unchanged: it still receives only
the accepted packets, while the campaign adapter writes status and failure
metadata to run state.

Provider exceptions must not be silently converted to a normal empty result.
They are mapped to safe codes without persisting exception messages, request
bodies, credentials, raw prompts, or provider responses.

### 3. Qualification-authoritative observability

Replace the obsolete `semantic_qualification="not_enabled"` value with the
outcome status. `qualification_failure_code` is populated whenever the outcome
failed and remains null for successful or valid no-match outcomes.

The following invariants are required:

- `qualified_packet_count` counts only packets accepted by
  `is_qualified_evidence()`.
- `qualification_provider_call_count` equals persisted qualification LLM call
  attempts.
- `provider_failed` requires `provider_attempt_failed`.
- `not_attempted` may carry `budget_not_admitted` when the controller rejects
  the provider call; ordinary no-candidate/no-unresolved-slot paths keep a null
  failure code.
- `invalid_response` requires `invalid_provider_response`.
- `provider_qualified` requires at least one provider-derived qualified packet.
- A persisted qualification `LLM_CALL_FAILED` row cannot coexist with a null
  run-level failure code for the same qualification round.
- Failures still produce fail-closed sufficiency and may return `insufficient`;
  they must never be presented as ordinary evidence absence.

Update the backend schema, export schema, OpenAPI artifacts, frontend
TypeScript type, frontend Zod decoder, and observability UI labels together.
The UI displays only safe status, counts, and failure code.

### 4. Provider compatibility and regression tests

Add four layers of verification:

1. **Offline adapter-shape test:** construct the real installed Gemini adapter
   request without a network call and assert the evidence request has a
   non-empty `contents` list.
2. **Provider-boundary test:** assert evidence qualification uses its fixed JSON
   schema and rejects prose, missing fields, and additional fields.
3. **Runtime propagation tests:** assert provider failure, invalid response,
   no-match, deterministic success, and provider success produce the exact
   packets/status/failure-code combination.
4. **Positive controls:** Q5 and Q23 must each retain at least one qualified
   packet and become answerable in the integration fixture. A forged
   unqualified candidate must remain insufficient.

Existing mock tests remain useful for local parsing and control-flow coverage,
but they are not considered provider compatibility evidence.

### 5. Real-server canary and deployment checkpoint

Add an evidence qualification canary that performs one request through the
same model configuration, provider factory, bound schema, message construction,
budgeted invoker, and parser used by campaign execution. It must support a
no-network construction/check mode and a deliberate one-call real-server mode.
The real-server mode uses zero automatic retries so one invocation remains one
observable attempt.

The corrective deployment sequence is:

1. Keep Wave 2 deployed but pause campaign test traffic.
2. Deploy the corrective backend and synchronized frontend.
3. Run the existing current/minimal Atomic Planner canary using the deployed
   server model configuration.
4. Run the evidence qualification canary.
5. Run a Q5/Q23 campaign and inspect full observability.
6. Only after both positive controls pass, run the full Q1-Q32 campaign.

Wave 2 corrective is accepted only when:

- both planner canary modes pass;
- the evidence qualification canary passes;
- Q5 and Q23 are answerable and contain qualified packets;
- at least one Q1-Q32 run reaches final generation;
- Q1-Q32 does not repeat 32/32 `insufficient`;
- every failed qualification has a non-null safe failure code;
- qualification call metrics match persisted LLM calls;
- existing call and token ceilings remain unchanged.

If the Atomic Planner canary fails, the checkpoint remains blocked and its
server-side failure is diagnosed separately. This corrective must not mask that
failure by changing planner routing or fallback acceptance.

## Rollback and Safety

No data migration is required. Existing historical exports retain their old
semantic value and remain readable through version-tolerant display logic.
New runs use the corrected qualification statuses. If deployment validation
fails, keep Wave 2 test traffic paused and revert only the corrective commits;
do not enable permissive evidence promotion.

## Success Definition

The corrective is successful when real provider-backed evidence can reach the
existing sufficiency gate, provider failures remain fail-closed and visibly
diagnosed, Q5/Q23 pass as real positive controls, and the Q1-Q32 checkpoint is
no longer a uniform fallback campaign.
