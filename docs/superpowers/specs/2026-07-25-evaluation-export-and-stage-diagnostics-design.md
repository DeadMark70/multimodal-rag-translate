# Evaluation Export and Stage Diagnostics Design

> This document defines the approved repair for misleading v9 stage errors,
> incomplete failed-run diagnostics, and the inactive evaluation export control.

## Goal

Make the Evaluation Center distinguish execution failures from v9 capability
gaps, preserve safe failure diagnostics for every failed run, and make the
existing redaction-aware export endpoint produce a downloaded JSON artifact.

## Scope

This change is intentionally limited to the existing evaluation analytics and
Evaluation Center surfaces. It does not add a new preview endpoint, alter the
v9 query contract, change Graph/Visual execution policy, or retain raw
provider responses.

## Semantics

The UI and export contract have three distinct states:

| State | Definition | Surface |
| --- | --- | --- |
| Run failure | A run, trace event, or LLM call has `status == failed`. | Sanitized Errors |
| Capability gap | A required v9 stage is `partial` or `required_but_not_satisfied`. | Stage warnings / capability gaps |
| Normal telemetry | `success`, `not_requested`, or `not_triggered`. | Not displayed as an issue |

`not_triggered` is not a warning. A v9 `partial` stage is never counted as a
failed campaign run. The overview keeps separate counts for failed runs and
stage warnings.

### Stage warning projection

The backend supplies one explicit warning projection per relevant stage:

```text
run_id
campaign_id
question_id
mode
stage_name
status
failure_reason
created_at
```

`failure_reason` is read from the v9 trace payload/error record. It is
sanitized and displayed directly; the client must not reconstruct it from an
empty `error.message` field. The existing `CampaignErrorsResponse` remains
strictly for actual failures.

## Safe failed-run diagnostics

At the campaign execution boundary, every caught exception creates a safe,
structured failure projection:

```text
error_code
safe_error_message
last_completed_stage
provider_status
retry_count
timeout_state
budget_state
```

The values are stored with the run's existing metadata/derived metrics and
exposed through run detail and the Sanitized Errors projection. No raw prompt,
provider response body, stack trace, API key, or secret is persisted. When a
source has no safe message, `safe_error_message` is
`failure_reason_not_recorded`; it must never be an empty string.

## Export behavior

The existing endpoint remains authoritative:

```text
POST /api/evaluation/campaigns/{campaign_id}/export
```

The frontend sends the current checkbox options unchanged. On a successful
response it serializes the returned export object as a JSON Blob, starts a
browser download, and updates the visible run and LLM-call counts from that
same response. No preview endpoint is added.

The action label is **Export JSON**. The filename is:

```text
{campaignId}-redacted.json
```

only when the raw-content options remain disabled. If the user has explicitly
enabled either `include_full_prompts` or
`include_raw_trace_payloads`, the filename is:

```text
{campaignId}-custom.json
```

The existing badge continues to state whether full prompts are included or
redacted. The export response's `redaction` object remains the audit record of
the chosen options.

`include_answers` and `include_retrieved_excerpts` retain their existing
research-export defaults. They change the selected material but do not change
the filename classification; only raw prompts or raw trace payloads do.

## Backend boundaries

- `evaluation/campaign_engine.py` owns safe exception-to-run diagnostics.
- `evaluation/analytics.py` owns separate failure and stage-warning
  projections, including safe reason formatting.
- `evaluation/campaign_schemas.py` owns the explicit response schemas.
- Existing observability tables remain the source of trace/LLM stage data.

## Frontend boundaries

- `src/services/evaluationApi.ts` continues to own the HTTP export call.
- `src/pages/EvaluationCenter.tsx` loads Stage warnings with the existing
  campaign-tab data pattern.
- `src/components/evaluation/AblationDashboardTab.tsx` owns export controls,
  download behavior, preview values, Sanitized Errors, and Stage warnings.

## Error handling

- A failed export leaves the existing preview unchanged and displays the
  request error through the Evaluation Center's established error treatment.
- Download creation always revokes its object URL after clicking the temporary
  anchor.
- Legacy campaigns without the new diagnostic fields return empty warning rows
  and retain the existing safe empty states.

## Acceptance criteria

1. `partial` Graph/Visual events no longer appear in Sanitized Errors.
2. A `required_but_not_satisfied` Graph/Visual stage appears once in Stage
   warnings with its safe failure reason.
3. Every new failed run contains a nonempty safe failure reason and a last
   completed stage, without sensitive data.
4. Clicking Export JSON calls the current endpoint with checkbox options,
   downloads the JSON, and updates preview counts from the response.
5. Existing export redaction behavior and authorization tests continue to
   pass.

## Test strategy

- Backend analytics tests prove failures and warnings are mutually exclusive,
  `not_triggered` is absent, and reasons are mapped correctly.
- Campaign-engine tests prove a caught exception records the safe diagnostic
  projection.
- Export redaction tests keep current data-minimization guarantees.
- Frontend component tests prove the request body, download filename, preview
  update, and warning/error separation.

## Non-goals

- Diagnosing historical failed runs whose original server-side error was never
  persisted.
- Introducing a second export/preview endpoint.
- Treating missing page-image assets or graph provenance as provider failures.
