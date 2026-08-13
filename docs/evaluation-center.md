# Evaluation Center data contract

The Evaluation Center is a token-only research surface. It reports durable
quality, latency, retrieval, trace, and token observations; it does not use
USD pricing or monetary fallbacks to make a comparison appear complete.

## Availability states

Every nullable research field must preserve the distinction between a measured
zero and an unknown value:

| State | Meaning | UI rule |
| --- | --- | --- |
| `complete` | Required source records were present and passed reconciliation. | Render the measured value, including a real zero. |
| `partial` | Some source records exist, but the required set is incomplete or not fully reconciled. | Render the status and `N/A` for derived values that require the complete set. |
| `not_available` | The value is applicable in principle, but no compatible source value is available. | Render `N/A`. |
| `not_instrumented` | The run or execution path did not record the required telemetry. | Render `N/A` with the instrumentation reason. |
| `not_applicable` | The metric does not apply to this mode or run. | Render `N/A` with the applicability reason. |

Legacy runs are never backfilled by guessing from answer text, empty arrays,
claim ratios, lifecycle events, or generic result totals. A missing value is
not a measured zero.

## Evaluation Setup runtime contract

During evaluation execution, the selected Evaluation Setup model and thinking
controls are authoritative for every mode, including GraphRAG subcalls. When
`thinking_mode` is disabled, no nested GraphRAG override may add
`thinking_budget` or `thinking_level`. When it is enabled, Gemini 2.5-family
models use only `thinking_budget`, while Gemini 3-family models use only
`thinking_level`. Graph-specific defaults apply only to non-evaluation calls
that have no active Setup runtime context.

## Panel data sources

| Panel | Route / projection | Canonical fields |
| --- | --- | --- |
| Campaign Overview | `GET /api/evaluation/campaigns/{campaign_id}/research-summary` | Official campaign RAGAS observations, strict token accounting, latency, and warnings |
| Question Analysis | `GET /api/evaluation/campaigns/{campaign_id}/research-question-comparison` | Per-question/per-mode RAGAS, measured latency, and complete token comparisons |
| Run Trace | `GET /api/evaluation/campaigns/{campaign_id}/runs/{run_id}/observability` | Selected-run summary, lifecycle-folded trace events, and accounting diagnostics |
| Retrieval Evidence | Selected-run observability projection | Nullable retrieval scores/flags, `evidence_coverage_status`, and explicit GraphRAG status/events/evidence |
| Agent Behavior | `GET /api/evaluation/campaigns/{campaign_id}/agent-behavior` | Bulk trace aggregation per run, durable RAGAS values, and strict token status |
| Claim Evidence | Selected-run observability projection | Persisted claim extraction only; absent extraction remains unavailable |
| Router Lab | `GET /api/evaluation/campaigns/{campaign_id}/router-analysis` | Retrospective decisions; actual route outcomes are unavailable unless actual router runs exist |

The legacy `question-comparison` endpoint remains available for compatibility,
but the Evaluation Center uses the typed `research-question-comparison`
projection. Generic legacy analytics must not be interpreted as a complete
research comparison.

## Retrieval and claim provenance

Run observability exposes retrieval chunks through a safe typed projection. Each
row includes `provenance` and `availability`; raw persistence payloads are never
returned. Result-context reconstruction is `derived` with `partial`
availability and a `result_context_reconstruction` reason. Its answer-use flag
is heuristic, not a measured retriever fact. Only normalized retriever
instrumentation may report `measured`/`complete`; older rows are
`not_available` with `provenance_not_recorded`, and their retrieval booleans are
null rather than false.

Claim projections return only evidence ID, document ID, chunk ID, and integer
page references, plus scalar repair fields. Provider bodies and original claim
payloads are excluded. Run-level `claim_extraction_status` distinguishes a
recorded empty extraction from a run that did not instrument claims.

## Export v2 parity

`POST /api/evaluation/campaigns/{campaign_id}/export` returns one authenticated,
all-or-error snapshot. The seven named `sections` are the same serialized
objects returned by the active panel services. With safe-default content flags,
each full-export run observability object is the same safe projection returned
by the selected-run observability endpoint. Summary export is the default;
`include_run_observability=true` adds every campaign result ID using one
campaign observability snapshot and one accounting snapshot, independent of run
count. A campaign without a benchmark retains the release endpoint's empty
manifest/statistics objects and nested `not_applicable` state.

Release verification must cover an authenticated two-mode campaign with
official RAGAS and accounting, v9 evidence, routing, ablation, a human rating,
an error, and a stage warning. Also verify a v8/legacy run, missing
instrumentation, a failed or partial run when available, a terminal durable job,
and a multi-run full export. Confirm foreign ownership returns `404`, failures
never return a partial v2 body, and exported run IDs exactly equal campaign
result IDs. Record only response shapes and statuses; never copy secrets into
verification notes.

## Comparison rules

- `naive` is the quality baseline when the question has the required compatible
  quality observations in both compared modes. Quality deltas do not require
  token accounting, but token-derived comparisons do.
- Token deltas and ECR are `N/A` when either side has partial or unavailable
  accounting.
- Best quality mode is chosen from complete RAGAS quality observations using
  correctness and faithfulness; a complete token count is used as a tie-break
  only when both tied candidates have one, followed by mode name.
- Retrospective router rows describe recorded decisions, not actual router
  executions. Saved tokens, quality loss/gain, and regret are `N/A` without
  actual router-run data.
- A GraphRAG mode label alone is not proof that traversal occurred. The selected
  run must contain `graph_events`/`graph_evidence_items`; otherwise the UI shows
  `not_instrumented` (or an explicit `fallback` reason).
- A pair of `running` and terminal trace rows for the same span is one lifecycle,
  not two executions. The UI folds that pair by default and preserves the raw
  rows behind the lifecycle disclosure.
- Router retrospective rows carry `question_id`, `run_id`, and `repeat_number`
  so repeated questions and modes remain distinguishable.
