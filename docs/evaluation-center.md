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
