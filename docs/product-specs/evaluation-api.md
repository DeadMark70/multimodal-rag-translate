# Evaluation API

## User Outcomes

- Manage test cases and model presets.
- Create and list campaigns.
- Fetch persisted results, traces, and metrics.
- Trigger manual evaluate and cancellation.
- Reconnect to campaign progress through SSE.
- Inspect run-level research observability without downloading oversized campaign payloads.
- Compare modes, questions, costs, ablations, and human-vs-auto calibration from dedicated analytics endpoints.
- Export campaign research data with explicit prompt and trace redaction controls.

## Acceptance Notes

- Results and traces stay on separate endpoints to avoid oversized payloads.
- Campaign progress recovery must be keyed by persisted campaign state, not in-memory only state.
- Evaluation model discovery and runtime generation should remain architecturally separated.
- Evaluation model discovery (`GET /api/evaluation/models`) must require bearer authentication (no anonymous discovery path).

## Research API Surface

### Campaign Aggregates

- `GET /api/evaluation/campaigns/{campaign_id}/overview`
  - summary counts, token totals, cost rollups, and average latency
- `GET /api/evaluation/campaigns/{campaign_id}/runs`
  - lightweight run list keyed by `run_id`
- `GET /api/evaluation/campaigns/{campaign_id}/mode-comparison`
- `GET /api/evaluation/campaigns/{campaign_id}/question-comparison`
- `GET /api/evaluation/campaigns/{campaign_id}/cost-latency`
- `GET /api/evaluation/campaigns/{campaign_id}/router-analysis`
  - returns a typed retrospective-only `RouterAnalysisRow` projection
  - only persisted decisions with `analysis_type="retrospective"` are included;
    actual router decisions remain durable run telemetry but are excluded from
    campaign aggregates
  - response rows allow-list routing provenance fields and never expose raw
    persisted `payload`
- `GET /api/evaluation/campaigns/{campaign_id}/ablation`
  - legacy condition counts plus `summaries.condition_comparison` when two or more persisted condition IDs exist
- `GET /api/evaluation/campaigns/{campaign_id}/repeat-stability`
- `GET /api/evaluation/campaigns/{campaign_id}/human-vs-auto`
- `GET /api/evaluation/campaigns/{campaign_id}/human-eval-queue`
- `GET /api/evaluation/campaigns/{campaign_id}/errors`
- `POST /api/evaluation/campaigns/{campaign_id}/export`

### Run Detail

- `GET /api/evaluation/runs/{run_id}/trace`
- `GET /api/evaluation/runs/{run_id}/retrieval`
- `GET /api/evaluation/runs/{run_id}/context`
- `GET /api/evaluation/runs/{run_id}/llm-calls`
- `GET /api/evaluation/runs/{run_id}/tools`
- `GET /api/evaluation/runs/{run_id}/visual`
- `GET /api/evaluation/runs/{run_id}/graph`
- `GET /api/evaluation/runs/{run_id}/claims`
- `GET /api/evaluation/runs/{run_id}/metrics`
- `GET /api/evaluation/runs/{run_id}/diff?baseline_run_id=...`
- `POST /api/evaluation/runs/{run_id}/human-ratings`
- `GET /api/evaluation/campaigns/{campaign_id}/runs/{run_id}/observability`
  - canonical safe, normalized all-in-one projection for one owned run under one campaign
  - includes strict accounting diagnostics, normalized trace/retrieval/graph/claim/human-rating data, and optional materialized Agentic v9 observability
  - interactive output redacts raw payloads, provider errors, and captured prompt content

## Run Snapshot Contract

- Research surfaces treat `campaign_results.id` as `run_id`.
- Returned run rows now include:
  - `question_version`
  - `request_id`
  - `started_at`
  - `completed_at`
  - `total_latency_ms`
  - `total_tokens`
  - `question_snapshot`
  - `model_config_snapshot`
  - `system_version_snapshot`
  - `derived_metrics`
  - `final_answer_hash`
- `repeat_number` is returned separately from stored `run_number` so repeated and ablation-expanded campaigns can render correctly.
- `derived_metrics` is intentionally sparse and numeric-first; dashboards should not assume every metric exists on every run.
- `total_tokens` is `null` when a persisted run has no known token total; clients must not treat it as zero.

### Condition Comparison Contract

- Condition comparison is a server-side projection keyed by persisted `condition_id`; labels and `ablation_flags` come from the run snapshot, not current environment variables.
- Condition rows report completed/failed executions, finite RAGAS means and validity counts for `answer_correctness`, `faithfulness`, and `answer_relevancy`, plus mean tokens and latency. Missing/non-finite metrics are `null` with a missing count.
- The paired projection uses `(question_id, repeat_number)` and the configured baseline/guided ordering. A pair is counted for a metric only when both runs completed and both values are finite; `excluded_pairs` records failed, unpaired, and missing-metric reasons.
- Campaigns with fewer than two recorded conditions retain the existing generic Ablation summaries without a condition comparison section.

## SSE Contract

- `GET /api/evaluation/campaigns/{campaign_id}/stream` emits:
  - `campaign_snapshot`
  - `campaign_progress`
  - one terminal event: `campaign_completed`, `campaign_failed`, or `campaign_cancelled`
- `campaign_snapshot` and terminal events serialize `CampaignStatus`.
- `campaign_progress` serializes `CampaignProgressEvent`.
- Current implementation detail: event versioning and monotonic sequencing are available on persisted run trace rows, not on the campaign SSE envelope itself.
- Run trace rows exposed by trace/observability APIs carry:
  - `event_schema_version` (`"1.0"`)
  - `sequence` (monotonic per run)
  - `duration_ms=null` while a span is still open

## Prompt, Export, And Redaction Policy

- Default result and research APIs do not expose full prompts.
- LLM-call detail rows expose prompt metadata through:
  - `prompt_hash`
  - `prompt_preview`
  - optional payload fields when instrumentation provides them
- `POST /api/evaluation/campaigns/{campaign_id}/export` accepts:
  - `include_run_observability` (defaults to `false`; controls export scope)
  - `include_raw_trace_payloads`
  - `include_prompt_previews`
  - `include_full_prompts`
  - `include_answers`
  - `include_retrieved_excerpts`
  - `format` (`json`)
- Export guarantees:
  - the response is either one complete Schema v2 artifact or a non-2xx error; required-section failures never return a partial artifact
  - campaign ownership and the complete result set are loaded before section composition
  - every result has official finite-only RAGAS, accounting, and measured latency projections
  - trace payloads are blank unless explicitly requested and are credential-sanitized when included
  - prompt previews, answers, and retrieval excerpts can be independently suppressed
  - full prompts require both an explicit request and execution-time capture

### Export Schema v2 Contract

The typed v2 contract is owned by `evaluation/export_schemas.py`, and
`EvaluationExportService.export_campaign` is its only composer. The export
route now serves only Schema v2; the legacy response composer and schemas have
been removed.

Schema v2 has exactly five required top-level keys: `schema_version` (`"2.0"`),
`export_metadata`, `campaign`, `sections`, and `runs`. The seven required named
sections are `overview`, `question_analysis`, `agent_behavior`,
`router_analysis`, `ablation`, `human_evaluation`, and `diagnostics`. Every
section pairs typed data with an explicit availability status: `complete`,
`partial`, `not_instrumented`, `not_available`, or `not_applicable`.

The default request is a summary export. Detailed per-run observability is
included only when `include_run_observability=true`; this scope flag is
independent of the five content controls. Each run always has a fixed redacted
result, finite-only RAGAS metrics, typed token accounting, and nullable measured
latency/timestamps. Detailed run rows use fixed allow lists for trace, LLM,
retrieval, context, tool, routing, graph, claim, rating, and evidence-coverage
families. Summary exports do not load the campaign observability snapshot;
full exports load it once in bulk and require its run IDs to equal the campaign
result IDs exactly. Event arrays are complete rather than dashboard-truncated.

Content controls never relax permanent exclusions. Provider bodies,
credentials, authorization headers, stack traces, unrestricted errors, and
non-trace arbitrary payloads are never exported. Raw trace authorization
applies only to sanitized trace-event payloads. Prompt previews follow
`include_prompt_previews`; a full prompt additionally requires both
`include_full_prompts=true` and execution-time capture. Answers and retrieved
excerpts are independently nullable under their respective controls.

## Compatibility And Empty States

- Legacy campaign APIs remain available for existing consumers.
- Older campaigns with no normalized research rows are still valid:
  - aggregate endpoints return empty or partial summaries
  - run detail endpoints for existing owned runs return empty collections instead of failing when normalized research rows are absent
  - human calibration returns `sample_count=0` when there are no paired samples
- Legacy `agentic` runs without an explicit `execution_profile` are normalized to `legacy_shared`.
- Actual router execution is still disabled by default. Submitting a campaign with `router` mode without the execution flag returns `400` and callers should use retrospective router analysis instead.
