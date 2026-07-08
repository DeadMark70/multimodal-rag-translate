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
  - current implementation reports `analysis_type="retrospective"`
- `GET /api/evaluation/campaigns/{campaign_id}/ablation`
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
  - normalized all-in-one dump for one run under one campaign

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
  - `include_raw_trace_payloads`
  - `include_prompt_previews`
  - `include_full_prompts`
  - `include_answers`
  - `include_retrieved_excerpts`
  - `format` (`json`)
- Export guarantees:
  - trace payloads are blank unless explicitly requested
  - prompt previews can be suppressed
  - `payload.full_prompt` is removed unless explicitly requested
  - answers and retrieval excerpts can be independently removed
  - question snapshots are still partially redacted even when exporting runs

## Compatibility And Empty States

- Legacy campaign APIs remain available for existing consumers.
- Older campaigns with no normalized research rows are still valid:
  - aggregate endpoints return empty or partial summaries
  - run detail endpoints for existing owned runs return empty collections instead of failing when normalized research rows are absent
  - human calibration returns `sample_count=0` when there are no paired samples
- Legacy `agentic` runs without an explicit `execution_profile` are normalized to `legacy_shared`.
- Actual router execution is still disabled by default. Submitting a campaign with `router` mode without the execution flag returns `400` and callers should use retrospective router analysis instead.
