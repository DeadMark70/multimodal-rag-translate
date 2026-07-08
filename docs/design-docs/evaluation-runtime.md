# Evaluation Runtime

## Purpose

Describe evaluation as a persisted runtime subsystem that now serves both legacy campaign/result views and the Evaluation Research Dashboard.

## Ownership

- Router: `evaluation/router.py`
- Storage: `evaluation/storage.py` and `evaluation/db.py`
- Engine: `evaluation/campaign_engine.py`
- Read-side analytics: `evaluation/analytics.py`
- Normalized observability recording: `evaluation/observability.py`
- Normalized observability repositories: `evaluation/observability_storage.py`
- Trace and observability contracts: `evaluation/trace_schemas.py`
- Campaign/result schemas: `evaluation/campaign_schemas.py`

## Runtime Rules

- Campaigns persist status and results in SQLite.
- SQLite runs in WAL mode for concurrent campaign work.
- Results, traces, metrics, manual evaluate, cancel, and SSE stream are separate API concerns.
- Model discovery is control-plane behavior; runtime generation stays behind provider/factory seams.

## Research Runtime Model

### Run Identity

- New research APIs use `run_id`.
- Current implementation does not perform a separate identity migration. `run_id` is `campaign_results.id`.
- Legacy APIs still expose `campaign_result_id` where that surface already existed, especially persisted agent traces and RAGAS score rows.

### Write Path

- `CampaignEngine` expands campaign work into question/mode/repeat execution units.
- For ablation campaigns, execution expands into question/repeat/condition units and stores:
  - `condition_id`
  - `condition_label`
  - ablation flags in snapshots and derived metrics
- Each executed unit persists:
  - one `campaign_results` row
  - a minimal root trace span in `evaluation_trace_events`
  - optional normalized observability rows for routing, retrieval, context packing, tool calls, claims, and LLM usage
  - optional legacy `agent_traces` detail for compatibility viewers
- The observability recorder is best-effort by default. Runtime code logs recorder failures and continues unless strict mode is explicitly enabled.

### Snapshot Capture

- The engine captures three snapshot families per run:
  - `question_snapshot`: frozen test-case input, including research metadata
  - `model_config_snapshot`: normalized model preset actually used at runtime
  - `system_version_snapshot`: execution metadata such as mode, repeat, ablation condition, budget, execution profile, and context policy version
- `derived_metrics` is written at execution time for cheap aggregation and UI sorting.
- `final_answer_hash` is stored for answer-change comparisons without requiring raw-answer diffs.

### Normalized Observability

- `evaluation_trace_events` stores generic span/event rows.
- `evaluation_llm_calls` stores token and cost rows with prompt hashes/previews.
- `evaluation_retrieval_events` and `evaluation_retrieval_chunks` store result-level retrieval evidence.
- `evaluation_context_packs` stores pack/drop accounting.
- `evaluation_tool_calls` stores tool activity normalized out of agent traces.
- `evaluation_routing_decisions` stores retrospective routing analysis today.
- `evaluation_claims` stores claim/evidence support rows.
- `evaluation_human_ratings` stores rubric scores keyed by hashed rater identity.

### Event Schema And Ordering

- `EvaluationRunRecorder` emits `event_schema_version="1.0"` on every persisted trace event.
- Sequence numbers are monotonic within one run recorder instance and are used as the primary read order for run traces.
- Opening span rows have:
  - `status="running"`
  - `ended_at=null`
  - `duration_ms=null`
- Closing span rows reuse the same `span_id` and carry terminal status plus computed duration.
- Campaign SSE does not yet mirror the run-trace sequencing model. It remains a coarse status stream with `campaign_snapshot`, `campaign_progress`, and terminal `campaign_*` events.

## Read Path

- `CampaignResultRepository` loads modern rows and also normalizes legacy rows:
  - agentic rows without `execution_profile` are read back as `legacy_shared`
  - rows with no snapshot payloads keep `total_tokens=null` instead of a misleading zero
- `EvaluationAnalyticsService` powers the dashboard-specific read models:
  - campaign overview
  - mode/question comparison
  - cost and latency summaries
  - retrospective router analysis
  - ablation grouping
  - repeat stability
  - human-eval queue and human-vs-auto calibration
  - sanitized error reporting
  - redaction-aware campaign export
  - per-run trace/retrieval/context/LLM/tool/claim/metric views

## Empty-State Contract

- Research endpoints are expected to return partial or empty payloads for older campaigns.
- Absence of normalized observability rows is not treated as a backend error.
- Human calibration endpoints return empty rows and `sample_count=0` when no ratings exist.
- Router analysis is currently retrospective and may return no decision rows.
- Export still succeeds for legacy campaigns; it simply emits less detail.

## Prompt And Error Handling

- The default API surface favors prompt hashes and previews rather than raw prompts.
- Export is the only research endpoint that intentionally exposes prompt-related redaction controls, and full prompts are excluded unless explicitly requested.
- Sanitized errors are surfaced instead of raw provider exception dumps:
  - obvious secrets are redacted
  - traceback-like multiline payloads collapse to a generic redaction message
