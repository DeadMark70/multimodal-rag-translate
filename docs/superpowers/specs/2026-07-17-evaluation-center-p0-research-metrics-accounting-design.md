# Evaluation Center P0 Research Metrics and Accounting Design

**Date:** 2026-07-17

**Status:** Approved design

**Scope:** Evaluation Center research-quality metrics, latency percentiles, token accounting, inference cost, and RAGAS evaluation overhead

## 1. Objective

Make the Evaluation Center safe for research comparison. Every value presented as correctness, faithfulness, relevance, latency percentile, token usage, or cost must come from a compatible measured source. Missing, partial, legacy, or unpriced data must remain visibly incomplete and must never be converted into a synthetic zero or an unlabeled proxy.

The design has four required outcomes:

1. The main dashboard uses official RAGAS scores rather than claim-support proxies.
2. Every LLM call made by an evaluation execution is measured and attributed to an execution phase.
3. Inference cost and RAGAS evaluator overhead are reported separately.
4. Only complete and compatible observations participate in cost-versus-quality ranking or ECR calculations.

## 2. Approved Product Decisions

The following decisions are fixed for this design:

- Use strict quality reporting. A missing or unfinished RAGAS score is `N/A`, `evaluating`, `partial`, or `failed`; it never falls back to a proxy.
- Compare RAG modes using inference cost only. Report RAGAS evaluator token usage and cost in a separate evaluation-overhead section.
- Do not estimate or backfill historical campaigns. Campaigns without the new accounting marker are `incomplete_legacy` for token and cost comparison.
- Replace placeholder latency and token charts in the same P0 change. The dashboard must use measured per-mode p50/p95 values and a reconciled, non-overlapping token breakdown.
- Keep claim-support and evidence heuristics available as explicitly named diagnostic proxies, but do not label them correctness, faithfulness, relevance, citation precision, or RAGAS metrics.

## 3. Current-State Findings

The backend already persists official RAGAS results and exposes `CampaignMetricsResponse`, including metric availability, per-mode summaries, invalid or missing rows, evaluator model, and detailed score rows. The main dashboard does not use this source; it currently derives correctness and faithfulness from unsupported-claim ratios and derives relevance from evidence coverage.

The existing evaluation observability model can persist per-run LLM calls, but campaign execution currently writes one aggregate `campaign_generation` call. The aggregate does not reliably include query expansion, graph reasoning, agent planning, visual verification, or synthesis. RAGAS batch calls also do not have a clean per-run ownership model because one evaluator call may score multiple campaign results.

The current overview contains average latency but not per-mode percentiles. The frontend currently displays that average as both p50 and p95 and places total tokens in the prompt-token series while setting other series to zero.

## 4. Core Invariants

1. A missing metric is never represented as zero.
2. A diagnostic proxy is never presented as an official quality metric.
3. Unknown cost is never represented as `$0.00`.
4. RAGAS evaluator cost is never included in inference-mode efficiency ranking.
5. A batched evaluator call is not divided into invented per-run costs.
6. Failed and superseded execution attempts do not enter benchmark execution cost.
7. Actual retry and failed-attempt spend remains available as operational cost.
8. A campaign is complete for token accounting only when an accounting scope proves instrumentation was active for the official attempt and every observed model response has reconcilable usage.
9. Different research-accounting or evaluator metric versions are never ranked together silently.
10. Frontend presentation code formats backend values but does not synthesize research metrics.

## 5. Chosen Architecture

Use a task-local accounting context and a LangChain callback attached at model construction time.

The callback is always installed but is a no-op when no evaluation accounting context is active. Evaluation workers activate the context for one durable attempt. Nested asynchronous work inherits the context through `ContextVar`, so concurrent campaigns remain isolated. Normal chat and other non-evaluation requests do not create accounting rows.

Pipeline boundaries add explicit phase tags. The callback guarantees that a model call is observed even if a phase tag is absent; such a call is stored under `unclassified` and makes only the phase breakdown partial, not the measured total.

This callback design is preferred over wrapping the model object because the project uses chat-model features such as structured output and may use tool binding or streaming. The callback system remains part of those Runnable executions without changing the model interface.

## 6. Accounting Scope Model

An accounting scope proves that instrumentation was active for a specific durable attempt. Add `evaluation_accounting_scopes` with these fields:

- `scope_id`
- `campaign_id`
- `scope_type`: `execution_run` or `ragas_batch`
- `scope_key`: run ID for execution or a stable batch identity for RAGAS
- `run_id`: nullable; populated only for execution scopes
- `job_id`
- `work_item_id`
- `attempt_id`
- `source_attempt_id`: nullable until an execution attempt becomes official
- `metric_name`: nullable; populated for RAGAS scoring scopes
- `target_result_ids_json`: empty for execution; populated for a RAGAS batch
- `accounting_schema_version`: `2` for this design
- `status`: `running`, `completed`, `failed`, `interrupted`, or `cancelled`
- `observed_call_count`
- `measured_call_count`
- `missing_usage_call_count`
- `unclassified_phase_call_count`
- `started_at`
- `completed_at`
- `created_at`
- `updated_at`

The scope row is created before provider work begins and finalized after the attempt ends. Startup recovery marks a scope left in `running` as `interrupted`. The absence of a version-2 scope for an old campaign is affirmative evidence of `incomplete_legacy`, not evidence of zero usage.

## 7. Usage Ledger

Add `evaluation_usage_events`. One row represents one actual provider model call:

- `usage_event_id`
- `scope_id`
- `campaign_id`
- `scope_type`
- `scope_key`
- `run_id`: nullable
- `job_id`
- `work_item_id`
- `attempt_id`
- `provider_run_id`: nullable callback/provider run identifier
- `phase`
- `purpose`: the logical `get_llm()` purpose
- `metric_name`: nullable
- `provider`
- `model_name`
- `input_tokens`
- `output_text_tokens`
- `reasoning_tokens`
- `other_tokens`
- `reported_total_tokens`
- `raw_usage_json`
- `usage_status`: `measured`, `missing`, or `failed`
- `reconciliation_status`: `balanced`, `partial`, or `unavailable`
- `estimated_cost_usd`
- `estimated_cost_twd`
- `pricing_status`: `priced`, `unknown_model`, `missing_price`, or `unavailable_usage`
- `price_snapshot_id`: nullable
- `latency_ms`
- `status`
- `error_json`
- `created_at`

The usage event is written when a callback finishes or fails. It is not buffered until the whole campaign unit completes. A callback-start registry keyed by provider run ID retains the accounting context and start time until the corresponding end or error callback, then releases it.

Required indexes cover campaign and scope, official run and phase, attempt, and RAGAS metric/batch lookups.

The existing `evaluation_llm_calls` table remains available for legacy trace compatibility. New research analytics use the version-2 accounting scopes and usage ledger. Historical rows are not copied or estimated.

## 8. Token Normalization

Provider adapters normalize raw usage into non-overlapping categories:

- `input_tokens`
- `output_text_tokens`
- `reasoning_tokens`
- `other_tokens`
- `reported_total_tokens`

The adapter must account for provider-specific semantics. If a provider reports reasoning as a subset of completion tokens, output text is completion minus reasoning. If reasoning is reported separately, it remains separate. Cached-input or otherwise unattributed usage is assigned to an explicit non-overlapping category or to `other_tokens`.

The reconciliation invariant is:

`input_tokens + output_text_tokens + reasoning_tokens + other_tokens = reported_total_tokens`

A call is `balanced` only when that equality is established from provider-reported data. Missing metadata is not converted to zero. It produces `usage_status=missing`, and the containing scope becomes partial.

Raw provider usage is retained for auditability. Pricing uses the same normalized event and a versioned price snapshot.

## 9. Phase Attribution

The initial controlled phase vocabulary is:

- `query_expansion`
- `retrieval_rewrite`
- `graph_reasoning`
- `agent_planning`
- `answer_generation`
- `visual_verification`
- `agent_synthesis`
- `ragas_scoring`
- `unclassified`

The central callback records every observed call. Evaluation-reachable pipeline boundaries set a nested phase context. A future model call added without a phase tag remains counted under `unclassified`, making the phase breakdown visibly partial rather than losing the call.

Phase names are data-contract values. Adding or renaming one requires a schema or contract update and frontend compatibility handling.

## 10. Attempt, Retry, and Cost Semantics

Every usage event belongs to a durable attempt.

For execution modes:

- `benchmark_execution_cost` includes only the accounting scope whose attempt is the official successful `source_attempt_id` for the campaign result.
- `operational_execution_cost` includes all actual execution scopes, including retry and failed-attempt calls.
- Superseded successful attempts remain auditable but do not enter the current benchmark comparison.

For RAGAS:

- Every actual evaluator retry contributes to `evaluation_overhead`.
- RAGAS usage is stored at batch scope and is not evenly allocated to target runs.
- The scope records the metric name and target result identifiers for traceability without asserting a per-run evaluator cost.

At-least-once worker execution can still incur an external provider charge immediately before a process crash prevents the callback checkpoint. The system cannot claim exactly-once billing without provider idempotency. An interrupted scope therefore becomes partial and cannot enter strict benchmark cost ranking.

## 11. Quality Metric Semantics

Official dashboard quality fields are sourced only from compatible RAGAS score projections:

- `answer_correctness`
- `faithfulness`
- `answer_relevancy`
- `context_precision` when enabled
- `context_recall` when enabled

Each metric contains:

- value
- status
- valid sample count
- missing sample count
- failed sample count
- evaluator model
- evaluation signature or metric version

Statuses are `complete`, `evaluating`, `partial`, `failed`, or `not_requested`. One missing metric does not erase unrelated successful metrics.

Claim support, unsupported-claim ratio, lexical fact matching, and document-level evidence hit rates remain available only under explicit proxy or diagnostic names.

## 12. Completeness and Comparability

Expose separate campaign and mode statuses:

### 12.1 Quality status

- `complete`
- `evaluating`
- `partial`
- `failed`
- `not_requested`

### 12.2 Token accounting status

- `complete`
- `partial`
- `incomplete_legacy`

### 12.3 Pricing status

- `complete`
- `partial`
- `unknown`

### 12.4 Phase attribution status

- `complete`
- `partial`
- `not_available`

A mode is eligible for cost-versus-quality ranking and ECR only when:

1. The selected RAGAS quality metric is complete and has valid samples.
2. Token accounting is complete for every official included execution attempt.
3. Pricing is complete for every included usage event.
4. Accounting schema versions match.
5. Evaluator model, metric version, and score signature are compatible.

An ineligible mode remains visible with `comparable=false` and structured `not_comparable_reasons`. It is neither plotted nor ranked.

## 13. Latency Statistics

Latency is computed from official successful execution results and grouped per mode. Use `total_latency_ms` when available and the legacy latency field only for legacy detail display, not for strict cross-mode comparison.

Expose:

- arithmetic mean
- p50
- p95
- sample count
- percentile method

Use deterministic nearest-rank percentiles:

`rank = ceil(percentile * sample_count)`

Return the observed value at that one-based rank after sorting. No sample produces `N/A`. A mode with fewer than five observations receives `low_sample_size` but retains its measured values.

## 14. Research Summary API

Add:

`GET /api/evaluation/campaigns/{campaign_id}/research-summary`

The response is a strict typed contract rather than an open dictionary. It contains:

- `campaign_id`
- `research_schema_version`
- campaign quality, token, pricing, and phase-attribution statuses
- per-mode research summaries
- campaign-level RAGAS evaluation overhead
- operational totals
- structured warnings

Each mode summary contains:

- mode and sample counts
- `comparable`
- `not_comparable_reasons`
- RAGAS metric observations
- mean, p50, and p95 latency
- normalized token categories
- token totals by phase
- benchmark execution cost
- operational execution cost
- accounting, pricing, metric, and sample metadata

Evaluation overhead contains RAGAS token categories, total tokens, cost, pricing status, evaluator models, metric names, batch count, retry count, and partial-scope warnings.

The existing metrics, overview, analytics, export, and run-detail endpoints remain available for compatibility and detailed inspection. The main Campaign Overview tab uses only `research-summary` for research claims and charts.

## 15. Frontend Presentation

### 15.1 Status header

Display independent badges for quality, token accounting, pricing, and phase attribution. A legacy campaign shows a prominent `Legacy accounting incomplete` banner.

### 15.2 Quality cards and mode comparison

Display only official RAGAS values under correctness, faithfulness, and relevance labels. `N/A` values include their status and valid, missing, and failed sample counts. The frontend does not calculate a fallback from claim metrics.

### 15.3 Cost versus quality

Plot only comparable modes. List excluded modes below the chart with structured reasons such as legacy accounting, missing RAGAS score, unknown pricing, incomplete scope, or metric-version mismatch.

### 15.4 Latency

Display backend-provided mean, p50, p95, sample count, and percentile method. Surface a low-sample warning when applicable. Never substitute an average for a percentile.

### 15.5 Token breakdown

Use a non-overlapping stack of input, output text, reasoning, and other tokens. Provide a phase breakdown and keep `unclassified` visible. Do not place total tokens in the prompt series.

### 15.6 Cost separation

Display these independently:

- benchmark execution cost
- operational execution cost
- RAGAS evaluation overhead

Unknown cost displays `Unknown`, not `$0.00`. RAGAS cost is not added to the mode-efficiency axis.

## 16. Failure Handling

- Missing usage metadata creates a persisted missing event and a partial scope.
- A callback error must not fail the model call; it records an internal accounting warning and makes the scope partial.
- Persistence failure makes the scope partial and emits a sanitized operational error.
- Provider failure records the failed call and any available usage. Retry accounting remains attached to the correct attempt.
- An interrupted scope is never considered complete after recovery.
- RAGAS partial completion leaves successful metrics visible and missing metrics as `N/A`.
- Frontend request failure clears incompatible campaign data and shows an error rather than retaining stale values from another campaign.

## 17. Module Boundaries

Add or extract focused units:

- `core/llm_usage_context.py`: neutral task-local scope and phase context; no evaluation repository dependency
- `core/llm_usage_callback.py`: LangChain callback, provider usage extraction, and sink protocol
- `evaluation/accounting_schemas.py`: scope, event, status, and research-summary models
- `evaluation/accounting_store.py`: durable scope and usage-event persistence
- `evaluation/token_normalizers.py`: provider-specific non-overlapping token normalization
- `evaluation/research_analytics.py`: strict aggregation, percentile, completeness, and comparability rules

The execution and RAGAS workers activate scopes and phases but do not implement aggregation. The existing campaign analytics remain available while the new research summary becomes the main dashboard contract.

## 18. Migration and Compatibility

1. Create accounting scope and usage event tables additively.
2. Do not backfill existing aggregate LLM rows into measured events.
3. Treat campaigns without version-2 official scopes as `incomplete_legacy`.
4. Preserve existing APIs and frontend detailed-result surfaces during migration.
5. Switch the Campaign Overview tab to the research-summary API only after backend contract and migration tests pass.
6. Keep legacy rows available in trace and export views with explicit legacy metadata.

No destructive table rewrite is required for the initial migration.

## 19. Testing Strategy

### 19.1 Callback and context isolation

- Plain invoke and async invoke produce usage events.
- Structured output, tool binding, and streaming paths preserve accounting where supported by the installed provider.
- Concurrent evaluation tasks never exchange campaign, run, attempt, purpose, or phase context.
- Calls outside an evaluation scope write nothing.
- Callback start state is released on success, failure, cancellation, and interruption.

### 19.2 Token and pricing

- Provider fixtures normalize to non-overlapping categories.
- Balanced totals satisfy the reconciliation invariant.
- Missing usage remains missing rather than zero.
- Unknown price remains unknown rather than zero.
- Price snapshots make historical cost deterministic.

### 19.3 Execution coverage

- Naive, Advanced Multi-Query, Graph, Agentic planning, visual verification, and synthesis calls all enter the ledger.
- Unclassified calls remain counted and make phase attribution partial.
- Official-attempt cost excludes failed and superseded attempts.
- Operational cost includes actual retries and failures.

### 19.4 RAGAS accounting

- A batched evaluator call creates one batch-scope event.
- Batch cost is not divided across campaign results.
- Retry overhead is retained.
- Missing or failed metrics do not become zero.

### 19.5 Research analytics

- Mean, nearest-rank p50, and nearest-rank p95 match fixed fixtures.
- Low sample sizes produce warnings.
- Legacy scopes are not comparable.
- Partial token or pricing data excludes a mode from ranking and ECR.
- Metric model, version, or signature mismatch excludes incompatible rows.
- RAGAS and inference costs remain separate.

### 19.6 Frontend semantics

- Missing RAGAS values display `N/A` and never proxies or zero.
- Unknown cost never displays `$0.00`.
- p50 and p95 come from the API.
- Token stacks use non-overlapping categories.
- Ineligible modes do not appear in the cost-quality plot.
- Legacy, evaluating, partial, failed, unknown-pricing, and low-sample states render distinctly.
- Frontend mappers contain no correctness, faithfulness, relevance, percentile, or token fallback calculations.

### 19.7 End-to-end verification

Run a deterministic fake-provider campaign containing Naive, Advanced Multi-Query, Graph, and Agentic modes. Assert exact call counts, phases, token totals, benchmark cost, operational cost, RAGAS overhead, quality statuses, and comparison eligibility through the API and mounted dashboard.

## 20. Acceptance Criteria

Implementation is complete only when all of the following are demonstrated:

1. The main dashboard displays official RAGAS metrics and no quality proxy under an official metric label.
2. Missing metrics, tokens, and prices remain explicit missing states.
3. Every evaluation LLM call is represented by a measured, missing, or failed usage event.
4. Normal chat behavior is unchanged and creates no evaluation accounting records.
5. Multi-Query, Graph, Agentic, visual verification, and synthesis costs are included in official execution accounting.
6. RAGAS evaluator overhead is measured separately and never changes the inference comparison axis.
7. Old campaigns are visibly legacy-incomplete and excluded from strict cost ranking.
8. Backend p50 and p95 are deterministic per-mode measurements and the frontend does not synthesize them.
9. Retry accounting distinguishes benchmark and operational cost.
10. Incompatible or partial modes remain visible with reasons but do not enter cost-quality ranking or ECR.
11. Focused accounting, analytics, API, migration, and frontend tests pass.
12. Full backend and frontend verification is run, with environment-limited failures reported exactly.

## 21. Out of Scope

- Backfilling or estimating historical token and cost data
- Changing the semantic definitions implemented by RAGAS
- Allocating one RAGAS batch cost to individual results
- Provider-side exactly-once billing guarantees
- Redesigning retrieval evidence instrumentation beyond truthful proxy labeling
- Replacing SQLite or adding distributed workers
- General-purpose billing for non-evaluation chat requests
