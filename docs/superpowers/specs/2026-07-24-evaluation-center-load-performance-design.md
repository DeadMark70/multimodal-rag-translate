# Evaluation Center Load Performance Design

**Status:** Proposed

## Goal

Make Evaluation Center load promptly for large historical campaigns while preserving all benchmark, research, trace, and observability capabilities.

## Product Decisions

- A missing `benchmark_id` is normal during the current rollout.
- A campaign without a `benchmark_id` is not an error and remains fully usable for ordinary evaluation analysis.
- Release Metrics is unavailable for such a campaign. The UI must show a clear “not applicable / benchmark not configured” state instead of an error, spinner, or empty failed report.
- When a `benchmark_id` is added later, Release Metrics must remain available without data migration.

## Current Failure Mode

The Evaluation Center requests both the research summary and Release Metrics when a campaign is selected. `ReleaseMetricsService.get_report()` loads each result and then invokes per-run accounting, detail, and graph-observability reads. For the current 160-result campaign, this took approximately 12.6 seconds on a copy of the production evaluation database, despite the report ultimately being marked non-comparable because `benchmark_id` is absent.

The current database also contains large payloads that must remain detail-only:

- 561 campaign results, with the largest campaign holding 160 results.
- About 2.8 MB of result answer/context/ground-truth payload for that campaign; one answer is about 1.54 MB.
- 274 persisted agent traces totaling about 7.1 MB; the largest campaign has 80 traces totaling about 1.9 MB.

## Scope

### P0 — Remove the initial-load blocker

1. Add an explicit `not_applicable` availability state to `ReleaseMetricsReport` (or a backwards-compatible equivalent response field).
2. In `ReleaseMetricsService.get_report()`, validate the anchor campaign’s `benchmark_id` before listing campaigns, results, scores, accounting, traces, or observability.
3. For a missing ID, return the normal not-applicable report with a stable reason code such as `benchmark_not_configured`.
4. In Evaluation Center, request Release Metrics only for campaigns with a `benchmark_id`; otherwise render the not-applicable state locally. Keep the backend guard as the authoritative protection for direct API clients.
5. Preserve the existing fallback for older deployments where the endpoint is unavailable.

### P1 — Make Release Metrics campaign-scoped rather than run-scoped

1. Introduce a read model that loads only the campaign-result fields required for release derivation. It must exclude answer text, retrieval contexts, ground truth, and unneeded JSON snapshots.
2. Replace the per-result `_run_tokens()` calls with one accounting snapshot per selected campaign: load scopes, targets, and usage events once, then group by official run ID.
3. Replace each `_analytics.run_detail()` call with a campaign-level release-observability snapshot. It must bulk-load only V9 materializations, evidence packets, slot resolutions, claims, context packs, and graph events used by release gates.
4. Build `ReleaseRun` instances purely from these grouped snapshots. No nested service call may open a connection for each result.
5. Cache terminal release reports using the complete set of participating campaign IDs and `updated_at` markers. Never cache running campaigns.

### P1 — Keep research analytics bounded

1. Add a dedicated research-result projection for `get_summary`, `get_question_comparison`, and `get_agent_behavior`.
2. Include only fields used by their calculations: identity, mode, run/repeat metadata, status, category/difficulty, latency, token/accounting references, result snapshots required by compatibility logic, and bounded previews where displayed.
3. Do not load `answer`, `contexts_json`, `ground_truth`, or source-content payloads for these endpoints.
4. Keep full result loading restricted to explicit result detail and export endpoints.

### P1 — Make agent-trace lists summary-first

1. Add an `agent_traces` index on `(campaign_id, user_id, created_at DESC)`.
2. Persist a bounded trace summary at trace-write time, containing the fields needed by `AgentTraceSummary`.
3. Make campaign trace lists select the summary projection only. Continue loading and parsing `trace_json` only for a selected trace detail.
4. Preserve a safe legacy fallback for existing rows whose summary has not yet been backfilled; it may parse one row only when requested, never all campaign traces during a list read.

### P2 — Eliminate repeated SQLite setup and scope-target N+1 reads

1. Replace `list_campaign_scopes()` plus one `_load_scope_targets()` query per scope with one joined or two-query bulk loader that groups targets by scope ID.
2. Move `PRAGMA journal_mode=WAL` to database initialization/migration. Per-connection setup retains only pragmas that are connection-local and needed at runtime, such as `busy_timeout` and `foreign_keys`.
3. Maintain WAL, busy timeout, foreign-key enforcement, existing transaction semantics, and database-path test isolation.

### P2 — Bound pathological persisted payloads

1. Add an explicit maximum size for persisted campaign answers, with a safe response policy that does not silently corrupt benchmark results.
2. Record a structured truncation/oversize status when a provider output exceeds the limit.
3. Evaluate compression or separate detail storage only if real retained payload volume remains material after bounded projections are deployed; this is deliberately not part of the first migration.

## Data Flow

```text
Campaign selection
  -> list campaigns (existing lightweight list)
  -> research summary (bounded research-result projection)
  -> benchmark_id present?
       no  -> release metrics = not_applicable, no expensive release reads
       yes -> release snapshot: one bulk read set per campaign
                -> in-memory grouping -> release report

Tab selection
  -> only the selected tab's bounded aggregate or selected-run detail
  -> trace list uses summary rows; trace JSON is loaded only for a selected run
```

## API and Compatibility

- Existing routes remain stable.
- `GET /api/evaluation/campaigns/{campaign_id}/release-metrics` continues returning HTTP 200 when `benchmark_id` is absent; it carries an explicit not-applicable state instead of a failed/comparable report.
- Existing full result, trace detail, export, and selected-run observability routes retain their payload contracts.
- Existing historical trace rows remain readable. Summary storage is additive and uses lazy fallback/backfill rather than destructive migration.

## Error Handling and Cache Rules

- Missing `benchmark_id` is a deterministic product state, never logged as a server error.
- Database failures retain the current error envelope; they are distinct from not-applicable Release Metrics.
- Terminal-cache keys include every campaign contributing to a benchmark report. A changed `updated_at` invalidates the report.
- Running/evaluating campaigns do not use terminal aggregation caches.

## Verification and Acceptance Criteria

1. A campaign without `benchmark_id` returns a not-applicable release response without invoking result, score, accounting, trace, or observability repositories.
2. Evaluation Center does not issue a Release Metrics HTTP request for a selected campaign without `benchmark_id`.
3. For a 160-result fixture, Release Metrics issues a bounded number of repository queries per campaign, not a number proportional to runs; instrumented query count must remain constant as run count grows.
4. Release Metrics values and release-gate decisions match the existing implementation on an instrumented benchmark fixture.
5. Research summary and question comparison produce unchanged public responses while their database projection excludes full answers, contexts, and ground truth.
6. Trace-list query plan uses the new campaign/user/created index and does not select `trace_json` for list responses.
7. Accounting scope loading executes a constant number of SQL statements per campaign.
8. Existing SQLite lifecycle tests continue to verify WAL, busy timeout, foreign keys, migrations, and test-specific database paths.
9. Add response-size and oversized-answer regression tests; no silent answer truncation is allowed.

## Rollout Order

1. P0 backend guard and frontend no-request behavior.
2. P1 release bulk snapshot, with parity tests before enabling terminal caching.
3. P1 research projection and trace-summary migration/index.
4. P2 accounting bulk loading and connection initialization cleanup.
5. P2 answer size policy after collecting post-P1 payload measurements.

## Out of Scope

- Changing evaluation scoring, benchmark semantics, or Release Metrics gate rules.
- Deleting historical campaigns, answers, or traces.
- Migrating from SQLite to another database.
- Broad frontend redesign unrelated to data loading.
