# Evaluation PostgreSQL and scoped reads

Implemented opt-in PostgreSQL via `EVALUATION_DATABASE_URL`, preserving the SQLite deployment path. Source IDs, JSON text and timestamps remain unchanged. Explicit SQL migrations run through deployment tooling; runtime startup checks their versions.

Single-run observability/accounting now filters by run ID. Three research analysis pages use revision-checked persistent summaries with background refresh. Campaign/run/question/behavior lists have bounded pages; job lists aggregate counts in one query. Worker recovery uses stale heartbeats and shutdown only recovers attempts owned by that process.

The user's latest 667,938,816-byte snapshot was opened read-only. All 25 tables passed full row-count/content-hash checks after import; all 172 campaigns produced 516 analysis pages without model calls. Three large campaigns retained identical run/summary results. Local cached summaries were 0.5–2.4 ms; single-run reads 16–54 ms, compared with 113–375 ms for the old full-campaign path. These are local function timings, not server/browser latency.

Validation covers existing SQLite/API contracts, PostgreSQL research/job/accounting/worker contracts, cache invalidation/rollback/ownership/concurrent updates, parallel claims, import rollback/non-overwrite, HTTP pagination, frontend integration and build. See [deployment guide](../docs/evaluation-postgresql.md). Server migration, Docker execution and production latency measurement remain deployment steps.
