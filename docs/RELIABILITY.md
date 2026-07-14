# RELIABILITY

## Reliability Objectives

1. Keep long-running tasks observable, recoverable, and restart-safe.
2. Keep startup degradation explicit when external providers are unavailable.
3. Prevent streaming, persistence, and indexing failures from collapsing into opaque 500s.

## Main Mechanisms

- FastAPI lifespan initialization for env, providers, Supabase, evaluation DB, RAG warmup, and OCR warmup
- Request-id middleware plus standard error envelopes
- Typed repository and background-task error handling
- Authenticated SSE for evaluation and research execution
- SQLite WAL mode for evaluation campaign persistence
- Bounded analytics projections and bulk observability reads so evaluation latency does not grow with raw answer/context payloads or run count
- Terminal-campaign context reuse is guarded by lifecycle status and an updated-at marker; running campaigns continue to use live reads
- Compatibility fallbacks for legacy metadata reads while new writes stay canonical

## Known Failure Classes

1. Provider or network failures
2. Validation / contract errors
3. Storage or indexing failures
4. User cancellation and interrupted streams
5. Startup warmup degradation in test or limited environments

## Recovery Rules

- Prefer structured `AppError` responses over raw exceptions.
- Preserve durable artifacts and campaign/conversation state whenever a later stage fails.
- Surface skipped/fake-provider startup explicitly in logs rather than pretending warmups succeeded.
- Keep graph/document maintenance retryable through dedicated endpoints instead of hidden background retries.

## Evaluation Recovery Contract

Evaluation execution is backed by a SQLite WAL ledger. Claims, attempts, retries, heartbeats, cancellation, and official-result promotion are durable and safe to repeat. A process restart converts running claims to interrupted work; it does not delete attempt history. Provider failures are classified into retryable transport/timeout/rate-limit/server errors and permanent authentication/configuration/data errors. Failed and missing metrics are excluded from statistical aggregates rather than coerced to zero. The UI exposes job and attempt history, safe error messages, partial completion (`completed_with_errors`), and retry/cancel actions.

## Operational Checks

- `.\.venv\Scripts\python.exe -m pytest`
- `ruff check .` when the environment is scoped to production code
- spot-check `openapi.json` when endpoint families or schemas change

# Graph Evidence Locator Reliability

Graph query readers load `current.json` when a versioned snapshot exists and fall back to the legacy root-layout graph when it does not. Writers build a complete temporary snapshot, validate required graph files, then atomically replace the pointer. A failed extraction or rebuild therefore leaves the prior readable snapshot in place.

Legacy edges without provenance remain readable for `graph_raw_current`, but are marked missing and cannot enter final evidence locator context.

The regression smoke covers legacy-store loading, raw-context rollback, locator rejection of unresolved legacy anchors, source-backed graph expansion, and snapshot promotion with provenance, asset, alias, and ready node-vector sidecars.
