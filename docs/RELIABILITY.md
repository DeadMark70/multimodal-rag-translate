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

## Operational Checks

- `.\.venv\Scripts\python.exe -m pytest`
- `ruff check .` when the environment is scoped to production code
- spot-check `openapi.json` when endpoint families or schemas change
