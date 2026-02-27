# RELIABILITY

## Reliability Objectives

1. Keep long-running tasks observable and recoverable.
2. Prevent known failure classes in parsing, retrieval, and streaming.
3. Keep behavior deterministic enough for debugging and replay.

## Main Reliability Mechanisms

- FastAPI lifespan initialization for core dependencies.
- SSE streaming paths for deep research execution updates.
- Protected endpoints with token-based auth dependency.
- Test suites covering planner/evaluator/research and API contracts.
- Modular router boundaries to isolate failures.

## Error Classes

1. Provider/network errors
2. Contract/format errors
3. Storage/indexing errors
4. User cancellation/interruption

## Recovery Strategy

- Return explicit error details with actionable messages.
- Preserve stable persisted artifacts and conversation history.
- Allow retriable operations where safe.
- Keep router-level failures isolated from unrelated modules.

## Operational Checks

- `python -m pytest`
- `ruff check .` (when configured in environment)
- targeted router/service tests for changed scope

