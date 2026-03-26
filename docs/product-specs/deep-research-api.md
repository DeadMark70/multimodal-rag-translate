# Deep Research API

## User Outcomes

- Ask ordinary questions synchronously or through streamed progress.
- Plan and execute multi-step Deep Research.
- Request one-shot research answers where the backend owns the full plan/execute cycle.

## Acceptance Notes

- `/rag/ask` and `/rag/ask/stream` should stay behaviorally aligned.
- Planning and execution endpoints should keep their response contracts explicit.
- Long-running execution should surface incremental progress through dedicated stream/event paths rather than ad hoc logging only.
