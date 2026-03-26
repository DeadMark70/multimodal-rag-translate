# Evaluation API

## User Outcomes

- Manage test cases and model presets.
- Create and list campaigns.
- Fetch persisted results, traces, and metrics.
- Trigger manual evaluate and cancellation.
- Reconnect to campaign progress through SSE.

## Acceptance Notes

- Results and traces stay on separate endpoints to avoid oversized payloads.
- Campaign progress recovery must be keyed by persisted campaign state, not in-memory only state.
- Evaluation model discovery and runtime generation should remain architecturally separated.
