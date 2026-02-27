# QUALITY_SCORE

## Goal

Track practical quality and reliability signals for backend behavior.

## Suggested Scorecard

| Dimension | Signal | Method |
|---|---|---|
| API stability | response/schema break rate | contract tests + integration checks |
| Retrieval quality | answer grounding and relevance | benchmark samples + RAG eval |
| Streaming health | interrupted or malformed event rate | SSE integration tests |
| Security hygiene | auth gaps and unsafe inputs | security review + static checks |
| Runtime reliability | failure/retry rate | logs + test regression |

## Current State

- Broad endpoint coverage exists in `tests/`.
- Security audit documents exist in `agentlog/audit_20260122/`.

## Next Up

1. Add contract tests for critical request/response schemas.
2. Keep RAG evaluation smoke checks for high-impact changes.
3. Track recurring failure classes and convert them into guardrails.

