# QUALITY_SCORE

## Goal

Track whether backend behavior remains stable, observable, and contract-safe as the code evolves.

## Active Scorecard

| Dimension | What we watch | Current evidence |
|---|---|---|
| API stability | route/schema drift | router and contract tests, `openapi.json` spot checks |
| Retrieval and graph quality | behavior regressions in ask/research/graph flows | focused RAG and GraphRAG pytest coverage |
| Streaming health | malformed or dropped SSE progress | chat/research/evaluation stream tests |
| Persistence safety | document, conversation, and campaign durability | repository, background, and campaign tests |
| Runtime reliability | startup, retries, warmup degradation | app-factory, repository, and evaluation coverage |

## Current State

- Broad backend regression coverage already exists under `tests/`.
- Evaluation, GraphRAG, PDF service, and conversation paths all have focused suites.
- Full acceptance remains `.\.venv\Scripts\python.exe -m pytest`.

## Next Quality Investments

1. Keep generated API docs in sync with router/openapi changes.
2. Expand focused SSE coverage whenever new stream events are added.
3. Keep dependency and startup checks aligned with real runtime imports and warmup behavior.
