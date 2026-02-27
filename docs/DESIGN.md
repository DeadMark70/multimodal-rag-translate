# DESIGN

## Intent

Document backend architecture decisions, API boundaries, and reliability contracts.

## Core Decisions

1. Keep app assembly in factory (`core/app_factory.py`) and business logic in modules.
2. Keep module boundaries explicit (`pdfserviceMD`, `data_base`, `graph_rag`, `multimodal_rag`, `conversations`, `stats`).
3. Keep auth dependency explicit on protected routers.
4. Keep long-running workflows observable through SSE and logs.
5. Keep retrieval and evaluation behavior compatible with existing tests and docs.

## Pointers

- Service overview: `docs/BACKEND.md`
- Existing stack notes: `conductor/tech-stack.md`
- Existing workflow rules: `conductor/workflow.md`
- Legacy implementation guide: `agent.md`
