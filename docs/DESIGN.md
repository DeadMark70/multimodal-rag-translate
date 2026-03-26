# DESIGN

## Intent

Capture stable backend architecture decisions and boundaries.

## Core Decisions

1. Keep app assembly in `core/app_factory.py`; keep business logic in modules and services.
2. Router modules must not import other router modules; shared behavior belongs in service/helper seams.
3. Keep auth, upload validation, and provider selection centralized in `core/`.
4. Treat generated API inventories and product specs as companion docs, not as part of one oversized overview file.
5. Preserve explicit request/response contracts and persistence seams when refactoring internals.

## Deep Dives

- `docs/design-docs/runtime-and-router-boundaries.md`
- `docs/design-docs/retrieval-and-indexing.md`
- `docs/design-docs/conversation-persistence.md`
- `docs/design-docs/evaluation-runtime.md`
