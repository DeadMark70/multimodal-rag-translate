# Backend Docs Index

Format direction:
- [OpenAI Harness Engineering](https://openai.com/zh-Hant/index/harness-engineering/)

## Read Order

1. `docs/BACKEND.md`
2. `docs/generated/api-surface.md`
3. `docs/design-docs/index.md`
4. `docs/product-specs/index.md`
5. `docs/exec-plans/index.md`
6. `docs/references/index.md`

## Current API Surface

- Document ingestion, OCR, translation, file retrieval, summary, retry-index, and deletion
- Ordinary ask, streamed ask, Deep Research planning/execution, and research answer flows
- Graph status, graph data, optimize/rebuild/full rebuild, per-document retry, orphan purge, and document inventory
- Evaluation test cases, model presets, campaigns, results, traces, metrics, manual evaluate, cancel, and SSE streaming
- Conversations, dashboard stats, and multimodal/image processing support routes

## Top-Level Guides

- `docs/DESIGN.md`: durable backend architecture decisions
- `docs/BACKEND.md`: current runtime and API map
- `docs/PRODUCT_SENSE.md`: user-visible API capabilities and boundaries
- `docs/RELIABILITY.md`: lifecycle, retries, persistence, and streaming guardrails
- `docs/SECURITY.md`: auth, UUID/path safety, and deployment hardening
- `docs/QUALITY_SCORE.md`: verification surface and quality signals
- `docs/PLANS.md`: plan naming and history maintenance rules

## Deep Dives

- Design docs: `docs/design-docs/index.md`
- Product specs: `docs/product-specs/index.md`
- Generated inventories: `docs/generated/api-surface.md`
- Execution plans: `docs/exec-plans/index.md`
- References and archives: `docs/references/index.md`

## Maintenance Rule

1. If a router prefix, endpoint family, persistence contract, or startup/runtime rule changes, update `docs/BACKEND.md` and `docs/generated/api-surface.md` in the same change set.
2. If the change alters user-visible behavior, update the matching `docs/product-specs/*` doc before moving work to completed plans.
3. Treat `agent.md`, `checklist/`, `agentlog/`, and `conductor/` as references or history, not the primary current-state source.
