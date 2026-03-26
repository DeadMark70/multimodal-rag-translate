# PRODUCT_SENSE

## Purpose

Describe the backend capabilities that users and operators depend on through the frontend and API clients.

## User-Visible Capabilities

1. Upload PDFs, track processing, download original or translated files, retry indexing, and request summaries.
2. Ask ordinary RAG questions or run multi-step Deep Research through synchronous or streamed endpoints.
3. Build and maintain GraphRAG state, including full rebuilds, per-document retries, and orphan purge.
4. Persist conversations and messages so the frontend can restore work.
5. Run evaluation campaigns with saved model presets, persisted results, metrics, and agent traces.

## Product Boundaries

- Authenticated API access is the default for user-facing routes.
- Evaluation is a first-class subsystem, not an afterthought to ordinary chat.
- Compatibility fallbacks exist for data migration and restore paths, but new writes should target canonical contracts.
