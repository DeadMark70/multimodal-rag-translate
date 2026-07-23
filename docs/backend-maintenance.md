# Backend Maintenance Guide

This guide records the current backend maintenance boundaries after the 2026-07-06 FastAPI cleanup. It is meant for future agents that need to change backend behavior without re-reading the entire project.

## Runtime Baseline

- Local Windows development currently uses Python 3.13.
- Run commands from the repository root.
- Prefer:
  - `.\.venv\Scripts\python.exe -m pytest ...`
  - `.\.venv\Scripts\python.exe -m ruff check ...`
- `pytest.ini` stores pytest cache under `output/test_tmp/.pytest_cache`.
- Docker is not required for the local verification flow in this workspace.

## Router Boundary Rules

Routers should stay thin:

- Parse request payloads.
- Enforce auth dependencies.
- Call service/helper functions.
- Schedule background tasks.
- Return response models.

Routers should not own:

- Long-running maintenance logic.
- File cleanup and copy orchestration.
- Graph rebuild/retry/optimize implementation details.
- PDF post-processing/indexing implementation details.
- Cross-router imports.

Static boundary expectations live in `tests/test_router_boundaries.py`.

## Current Maintenance Modules

### GraphRAG

- `graph_rag/router.py`
  - HTTP API layer for graph endpoints.
  - Should not grow new large maintenance helpers.
- `graph_rag/maintenance.py`
  - Graph maintenance operations extracted from router code.
  - Keep purge/rebuild/retry/optimize-style background work here or in adjacent service helpers.
- `graph_rag/service.py`
  - Shared graph service logic used outside router-only code.

When changing GraphRAG maintenance, run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py tests/test_rag_graph_evidence_docs.py tests/test_router_boundaries.py -q
```

### PDF Markdown Service

- `pdfserviceMD/router.py`
  - HTTP API layer for upload/translation/manual operations.
- `pdfserviceMD/indexing_tasks.py`
  - Background indexing and retry-index orchestration.
- `pdfserviceMD/service.py`
  - Shared PDF service helpers, including retry-index context preparation.

When changing PDF background/indexing behavior, run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_pdfservice_background_processing.py tests/test_pdfservice_manual_translation.py tests/test_router_boundaries.py -q
```

### Evaluation Release Metrics And Trace Lists

- `GET /api/evaluation/campaigns/{campaign_id}/release-metrics` is owned by `evaluation/release_metrics.py`; routers only authenticate and delegate.
- Treat `availability="not_applicable"` plus `not_applicable_reason="benchmark_not_configured"` as the expected result for a campaign with no benchmark. It is not a failed release calculation and it must return before any result, score, accounting, or observability bulk read.
- For configured benchmarks, keep projection reads bounded and campaign-scoped: bulk-load the selected campaigns' results, scores/work metadata, accounting snapshot, and release observability snapshot once each. Do not add per-run repository reads or pull large answer/context/trace blobs into the release or campaign-list paths.
- Terminal release reports are process-local cache entries keyed by each selected campaign's ID, `updated_at`, and status. Any marker change invalidates the entry; nonterminal campaigns always load fresh state.
- `agent_traces.summary_json` is the compact list projection introduced by the trace-summary migration. Preserve the legacy `trace_json` detail path, backfill/maintain summaries through the repository, and keep the `(campaign_id, user_id, created_at)` list index usable.
- The repository enforces a 1,048,576-byte UTF-8 answer limit. Preserve the explicit `EVALUATION_ANSWER_TOO_LARGE` error code through durable and legacy execution; do not truncate a provider answer to make it persist.

When changing these paths, run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; uv run --python 3.13 --with-requirements requirements.txt python -m pytest tests/test_evaluation_release_metrics.py tests/test_evaluation_api.py tests/test_evaluation_db.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py -q
uvx --from ruff==0.14.13 ruff check evaluation/release_metrics.py evaluation/db.py evaluation/execution_worker.py evaluation/campaign_engine.py tests/test_evaluation_release_metrics.py tests/test_evaluation_db.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py
```

## Upload And Path Boundaries

- Upload-root paths and PDF upload validation should stay centralized in `core/uploads.py`.
- Do not reintroduce ad hoc upload path handling in routers.
- New document/vector metadata writes should use canonical `doc_id`; `original_doc_uid` is compatibility fallback only for read/delete paths.

## Production Scope

Production architecture checks should exclude:

- `experiments/`
- `bergen/`
- `scripts/`

Do not weaken this exclusion list casually. These directories contain experiments, evaluation utilities, or scripts that may intentionally violate production layering rules.

## Focused Verification

Use focused verification before broad test runs:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_router_boundaries.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py tests/test_pdfservice_background_processing.py tests/test_pdfservice_manual_translation.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag pdfserviceMD tests/test_router_boundaries.py
```

Full-suite runs may still expose legacy environment or experiment/RAGAS blockers. If that happens, report exact blockers and the focused suites that passed.
