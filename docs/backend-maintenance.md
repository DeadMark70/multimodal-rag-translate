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
