# Wave 2 Task 4B Report

## Scope

Implemented only Task 4B: extracted the legacy dense/BM25 retrieval execution,
HyDE and multi-query expansion, and reciprocal-rank fusion into an evidence-only
retrieval module. Filtering, reranking, CRAG, graph location, and answer
generation remain in `data_base/RAG_QA_service.py` for Tasks 4C-4F.

## Changes

- Added `data_base/rag_retrieval.py`.
  - `retrieve_hybrid_documents` accepts the request-scoped hybrid retriever.
  - It performs HyDE or multi-query expansion, runs retrieval, and applies RRF
    when more than one query batch is returned.
  - It returns `RagRetrievalResult` only and has no answer prompt or generation
    dependency.
  - Its metadata records the original query, expansion mode/use/queries,
    per-query origin, per-batch one-based ranks and document metadata, fusion
    usage, final one-based ranks, and ordered unique source document IDs.
- Updated `data_base/RAG_QA_service.py` to delegate only Steps 3-4 to the new
  module, while passing existing transformation/execution seams to retain
  legacy behavior and tests.
- Added `tests/test_rag_retrieval_pipeline.py`.
  - Covers multi-query expansion, query origins, RRF rank ordering, metadata,
    source IDs, and progress events.
  - Covers HyDE origin metadata and the single-query no-fusion path.

## TDD Evidence

1. RED: the new test failed at collection with
   `ModuleNotFoundError: No module named 'data_base.rag_retrieval'`.
2. GREEN: added the retrieval module and wired the legacy service; focused and
   legacy retrieval tests passed.

## Verification

- `D:\\flutterserver\\pdftopng\\.venv\\Scripts\\python.exe -m pytest tests\\test_rag_retrieval_pipeline.py tests\\test_rag_retrieval_logic.py tests\\test_rag_retrieval_generation_split.py -q`
  - `16 passed`.
- `D:\\flutterserver\\pdftopng\\.venv\\Scripts\\python.exe -m ruff check --no-cache data_base\\rag_retrieval.py data_base\\RAG_QA_service.py tests\\test_rag_retrieval_pipeline.py`
  - all checks passed.
- `git diff --check`
  - no whitespace errors.

Pytest emitted existing third-party Pydantic deprecation warnings and could not
write its configured worktree cache due to permissions. `compileall` also could
not write the worktree `__pycache__`; imports and executed pytest coverage were
successful.

`tests/test_rag_ask_stream.py` could not collect because application startup
attempted to create the worktree `uploads` directory and received
`PermissionError`; it did not reach the retrieval code.

## Deferred

No 4C filtering/reranking extraction, 4D CRAG extraction, 4E graph locator,
4F generation extraction, visual behavior changes, or v9 executor changes were
made.
