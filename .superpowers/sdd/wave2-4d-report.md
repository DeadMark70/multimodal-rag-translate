# Wave 2 Task 4D Report

## Scope

Implemented Task 4D only: isolated the legacy generic RAG corrective-retrieval
guard in `data_base/rag_crag.py`. Graph location and answer generation remain
in `data_base/RAG_QA_service.py` for Tasks 4E and 4F.

## Changes

- Added `data_base/rag_crag.py`.
  - `classify_crag_retrieval()` is deterministic: an empty retrieval is
    insufficient; a populated retrieval requires a relevance decision. It does
    not infer relevance or invoke a provider.
  - `run_corrective_retrieval()` accepts an optional judge plus injected HyDE,
    multi-query, retrieval, RRF, candidate-limit, and rerank seams. A missing
    judge safely accepts the original documents without any rewrite or provider
    call.
  - The legacy `judge_retrieved_documents()` adapter lazily constructs
    `RAGEvaluator`, preserving the v8 LLM judge and its provider behavior.
  - Rejected retrieval preserves the legacy correction order: progress event,
    rewrite, retrieval, RRF for multi-query batches, optional document-ID
    filtering, optional capped reranking in the threadpool, and target-K
    truncation. Empty corrected evidence is reported as insufficient.
- Updated `data_base/RAG_QA_service.py` to delegate its Step 5.4 CRAG branch to
  the new module.
  - Retained `_build_crag_queries()` as a compatibility facade for existing
    callers/tests.
  - Retained opt-in CRAG, HyDE as the default rewrite, existing progress event
    names/payloads, and fallback to the initial retrieval on any CRAG error.
  - Kept the legacy service out of v9 provider paths; the injected v9 CRAG
    judge/budget-invoker seam remains unchanged and `rag_crag.py` creates no
    direct provider invocation.
- Added `tests/test_rag_crag.py` covering deterministic classification,
  provider-free acceptance when no judge is supplied, and rejected
  multi-query corrective retrieval with RRF, source scope filtering, and
  progress emission.

## TDD Evidence

1. RED: `tests/test_rag_crag.py` initially failed during collection with
   `ModuleNotFoundError: No module named 'data_base.rag_crag'`.
2. GREEN: added the smallest isolated CRAG module needed for those contracts;
   the new tests passed.
3. Integration: delegated the legacy service branch and reran the existing
   CRAG parity regressions alongside the new tests.

## Verification

- `D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests\test_rag_crag.py tests\test_rag_retrieval_logic.py tests\test_rag_retrieval_pipeline.py tests\test_rag_filtering.py tests\test_reranker_logic.py tests\test_rag_retrieval_generation_split.py -q`
  - `28 passed`.
- `D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base\rag_crag.py data_base\RAG_QA_service.py tests\test_rag_crag.py`
  - all checks passed.
- `git diff --check`
  - no whitespace errors.

Pytest emitted pre-existing third-party Pydantic deprecation warnings and a
worktree pytest-cache permission warning. Neither affected test execution.

## Deferred

No Graph locator extraction (4E), generation extraction or visual behavior
work (4F), or agentic-v9 runtime/provider-policy changes were made.
