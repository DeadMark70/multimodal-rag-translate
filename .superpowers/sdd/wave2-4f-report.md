# Wave 2 Task 4F Report — Legacy Generation Parity

## Scope

Implemented Task 4F only. The v8 compatibility wrapper retains retrieval,
filtering, CRAG, graph-location, and legacy response projection. Answer
generation is delegated to `data_base/rag_generation.py`.

## Changes

- Added `generate_legacy_answer_from_evidence()`.
  - Builds the historical plain and advanced prompts from retrieved documents.
  - Retains document filename labels, image encoding, graph context, history,
    intent constraints, progress events, usage metrics, and legacy error text.
  - Returns `GeneratedRagAnswer`, preserving the retrieval/generation boundary.
- Moved the active v8 visual verification and re-synthesis loop into
  `rag_generation.py`.
  - `VERIFY_IMAGE` parsing, forced one-shot fallback, tool-call metadata, and
    visual synthesis remain legacy-only.
  - `data_base/agentic_v9` has no import of the legacy visual tool or
    `rag_generation`.
- Updated `rag_answer_question()` to delegate post-retrieval generation, then
  project the result back to the unchanged `RAGResult` or `(answer, ids)` v8
  public return shape.
- Expanded parity coverage for plain/naive, advanced, graph-context, empty
  retrieval, provider error, visual verification, and wrapper delegation.

## Verification

- `pytest tests/test_rag_retrieval_generation_split.py tests/test_rag_graph_evidence_docs.py tests/test_rag_modes_agentic.py tests/test_rag_qa_prompts.py tests/test_visual_tool_trigger.py -q`
  - 34 passed.
- `ruff check --no-cache data_base/rag_generation.py data_base/RAG_QA_service.py tests/test_rag_retrieval_generation_split.py`
  - Passed.
- `git diff --check`
  - Passed.

Pytest emitted pre-existing third-party Pydantic deprecation warnings and a
worktree pytest-cache permission warning; neither affected outcomes.
