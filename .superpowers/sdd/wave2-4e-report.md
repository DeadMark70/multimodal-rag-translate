# Wave 2 Task 4E Report — Graph Source Locator

## Scope

Implemented Task 4E only. The graph source-location phase now lives in
`data_base/rag_graph_locator.py`; legacy answer generation and visual behavior
remain in `data_base/RAG_QA_service.py` for Task 4F.

## Changes

- Added `GraphSourceLocatorResult` and `locate_graph_sources()`.
  - The result explicitly exposes `route`, `path`, `fallback`, timing,
    candidate/resolved/scope-approved/scored/packed item IDs, and resolved
    source document/chunk IDs.
  - A graph bundle is consumed only through a caller-injected `bundle_locator`.
    The existing RAG service therefore retains authority over graph routing,
    feature flags, execution hints, and Evaluation Setup configuration.
  - The locator does not create an LLM, read or override model setup, render
    graph text, or expose a raw graph-context property.
  - Graph-derived documents originate only from provenance-eligible anchors
    re-resolved to persisted chunks. Raw summaries, edges, relations, and
    community hints cannot enter `documents` as claim-supporting evidence.
  - Lookup/expansion failures preserve the vector documents and return the
    observable `source_expand_failed` fallback.
- Updated the existing source-expand branch of
  `data_base/RAG_QA_service.py` to delegate to the locator, then preserve its
  existing graph lifecycle and observability event contract. The separately
  gated legacy raw-graph compatibility route is unchanged.
- Added `tests/test_rag_graph_locator.py` covering:
  - raw graph content is absent from claim-supporting documents;
  - route/path/fallback and vector fallback observability;
  - verified graph anchors report resolved source IDs and merge the persisted
    source chunk rather than inferred graph text.

## TDD evidence

The initial focused test run failed during collection because
`data_base.rag_graph_locator` did not exist. After implementation, the focused
tests passed. A later resolved-source test fixture initially omitted the full
provenance `chunk_hash` required by `GraphEvidenceBundle`; it was corrected to
exercise a valid provenance-eligible source anchor.

## Verification

- `D:\flutterserver\pdftopng\.venv\Scripts\python.exe -B -m pytest tests\test_rag_graph_locator.py tests\test_graph_evidence_bundle_wrapper.py tests\test_rag_graph_evidence_docs.py -q -p no:cacheprovider`
  - 12 passed.
- `D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base\rag_graph_locator.py data_base\RAG_QA_service.py tests\test_rag_graph_locator.py`
  - All checks passed.
- `git diff --check`
  - No whitespace errors.

Pytest reported pre-existing third-party Pydantic deprecation warnings and an
unknown `cache_dir` warning when the cache provider was disabled. Neither
affected test execution.

## Commit

`refactor(rag): isolate graph source locator`
