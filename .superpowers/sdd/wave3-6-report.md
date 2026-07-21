# Wave 3 Task 6 — Evidence Pool

## Delivered

- Added `data_base/agentic_v9/evidence_pool.py` as a thread-safe evidence registry.
- Source identity uses the packet's `doc_id`, `chunk_id`, `parent_id`, PDF page, and
  `asset_id`; only records with no precise source location use a normalized statement
  hash as a fallback. Aggregate metadata such as `source_doc_ids` never supplies a
  packet's source identity.
- Each source-identity entry retains immutable observations, including the original
  metadata and retrieval-score mappings, and chooses a canonical observation through
  stable serialization so concurrent additions are deterministic and idempotent.
- Retrieved, accepted, packed, used, and rejected lifecycle sets are separately
  observable. Rejection removes an item from non-terminal sets and prevents later
  acceptance, packing, or use.

## Focused verification

- Red: `tests/test_agentic_v9_evidence_pool.py` failed with
  `ModuleNotFoundError: data_base.agentic_v9.evidence_pool` before implementation.
- Green: `..\\..\\.venv\\Scripts\\python.exe -m pytest -q
  tests\\test_agentic_v9_evidence_pool.py` — 5 passed.
- Regression: `..\\..\\.venv\\Scripts\\python.exe -m pytest -q
  tests\\test_agentic_v9_schemas.py tests\\test_agentic_v9_context_packer.py
  tests\\test_agentic_v9_evidence_pool.py` — 48 passed.
- Lint: `..\\..\\.venv\\Scripts\\python.exe -m ruff check --no-cache
  data_base\\agentic_v9\\evidence_pool.py tests\\test_agentic_v9_evidence_pool.py`
  — passed.
- `git diff --check` — passed.

The project pytest cache is currently unwritable in this worktree, so pytest emitted a
cache-provider warning; all test assertions completed successfully.
