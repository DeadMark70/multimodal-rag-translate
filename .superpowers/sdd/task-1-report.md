# Task 1 Report: Graph Evidence Anchor Contract

## Outcome

Implemented the Task 1 anchor contract in the isolated worktree and committed it as:

- `0fd5116` `feat: define graph evidence anchor contract`

## Files Changed

- `graph_rag/schemas.py`
  - Added `GraphEvidenceMode`.
  - Added `EvidenceAnchor` with computed `provenance_status`.
- `graph_rag/feature_flags.py`
  - Added `GraphFeatureFlags`.
  - Added `get_graph_feature_flags(...)`.
- `graph_rag/anchor_resolver.py`
  - Added `ChunkLookup` protocol.
  - Added `AnchorResolutionResult`.
  - Added `ChunkAnchorResolver.resolve(user_id, anchor)`.
- `tests/test_graph_anchor_contract.py`
  - Added contract tests for evidence modes, anchor serialization, flag defaults/snapshots, hash mismatch, and fuzzy quote resolution.
- `docs/BACKEND.md`
  - Documented the split between legacy `graph_raw_current` and the provenance-aware evidence-locator path.

## TDD Record

### Red

Ran the required focused test command before implementing production code:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests\test_graph_anchor_contract.py -q
```

Observed expected failure during collection:

- `ModuleNotFoundError: No module named 'graph_rag.anchor_resolver'`

This confirmed the test was failing because the new Task 1 contract had not been implemented yet.

### Green

After implementing the minimal production code, reran the focused tests:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests\test_graph_anchor_contract.py -q
```

Result:

- `8 passed`
- One pytest cache warning from the linked worktree filesystem; no test failures.

## Verification

### Pytest

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests\test_graph_anchor_contract.py -q
```

Result:

- Pass
- `8 passed, 1 warning in 0.31s`

### Ruff

The exact ruff command initially failed because ruff could not create its cache directory inside the linked worktree:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check graph_rag\schemas.py graph_rag\feature_flags.py graph_rag\anchor_resolver.py tests\test_graph_anchor_contract.py
```

Initial failure reason:

- cache initialization error
- filesystem permission denied for cache path creation

Reran the same focused check with `RUFF_CACHE_DIR` pointed at the user temp directory:

```powershell
$env:RUFF_CACHE_DIR='C:\Users\user\AppData\Local\Temp\codex-ruff-cache'; D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check graph_rag\schemas.py graph_rag\feature_flags.py graph_rag\anchor_resolver.py tests\test_graph_anchor_contract.py
```

Result:

- Pass
- `All checks passed!`

## Constraint Check

- Kept `NetworkX` untouched.
- Did not remove or rewrite `graph_raw_current`.
- Did not change prompts.
- Preserved router/service boundaries.
- Added `provenance_status` to every `EvidenceAnchor`.
- Added `resolution_status` and `verification_status` to resolved anchor results.
- Did not make graph summaries or unprovenanced relations eligible as final evidence.

## Notes

- `ChunkAnchorResolver` verifies quote matches when quote text is available and reports `quote_mismatch` instead of assuming verification succeeded.
- The worktree still has the untracked `.superpowers/` task artifact directory, which is expected for the report file.

## Review Finding Fix: 2026-07-09

### Scope

- `graph_rag/anchor_resolver.py`
- `tests/test_graph_anchor_contract.py`

### Changes

- Added wrong-document rejection coverage for `chunk_id` and fuzzy-quote candidates.
- Updated the fuzzy-quote contract test to assert `verification_status == "quote_match"`.
- Hardened `ChunkAnchorResolver` to validate candidate metadata with `data_base.document_metadata.matches_document_id(...)` before accepting any lookup result.
- Wrong-document candidates now fall through to an unresolved result with:
  - `resolution_status = "unresolved"`
  - `verification_status = "not_checked"`
  - `reason = "doc_id_mismatch"`
- Successful fuzzy-quote resolution now reports `verification_status = "quote_match"` as required by the contract.

### TDD Record

#### Red

Ran the required focused test command after adding the regression coverage:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests\test_graph_anchor_contract.py -q
```

Observed expected failures before the resolver fix:

- `test_anchor_resolver_fuzzy_quote_match`
- `test_anchor_resolver_rejects_chunk_id_from_wrong_document`
- `test_anchor_resolver_rejects_fuzzy_quote_from_wrong_document`

Summary:

- `3 failed, 7 passed, 2 warnings`

#### Green

Reran the same focused test command after the resolver change:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests\test_graph_anchor_contract.py -q
```

Summary:

- `10 passed, 1 warning in 0.30s`

### Lint Verification

```powershell
$env:RUFF_CACHE_DIR='C:\Users\user\AppData\Local\Temp\codex-ruff-cache'; D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check graph_rag\anchor_resolver.py tests\test_graph_anchor_contract.py
```

Summary:

- `All checks passed!`
