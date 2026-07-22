# Wave 3 R3 fixes report

## Delivered

- Raised the deterministic `bounded_compare` retrieval budget to two rounds and added the planner-to-compiler R2 regression.
- Enforced one selected page per `asset_id` using the existing deterministic priority/order ranking; duplicate pages are persisted as `duplicate_asset_page` drops.
- Made visual extraction fail closed when `RetrievalTask.visual_required` is false, with no locator/model invocation and persisted `visual_not_required` drops.
- Bound visual packets to every supplied `EvidenceSource` field, excluding only a model-assigned `source_span_hash`; chunk, parent, and document-name rewrites are rejected.
- Required calculations to use span-hashed, validated direct premises and marked calculated packets `derived_non_evidence`, preventing derived prose from being presented as deterministic source evidence.
- Replaced the tautological route-policy assertion with the explicit frozen expected graph policy for each golden route.

## Verification

- RED: the focused regression run failed for the R2 compiler budget, duplicate-page selection, non-visual invocation, source-provenance rewrite, derived validation status, and raw-pool calculation acceptance before the implementation changes.
- `D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest <all test_agentic_v9_*.py> -q` — 128 passed. Pytest emitted one environment cache permission warning under `output\\test_tmp`; no tests failed.
- `D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache <modified files>` — passed. `--no-cache` is required because the worktree `.ruff_cache` is not writable.
- `git diff --check` — passed.
