# Wave 5 Task 14 report

## Delivered

- Added explicit campaign `agentic_execution_version` (`v8`/`v9`) and validated
  shadow policy (`operational`/`research`) with formal identity aliases.
- Added a policy-free campaign identity adapter and typed used-evidence document
  projection; v9 agentic RAG now emits only final-claim-cited evidence as its
  documents/source contexts, with deterministic deduplication.
- Propagated condition, execution profile/version, response status, identity,
  and shadow policy through campaign units, durable work snapshots, worker
  results, DB projections, and `/runs` analytics rows.
- Product/operational shadow work remains diagnostic: it is excluded from
  authoritative RAGAS work; research shadow remains independently evaluated.
- Kept the v8 service construction and behavior unchanged when no v9 version
  is selected.

## Verification

- `ruff check --no-cache` for all changed source/test files: passed.
- Focused tests: `40 passed` across campaign schemas, rag modes, agentic
  evaluation service, and the new Task 14 contracts.
- `git diff --check`: passed.
- The durable execution-worker test module could not run in this worktree
  because the shared `output/test_tmp` directory rejects new test directories
  with Windows `PermissionError`; this is an environment permission issue,
  not a test assertion failure.
