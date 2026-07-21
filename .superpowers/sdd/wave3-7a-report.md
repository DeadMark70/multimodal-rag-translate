# Wave 3 — Task 7A report

## Outcome

Added the v9-only `SourceScopeResolver`. It normalizes source display names,
requires each name to resolve to exactly one internal document ID, and produces
an effective scope that cannot exceed both the request and adapter-provided
authorization. Unknown, ambiguous, or unauthorized request elements clear the
effective authorized scope.

`ResolvedSourceScope.resolved_doc_ids` retains successful name resolution for
authorized-scope diagnostics, while retrieval must consume only
`authorized_doc_ids`.

## Verification

- `python -m pytest tests/test_agentic_v9_*.py -q` — 93 passed.
- `ruff check --no-cache data_base/agentic_v9 tests/test_agentic_v9_source_scope_resolver.py` — passed.
- `git diff --check` — passed.

Pytest emitted its existing worktree cache-permission warning; it did not affect
test execution.
