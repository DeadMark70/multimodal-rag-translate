# Wave 4 R4/R5 fixes report

## Closed findings

1. Provider-supplied `FinalAnswerResult` values are now treated as untrusted
   drafts and pass the normal deterministic claim validation, qualification,
   and citation rendering path. The only direct result allowed is the exact,
   claim-free, zero-generation deterministic unavailable-final fallback.
2. The execution core records the final result's actual
   `final_generation_count`, allowing the budgeted fallback's count of zero
   without an assertion failure.
3. `V9RuntimeContext.cancellation_token` is passed to every bounded core stage,
   including scope resolution, retrieval, curation, conflict resolution,
   packing, and final generation.
4. The core requires an attempt-created `V9RuntimeContext` deadline before
   source scope resolution and no longer creates a deadline itself.
5. When the deadline is exhausted before packing, the core returns its
   deterministic partial and does not start a timed pack operation.

## Regression coverage

- `tests/test_agentic_v9_final_answer.py`: direct-result verification and
  qualification/citation, rejected arbitrary zero-generation results, and the
  exact claim-free fallback.
- `tests/test_agentic_v9_execution_core.py`: required attempt context,
  in-flight scope cancellation, zero-count final fallback, and
  exhausted-before-pack partial completion.

## Validation

- Project venv Agentic v9 tests: `172 passed`.
- Scoped focused tests: `17 passed`.
- Scoped Ruff check: passed.
