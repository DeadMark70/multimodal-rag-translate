# Task 3: Standard Evaluation Mode Wiring and Profiles Report

## Implementation summary

Normalized Evaluation Center standard and Graph-ablation modes with the shared
no-HyDE policy, made the main Graph baseline explicitly locator-to-chunk, and
persisted versioned execution profiles for changed modes. Naive and both
context-policy version constants remain unchanged; Agentic route internals,
Deep Research, chat, frontend, and schema behavior were not modified.

## Files changed

- `evaluation/rag_modes.py`
- `tests/test_rag_modes_agentic.py`
- `.superpowers/sdd/task-3-report.md`

## RED evidence

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_modes_agentic.py -q
```

Output:

```text
.......F.FFFF                                                            [100%]
5 failed, 8 passed, 25 warnings in 7.36s
```

The failures were the intended missing behavior: changed modes still enabled
HyDE, main Graph had no `graph_execution_hints`, and advanced, Graph, and
`graph_locator_to_chunk` results persisted no execution profile.

## GREEN evidence and regressions

Focused mode command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_modes_agentic.py -q
```

Output:

```text
13 passed, 24 warnings in 7.31s
```

Required cross-suite regression command (rerun after formatting):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_modes_agentic.py tests/test_graph_ablation_conditions.py tests/test_graph_anchor_contract.py tests/test_evaluation_retrieval_profiles.py -q
```

Output:

```text
31 passed, 24 warnings in 5.84s
```

## Lint and formatting

Commands:

```powershell
.\.venv\Scripts\python.exe -m ruff check evaluation/rag_modes.py tests/test_rag_modes_agentic.py
.\.venv\Scripts\python.exe -m ruff format --check evaluation/rag_modes.py tests/test_rag_modes_agentic.py
```

Output:

```text
All checks passed!
2 files already formatted
```

The initial sandboxed Ruff attempt could not create `.ruff_cache`; the exact
commands were rerun with the narrowly approved cache-write permission. Ruff
reflowed pre-existing long literals in the two scoped files so the mandatory
format check passes.

## Self-review

- `apply_no_hyde_policy()` is applied after the mode dictionary and affects
  only the Task 1-defined changed Evaluation Center modes.
- Main `graph` receives only `locator_to_chunk_graph_hints()`; the
  `graph_raw_current` hint stays `raw_current` and Graph ablation flags retain
  their exact semantics.
- The profile fallback preserves an agent-provided trace profile and otherwise
  uses `evaluation_execution_profile(mode)`.
- `CONTEXT_POLICY_VERSION` and `AGENTIC_CONTEXT_POLICY_VERSION` are unchanged.
- `git diff --check` completed successfully before commit.

## Commit

`feat(evaluation): use multi-query graph locator baseline`

## Concerns

No implementation concerns. Test output contains pre-existing third-party
`storage3` Pydantic-v2 deprecation warnings and a Pytest cache permission
warning under `output/test_tmp`; all required commands exited successfully.

---

# Task 3 Follow-up — Cache terminal Release Metrics reports safely

## Scope

- Modified `evaluation/release_metrics.py` and `tests/test_evaluation_release_metrics.py` only.
- Added an instance-local terminal report cache keyed exactly by the sorted selected campaign `(id, updated_at.isoformat(), status.value)` tuples.
- Cache entries are used only when every selected campaign is terminal: `completed`, `completed_with_errors`, `failed`, or `cancelled`.
- Cached reports are deep-copied on both write and read. The pre-existing no-benchmark `not_applicable` return remains before cache evaluation.

## TDD evidence

RED command:

```powershell
uv run --python 3.13 --with-requirements requirements.txt python -m pytest -q tests/test_evaluation_release_metrics.py -k "caches_unchanged_terminal or reloads_when_any_selected or does_not_cache_nonterminal"
```

Result: 1 failed, 3 passed. The unchanged completed benchmark loaded each bulk source twice: actual `(2, 2, 2, 2, 2)`, expected `(1, 1, 1, 1, 1)`.

GREEN command: same command after the minimal implementation.

Result: 4 passed, 12 deselected.

## Final verification

```powershell
uv run --python 3.13 --with-requirements requirements.txt python -m pytest -q tests/test_evaluation_release_metrics.py
uv run --python 3.13 --with-requirements requirements.txt ruff check evaluation/release_metrics.py tests/test_evaluation_release_metrics.py
git diff --check
```

Results:

- `16 passed` for the focused release-metrics module.
- Ruff: `All checks passed!`
- Diff check: no whitespace errors.

The pytest environment emitted 23 pre-existing third-party `storage3` Pydantic deprecation warnings; they are unrelated to this change.

## Self-review

- Unchanged terminal campaigns reuse the process-local cache.
- Changing any selected campaign status or timestamp changes the exact cache key and reloads all selected campaign inputs.
- Running and evaluating campaigns bypass the terminal cache on every request.
- The P0 missing-benchmark fast return is still first and no `not_applicable` result is cached.
