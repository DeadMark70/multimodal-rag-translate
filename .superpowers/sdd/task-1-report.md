# Task 1: Shared Evaluation Retrieval Policy Report

## Implementation summary

Added a dependency-neutral `evaluation.retrieval_profiles` policy module for
Evaluation Center execution paths. It exposes fresh query-expansion settings,
source-backed locator-to-chunk graph hints, a non-mutating no-HyDE normalizer,
and versioned execution profiles for changed evaluation modes. No ordinary chat
or Deep Research execution code was modified.

## Files changed

- `evaluation/retrieval_profiles.py`
- `tests/test_evaluation_retrieval_profiles.py`
- `.superpowers/sdd/task-1-report.md`

## RED evidence

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_retrieval_profiles.py -q
```

Output:

```text
==================================== ERRORS ====================================
________ ERROR collecting tests/test_evaluation_retrieval_profiles.py _________
ImportError while importing test module 'D:\flutterserver\pdftopng\tests\test_evaluation_retrieval_profiles.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\user\AppData\Local\Programs\Python\Python313\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_evaluation_retrieval_profiles.py:1: in <module>
    from evaluation.retrieval_profiles import (
E   ModuleNotFoundError: No module named 'evaluation.retrieval_profiles'
=========================== short test summary info ===========================
ERROR tests/test_evaluation_retrieval_profiles.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.24s
```

## GREEN evidence

Focused policy command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_retrieval_profiles.py -q
```

Output:

```text
....                                                                     [100%]
4 passed in 4.96s
```

Evaluation-area regression command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_retrieval_profiles.py tests/test_rag_retrieval_logic.py tests/test_query_transformer.py tests/test_rag_modes_agentic.py tests/test_agentic_evaluation_service.py tests/test_graph_context_packing.py tests/test_graph_auto_gate.py tests/test_graph_evidence_bundle_wrapper.py tests/test_evaluation_graph_events.py tests/test_research_execution_core_generic.py tests/test_deep_research.py -q
```

Output:

```text
........................................................................ [ 66%]
....................................                                     [100%]
108 passed, 23 warnings in 7.09s
```

The 23 warnings are pre-existing third-party `storage3` Pydantic v2
deprecation warnings; the command exited successfully.

## Lint and formatting

Command:

```powershell
.\.venv\Scripts\python.exe -m ruff check evaluation/retrieval_profiles.py tests/test_evaluation_retrieval_profiles.py
.\.venv\Scripts\python.exe -m ruff format --check evaluation/retrieval_profiles.py tests/test_evaluation_retrieval_profiles.py
```

Output:

```text
All checks passed!
2 files already formatted
```

The initial sandboxed Ruff attempt could not create `.ruff_cache`; rerunning
the exact checks outside the sandbox produced the successful result above.

## Self-review

- Verified all requested exports and exact profile naming conventions are
  present.
- Confirmed `apply_no_hyde_policy()` deep-copies supplied mode settings before
  changing only the specified Evaluation Center modes.
- Confirmed graph hints explicitly disable automatic gating and keep locator
  evidence/provenance/chunk flags enabled.
- Confirmed the focused 108-test regression set, including Deep Research
  coverage, passes without modifying chat or Deep Research behavior.
- `git diff --check` completed without whitespace errors.

## Commit

`feat(evaluation): centralize retrieval policies`

## Concerns

None. The only non-clean test output was the unrelated third-party deprecation
warnings noted above.
