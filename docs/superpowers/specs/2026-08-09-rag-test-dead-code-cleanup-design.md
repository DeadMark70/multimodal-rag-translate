# RAG Test And Dead-Code Cleanup Design

## Goal

Remove obsolete RAG/Graph verification artifacts and brittle source-text tests
without changing runtime behavior, public APIs, or the current evaluation
system.

## Scope

This cleanup removes only artifacts whose repository usage has been checked and
whose behavior is either obsolete or covered by a stronger test.

### Remove obsolete manual scripts

- Delete `tests/run_arena.py`. It is not collected by pytest, uses a mock RAG
  answer, and is no longer part of the evaluation or smoke workflow.
- Delete `tests/verify_agentic_fix.py`. It is a one-off visual-verification
  debugging script, is not collected by pytest, and is superseded by the
  maintained visual and agentic smoke tests.
- Remove `compare_rag_vs_pure_llm()` from `agents/evaluator.py`. The deleted
  Arena script is its only Python caller.
- Remove the active `tests/run_arena.py` entry from
  `agentlog/codebase_overview.md`. Historical checklists, archived plans, and
  dead-code audit records remain unchanged because they describe past work.

### Replace the harmful structural GraphRAG test

- Delete `tests/test_graphrag_static_analysis.py`.
- Its module-level replacement of `networkx`, `igraph`, and `leidenalg` in
  `sys.modules` is process-global and order-dependent. It can make later test
  collection fail with `ValueError: networkx.__spec__ is not set`.
- The four symbol-existence checks are redundant with imports and behavioral
  tests elsewhere in the suite.
- Replace the source-text Leiden fallback assertion with one scoped behavioral
  test in `tests/test_community_builder_budget.py`. The test temporarily makes
  `igraph` unavailable, calls `detect_communities_leiden()`, and verifies that
  connected graph components become separate `Community` results. Pytest's
  `monkeypatch` fixture must restore import state after the test.

### Remove brittle source-text assertions

In `tests/test_rag_retrieval_generation_split.py`:

- Remove `test_legacy_wrapper_delegates_generation_without_exposing_visual_synthesis`.
  Facade delegation and visual generation are already protected by direct
  behavior tests.
- Remove `test_legacy_generation_avoids_python_311_incompatible_fstring_backslashes`.
  Python collection and the repository Ruff syntax rules already reject the
  original syntax error.
- Remove the unrelated assertion that reads
  `data_base/agentic_v9/model_paths.py` from the visual-generation test.
- Remove the now-unused `Path` import.

In `tests/test_rag_qa_prompts.py`:

- Remove `test_rag_qa_service_no_long_prompt_constants` and its `Path` import.
  Prompt registry keys, required variables, and formatting behavior remain
  covered directly.

## Non-Goals

- Do not modify `rag_pipeline.py`, `rag_graph_runtime.py`, or RAG behavior.
- Do not consolidate `test_reranker_logic.py` in this batch.
- Do not delete `experiments/evaluation_pipeline.py`; standalone experiment
  ownership requires a separate decision.
- Do not modify `CampaignEngine`, evaluation schemas, public APIs, Docker, or
  dependency versions.
- Do not create a shared test utility or dependency-injection framework.

## Verification

The implementation must demonstrate all of the following:

1. The new Leiden fallback behavior test passes with real `networkx` and a
   scoped missing-`igraph` simulation.
2. `tests/test_community_builder_budget.py` and
   `tests/test_reranker_logic.py` collect and run together without the former
   global-module contamination.
3. Focused evaluator, RAG generation, prompt, GraphRAG, and visual tests pass.
4. The complete backend pytest gate passes with the existing warning budget of
   56 and without external API calls.
5. Repository Ruff checks for `E9,F63,F7,F82,F401,F841`, the complexity
   ratchet, Markdown-link check, and `git diff --check` pass.
6. A repository search finds no live Python caller of
   `compare_rag_vs_pure_llm` and no active documentation directing users to
   either deleted script. Historical records may retain their original paths.

## Acceptance Criteria

- Runtime behavior and public imports are unchanged.
- The three obsolete files and the dead Arena helper are removed.
- One behavior-level fallback test replaces five structural/source tests.
- Brittle source-text assertions are removed while their behavior-level
  coverage remains.
- The final diff is limited to the files named in this design and contains no
  reranker consolidation or unrelated refactoring.
