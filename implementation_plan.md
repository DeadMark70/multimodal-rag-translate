# Implementation Plan: Backend Refactor & Standardization

This plan is the execution source of truth for the current refactor track.

## Status Legend
- `[ ]` not started
- `[~]` in progress
- `[x]` completed
- `[c]` canceled

## Phase 0: Baseline & Safety Net [checkpoint: pending]
- [ ] Task: Capture current lint/test baseline and list known failing tests.
- [ ] Task: Confirm touched modules and map risk areas before code edits.

## Phase 1: Security Hardening (P0) [checkpoint: completed]
- [x] Task: Refactor `pdfserviceMD/router.py` endpoints to use `doc_id: UUID`.
- [x] Task: Refactor `multimodal_rag/router.py` delete endpoint to use `doc_id: UUID`.
- [x] Task: Verify path handling remains traversal-safe after type changes.
- [x] Task: Add/update targeted tests for invalid `doc_id` inputs.

## Phase 2: Lint & Style Cleanup (P1) [checkpoint: completed]
- [x] Task: Run lint autofix and formatting with controlled scope.
- [x] Task: Resolve remaining high-impact style items (`E701`, `E741`, `F541`, selective `E402`).
- [x] Task: Re-run affected tests to confirm no behavior regressions.

## Phase 3: Dead Code Cleanup (P2) [checkpoint: completed]
- [x] Task: Re-validate dead code candidates against runtime usage and tests.
- [x] Task: Remove validated dead helpers in `data_base/`, `multimodal_rag/`, `agents/`.
- [x] Task: Keep `TaskPlanner.needs_planning` as an intentional complexity heuristic for Agentic RAG routing.
- [x] Task: Confirm no API endpoint/schema false positives were removed.

## Phase 4: Maintainability Refactor (P3) [checkpoint: completed]
- [x] Task: Reduce `main.py` setup complexity using helper/app-factory style extraction.
- [x] Task: Refactor `pdfserviceMD/markdown_cleaner.py` for readability and naming clarity.
- [x] Task: Improve type hints/docstrings for modified public functions.

## Phase 5: Regression & Evaluation Validation [checkpoint: canceled_by_user_20260208]
- [c] Task: Run targeted backend tests for touched modules.
- [c] Task: Run broader test pass (`pytest`) when feasible.
- [c] Task: Run Bergen smoke benchmark:
      `python bergen/benchmark_rag_modes.py --limit 1 --modes naive,advanced,graph,agentic --output bergen/benchmark_results_smoke.json`
- [c] Task: Run Bergen RAGAS scoring:
      `python bergen/evaluate_ragas.py --input bergen/benchmark_results_smoke_ragas.json --output bergen/evaluation_results_smoke.json`

## Phase 6: Documentation & Handoff [checkpoint: completed]
- [x] Task: Update `agentlog/` and `checklist/` docs for changed APIs/schemas.
- [x] Task: Update `agent.md`, `task.md`, and this plan to reflect final state.
- [x] Task: Record residual risks, open issues, and next recommended batch.

### Phase 6 Residual Risks / Open Issues
- Full suite regression was executed on 2026-02-09 via `.venv\Scripts\python.exe -m pytest`: `225 passed`.
- Phase 5 Bergen benchmark/RAGAS command set remains canceled by user decision (`canceled_by_user_20260208`) and was not executed.
- `markdown_cleaner` now uses scanner-based math protection and a seeded regression corpus (`tests/fixtures/markdown_cleaner/regression_corpus.json`), but very rare malformed LaTeX edge cases may still require ongoing corpus expansion from real failures.
- Recommended next batch: expand corpus with production failure samples and run optional non-Phase-5 PDF->Markdown->PDF integration checks on representative hard documents.
