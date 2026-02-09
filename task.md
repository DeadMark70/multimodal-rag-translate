# Task Specification: Backend Refactor & Standardization (2026-02-09)

## 1. Overview
This track focuses on backend refactoring and code-standardization for the FastAPI multimodal Agentic RAG system.
The objective is to reduce technical debt while keeping runtime behavior stable.

## 2. Context
- Core backend modules: `pdfserviceMD/`, `data_base/`, `agents/`, `graph_rag/`, `multimodal_rag/`, `core/`.
- Existing audit findings are recorded in:
  - `agentlog/optimization_audit.md`
  - `agentlog/audit_20260122/security_audit.md`
  - `agentlog/audit_20260122/style_audit.md`
  - `agentlog/audit_20260122/unused_functions_audit.md`
- Conductor workflow references:
  - `conductor/workflow.md`
  - `conductor/archive/*/(spec.md|plan.md)`

## 3. Evaluation Constraint (Critical)
This project uses `bergen/` for benchmark and RAGAS scoring:
- `bergen/benchmark_rag_modes.py` produces benchmark results and `*_ragas.json`.
- `bergen/evaluate_ragas.py` produces faithfulness and answer_correctness reports.

Any refactor touching retrieval, agent orchestration, schemas, or LLM config must keep this pipeline runnable.

## 4. In Scope
1. Security hardening:
   - Change `doc_id: str` to `doc_id: UUID` in deletion/status endpoints where applicable.
2. Lint/style normalization:
   - Remove unused imports/variables and style violations with minimal behavior impact.
3. Dead code cleanup:
   - Remove only validated high-confidence dead internal helpers.
   - Keep `agents/planner.py::TaskPlanner.needs_planning` (intentional Agentic RAG complexity heuristic).
4. Maintainability refactor:
   - Improve readability in high-impact files (`main.py`, `pdfserviceMD/markdown_cleaner.py`).
5. Documentation sync:
   - Update `agentlog/` and `checklist/` when API/schema behavior changes.

## 5. Out of Scope
- Product feature redesign.
- Frontend visual/UI changes.
- Major architecture migration (database switch, framework migration).
- Arbitrary model strategy changes not required by refactor tasks.

## 6. Acceptance Criteria
- [x] P0 security issue (UUID path safety) is resolved in target routers.
- [x] Lint/style state is improved and not worsened.
- [x] No protected endpoint loses auth dependency.
- [x] `TaskPlanner.needs_planning` remains available and documented as intentional logic.
- [x] Impacted regression checks pass on touched scope (`ruff`, `py_compile`, function-level smoke checks).
- [x] Documentation is updated for interface/behavior changes and final handoff.

## 7. Deliverables
- Refactored backend code with minimal regression risk.
- Updated docs and checklists.
- Updated `implementation_plan.md` status markers.
- Clear summary of residual risks and next-step tasks.

## 8. References
- `agent.md`
- `implementation_plan.md`
- `README.md`
- `agentlog/codebase_overview.md`
- `agentlog/optimization_audit.md`
- `conductor/workflow.md`

## 9. Execution Notes (2026-02-09)
- Phase 5 in `implementation_plan.md` is canceled by user decision; associated Bergen/RAGAS commands are intentionally not executed.
- Full regression suite was additionally executed via `.venv\Scripts\python.exe -m pytest` with result: `225 passed`.
- Additional regression hardening completed with corpus-driven markdown cleaner tests (`tests/test_markdown_cleaner.py`, including fixture corpus and Pandoc sanitization integration stub): 18 passed.
- Validation commands in this hardening batch did not execute Bergen/RAGAS benchmark commands or paid LLM API evaluation flows.
