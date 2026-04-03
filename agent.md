# Agent Working Guide (`agent.md`)

## 1. Purpose
This file defines how refactoring and code-standardization work should be executed in this repository.
Primary goals:
- keep behavior stable (no regressions)
- reduce technical debt with measurable steps
- improve readability, maintainability, and security

## 2. Current Project Understanding
This is a FastAPI-based multimodal Agentic RAG backend.

### Runtime entry
- `main.py`: thin entrypoint (`app = create_app()`).
- `core/app_factory.py`: app bootstrap, env loading, CORS, lifespan startup warmup, router registration.

### Core modules
- `pdfserviceMD/`: PDF upload -> OCR -> Markdown translation -> PDF regeneration -> async post-processing.
- `data_base/`: FAISS indexing, hybrid retrieval, reranking, Deep Research orchestration.
- `multimodal_rag/`: visual extraction and image summarization for multimodal retrieval.
- `graph_rag/`: entity/relation extraction, graph store, local/global graph search.
- `agents/`: planner, evaluator, synthesizer.
- `image_service/`: in-place image OCR translation.
- `conversations/`: conversation + message persistence.
- `stats/`: dashboard aggregation from query logs.
- `core/`: auth, LLM factory, summary service.

### Documentation sources reviewed
- `README.md`
- `agentlog/*.md`
- `agentlog/audit_20260122/*.md`
- `checklist/*_guide.md`
- `conductor/workflow.md`
- `conductor/archive/*/(spec.md|plan.md)`

### Evaluation stack note (critical)
- This repository also uses `bergen/` for RAG benchmarking and RAGAS scoring.
- Core evaluation flow:
  - `bergen/benchmark_rag_modes.py` generates benchmark outputs and `*_ragas.json`.
  - `bergen/evaluate_ragas.py` computes RAGAS metrics (faithfulness, answer_correctness).
- Refactors in `data_base/`, `agents/`, `core/llm_factory.py`, schema fields, or output formats must preserve this evaluation flow unless explicitly planned.
- User-facing `Deep Research` and evaluation `agentic` may share execution primitives, but must not share an unversioned wrapper once their behavior diverges. Persist an explicit `execution_profile` for evaluation baselines.

## 3. Refactor Priority (Execution Order)
### P0: Security hardening (must do first)
- change delete/status-related `doc_id: str` to `doc_id: UUID` in:
  - `pdfserviceMD/router.py`
  - `multimodal_rag/router.py`
- verify path handling is traversal-safe (`os.path.basename`, normalized paths, UUID-only IDs).

### P1: Style/lint noise reduction
- run and apply:
  - `ruff check . --fix`
  - `ruff format .`
- manual cleanup for remaining categories (`E701`, `E741`, `F541`, selective `E402` handling in `main.py`).

### P2: Dead code cleanup
- validate and remove high-confidence dead symbols (from audit), including candidates in:
  - `data_base/RAG_QA_service.py`
  - `multimodal_rag/utils.py`
  - `agents/evaluator.py`
- do not remove FastAPI endpoints, Pydantic fields, or test helpers without usage verification.
- keep `TaskPlanner.needs_planning` in `agents/planner.py` (intentional complexity heuristic for Agentic RAG / Deep Research).

### P3: Maintainability refactor
- split `main.py` bootstrap logic into smaller functions/module(s) (e.g. app factory pattern).
- improve readability of `pdfserviceMD/markdown_cleaner.py` (ambiguous names, one-line multi-statements).
- improve type hints and docstrings for public interfaces.

### P4: Documentation consistency
- when API/schema/runtime behavior changes, update related docs in `agentlog/` and `checklist/`.
- keep endpoint paths consistent with runtime registration in `main.py`.

## 4. Non-Negotiable Coding Rules
- Add type hints to new/modified function signatures.
- Keep import order: stdlib -> third-party -> local modules.
- No bare `except:`; catch specific exception types.
- Use `logging` instead of `print`.
- CPU-bound/blocking work must use `run_in_threadpool` in FastAPI paths.
- Protected endpoints must keep `Depends(get_current_user_id)`.
- Use safe path building (`os.path.join`, `os.path.normpath`) and strict input validation.
- Do not mutate global hardware env vars (for example `CUDA_VISIBLE_DEVICES`) at module import time; keep device policy local to the component being initialized.
- When adding streamed and non-streamed variants of the same API capability, keep them on one shared execution path and make the streamed `complete` payload match the synchronous response contract.
- Production-only architecture checks must exclude `experiments/`, `bergen/`, and `scripts/`.
- Router modules must not import other router modules; shared cross-domain logic belongs in service/helper modules.
- Upload-root paths and PDF upload validation must stay centralized in `core/uploads.py`.
- New vector/document metadata writes must use canonical `doc_id`; `original_doc_uid` is compatibility fallback only on read/delete paths.
- Keep changes incremental and reviewable; avoid large mixed-purpose patches.

## 5. Refactor Workflow (Per Task)
1. Confirm scope and impacted files.
2. Read related tests and docs before editing.
3. Implement minimal patch for one concern at a time.
4. Run focused tests first, then broader checks.
5. Update docs/checklists if behavior or interfaces changed.
6. Record residual risk and follow-up items.

## 6. Validation Baseline
Recommended command order inside `.venv`:

```bash
# lint/style
ruff check .
ruff format .

# tests (start targeted, then broader)
pytest tests/test_markdown_cleaner.py
pytest tests/test_conversations_api.py
pytest tests/test_deep_research.py
pytest
```

If a full suite is too slow or environment-limited, run impacted subsets and state coverage gaps explicitly.

Evaluation compatibility check (when retrieval/agent logic changes):

```bash
python bergen/benchmark_rag_modes.py --limit 1 --modes naive,advanced,graph,agentic --output bergen/benchmark_results_smoke.json
python bergen/evaluate_ragas.py --input bergen/benchmark_results_smoke_ragas.json --output bergen/evaluation_results_smoke.json
```

## 7. Definition of Done (Per Refactor Batch)
A batch is done only if all are satisfied:
- target risk/technical-debt item is resolved
- no known regression in affected tests
- lint/style state is not worsened
- docs are updated when interface/behavior changes
- outstanding risks are listed clearly

## 8. Collaboration Contract
For upcoming complex refactors, execution should follow this sequence:
- propose focused batch
- implement
- run validation
- summarize changes + risks + next batch

This keeps velocity high without losing safety.

## 9. Low-Context Handoff Protocol
When context becomes tight, the next agent should rely on:
- `agent.md`: stable engineering rules and priorities.
- `task.md`: current track specification (what to build and acceptance criteria).
- `implementation_plan.md`: phased execution checklist and progress state.

Keep these three files synchronized before ending a session.

## 10. Current Track Status (2026-02-09)
- Phase 1 (P0), Phase 2 (P1), Phase 3 (P2), and Phase 4 (P3) are completed.
- Phase 5 remains canceled by user decision (`canceled_by_user_20260208`); do not run Bergen/RAGAS commands from that phase.
- Full test regression was run in `.venv` (`python -m pytest`): `225 passed`.
- Current remaining work is optional warning/deprecation cleanup and future corpus-driven markdown edge-case expansion.

## 11. Backend Cleanup Track Status (2026-03-26)
- Completed scope separation for production enforcement via `core/production_scope.py` and static coverage in `tests/test_production_scope.py`.
- Removed high-confidence orphan modules:
  - `pdfserviceMD/markdown_to_pdf.py`
  - `multimodal_rag/utils.py`
- Standardized repository Supabase access through `core/supabase_repository.py`.
- Canonicalized vector metadata writes onto `doc_id` with compatibility fallback helpers in `data_base/document_metadata.py`.
- Extracted production indexing orchestration to `data_base/indexing_service.py`.
- Repaired router/service boundaries by moving GraphRAG extraction into `graph_rag/service.py` and centralizing upload helpers in `core/uploads.py`.
- Added backend boundary guards:
  - `tests/test_router_boundaries.py`
  - `tests/test_uploads.py`
- Full backend verification after this cleanup track: `.\\.venv\\Scripts\\python.exe -m pytest` -> `375 passed`.


## 12. Continuous Learning Notes
- 2026-03-28: When extending persisted or transport dataclasses used by tests and helper runners, preserve backward compatibility for existing constructor call sites by giving new optional fields explicit defaults. Always run the relevant integration tests after schema/dataclass changes to catch silent execution-path regressions.

- 2026-03-28: When a user asks for full CI or "全線" verification, do not stop at targeted tests, lint, or typecheck. Run the workflow-equivalent full suites for both backend and frontend, request sandbox escalation immediately when test runners are blocked, and report exact pass/skip/fail counts rather than partial confidence claims.
- 2026-03-29: Do not persist raw LangChain/Gemini `usage_metadata` directly into evaluation or trace payloads. Normalize token usage first (at minimum flat `input_tokens`, `output_tokens`, `total_tokens`, `reasoning_tokens`), and treat stored `token_usage` schemas as potentially nested/forward-compatible when reading existing rows.
- 2026-03-29: When evaluation `agentic` baseline behavior changes, pin the new planner/drill-down/image constraints inside `evaluation/agentic_evaluation_service.py` and bump `execution_profile` so historical campaign results remain comparable.
- 2026-03-29: When evaluator context-packing logic changes, version it explicitly (`context_policy_version`) and propagate that version through persisted campaign results, RAGAS score details, metrics rows, docs, and frontend types in the same change set. This preserves score comparability and prevents silent evaluator-policy drift.
- 2026-03-29: In this workspace, `rg.exe` can intermittently fail with permission errors; after one failed attempt, switch immediately to `Select-String`/targeted file reads and continue without repeated retries.
- 2026-04-04: For repository-wide string search on Windows, avoid `Get-ChildItem -Recurse | Select-String` because denied cache directories can break scans; prefer `git grep` first, then file-targeted `Select-String`.



