# Plan: Core Agentic Workflow Audit & Optimization

## Phase 1: Structural & Static Analysis
Goal: Verify code structure, dependencies, and static quality against the Tech Stack and Style Guides.

- [x] Task: Dependency & Environment Verification 512b41b
    - Check `requirements.txt` vs actual imports.
    - Ensure `.env` structure matches `config.env.example`.
- [x] Task: Static Code Analysis (Agents) 827402e
    - Review `agents/planner.py`, `agents/evaluator.py`, `agents/synthesizer.py`.
    - Check for error handling and type hints.
- [x] Task: Static Code Analysis (RAG & Graph) 192c8e4
    - Review `data_base/` and `graph_rag/`.
    - Verify implementation of GraphRAG logic (NetworkX/Leidenalg usage).
- [ ] Task: Conductor - User Manual Verification 'Structural & Static Analysis' (Protocol in workflow.md)

## Phase 2: Logic & Flow Verification (Dry Run)
Goal: Trace the data flow of a "Research Request" to ensure components interact correctly.

- [ ] Task: Planner Logic Audit
    - Create a test script to simulate a user query.
    - Verify Planner output format and step decomposition.
- [ ] Task: RAG & Context Retrieval Audit
    - Verify `data_base/RAG_QA_service.py` effectively retrieves chunks without overflowing tokens.
    - Check integration of `graph_rag` in the retrieval process.
- [ ] Task: Evaluator & Feedback Loop Audit
    - Verify `agents/evaluator.py` can trigger a "retry" or "correction" in the workflow.
- [ ] Task: Conductor - User Manual Verification 'Logic & Flow Verification' (Protocol in workflow.md)

## Phase 3: Integration Testing & Optimization
Goal: Run a full end-to-end test and apply fixes.

- [ ] Task: Create E2E Test Case
    - Implement `tests/test_full_workflow.py` using `pytest`.
    - Simulate: Upload PDF -> Ask Complex Question -> Verify Answer Quality.
- [ ] Task: Optimization - Critical Fixes
    - Fix any blocking issues identified in Phase 1 & 2.
    - Refactor any "Token Heavy" functions to be more efficient.
- [ ] Task: Conductor - User Manual Verification 'Integration Testing & Optimization' (Protocol in workflow.md)
