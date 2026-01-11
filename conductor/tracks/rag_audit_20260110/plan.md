# Plan: RAG System Audit & Experimental Baseline Verification

## Phase 1: Dataset Confirmation & Test Config
Goal: Map the user's uploaded files to specific test cases.

- [ ] Task: Identify Target Documents
    - Verify files for SwinUNETR and nnU-Net exist in user `c1bae279-c099-4c45-ba19-2bb393ca4e4b`'s storage.
    - Identify at least 10 irrelevant documents for noise testing.
- [ ] Task: Configure Test Environment
    - Set up `tests/conftest.py` to use the specified `user_id` for integration tests.
- [ ] Task: Conductor - User Manual Verification 'Dataset Confirmation & Test Config' (Protocol in workflow.md)

## Phase 2: Component Functionality Audit
Goal: Ensure backend components are active and correctly logging.

- [ ] Task: Verify HyDE & Multi-Query
    - Create `tests/test_query_transform.py`.
    - Assert internal query expansion is happening.
- [ ] Task: Verify Reranker & Noise Filtering
    - Create `tests/test_reranker_logic.py`.
    - Assert top-k chunks belong to relevant documents despite noise.
- [ ] Task: Verify GraphRAG hybrid search
    - Create `tests/test_graphrag_integration.py`.
    - Assert graph context is retrieved for relational queries.
- [ ] Task: Conductor - User Manual Verification 'Component Functionality Audit' (Protocol in workflow.md)

## Phase 3: Agentic Workflow & Integration
Goal: Verify Deep Research and Visual Verification logic.

- [ ] Task: Verify Sub-task Count Enforcement
    - Create `tests/test_deep_research_constraints.py`.
    - Verify that N tasks (1, 5, 10) are strictly generated as per params.
- [ ] Task: Verify Visual Verification Trigger
    - Create `tests/test_visual_tool_trigger.py`.
    - Assert that queries about image details trigger the `VERIFY_IMAGE` tool.
- [ ] Task: Conductor - User Manual Verification 'Agentic Workflow & Integration' (Protocol in workflow.md)

## Phase 4: Baseline Experimentation
Goal: Conduct the academic ablation study.

- [ ] Task: Run Baseline Comparison
    - Implement `experiments/audit_report.py`.
    - Compare: Vanilla LLM vs Naive RAG vs Full Agentic RAG.
- [ ] Task: Evaluation Metrics Audit
    - Run RAGAS metrics on the results and verify the current calculation logic.
- [ ] Task: Conductor - User Manual Verification 'Baseline Experimentation' (Protocol in workflow.md)
