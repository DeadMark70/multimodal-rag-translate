# Plan: RAG System Audit & Experimental Baseline Verification

## Phase 1: Dataset Confirmation & Test Config [checkpoint: ec85fd4]
Goal: Map the user's uploaded files to specific test cases.

- [x] Task: Identify Target Documents 5ccaa41
    - Verify files for SwinUNETR and nnU-Net exist in user `c1bae279-c099-4c45-ba19-2bb393ca4e4b`'s storage.
    - Identify at least 10 irrelevant documents for noise testing.
    - *Note: Confirmed existence of SwinUNETR.pdf and nnU-Net Revisited.pdf. Identified 11 other PDFs (e.g., Attention Is All You Need, SAM-Med3D) for noise testing.*
- [x] Task: Configure Test Environment 5ccaa41
    - Set up `tests/conftest.py` to use the specified `user_id` for integration tests.
- [x] Task: Conductor - User Manual Verification 'Dataset Confirmation & Test Config' (Protocol in workflow.md) ec85fd4

## Phase 2: Component Functionality Audit [checkpoint: deb9692]
Goal: Ensure backend components are active and correctly logging.

- [x] Task: Verify HyDE & Multi-Query 4c88112
- [x] Task: Verify Reranker & Noise Filtering 797fb74
- [x] Task: Verify GraphRAG hybrid search de7a4d0
- [x] Task: Conductor - User Manual Verification 'Component Functionality Audit' (Protocol in workflow.md) deb9692

## Phase 3: Agentic Workflow & Integration [checkpoint: 8238b29]
Goal: Verify Deep Research and Visual Verification logic.

- [x] Task: Verify Sub-task Count Enforcement 6b5f41f
- [x] Task: Verify Visual Verification Trigger 32a724e
- [x] Task: Conductor - User Manual Verification 'Agentic Workflow & Integration' (Protocol in workflow.md) 8238b29

## Phase 4: Baseline Experimentation [checkpoint: 6e4a101]
Goal: Conduct the academic ablation study.

- [x] Task: Run Baseline Comparison 0a174bb
- [x] Task: Evaluation Metrics Audit 0a86f53
- [x] Task: Conductor - User Manual Verification 'Baseline Experimentation' (Protocol in workflow.md) 6e4a101

## Phase 5: Audit Conclusions & Recommendations
Goal: Summarize findings and suggest future improvements.

- [ ] Task: Document Optimization Recommendations
    - [ ] Suggestion: Implement direct Responsible AI sub-metrics (bias, safety) in `RAGEvaluator`.
    - [ ] Suggestion: Further refine `needs_planning` length thresholds for low-resource CJK queries.
    - [ ] Suggestion: Add a "Verbosity" parameter to control report detail levels.
- [ ] Task: Conductor - User Manual Verification 'Audit Conclusions' (Protocol in workflow.md)
