# Implementation Plan: Evaluation Pipeline & Ragas Integration

This plan outlines the steps to build a robust evaluation framework for the Multimodal Agentic RAG system using `ragas` and tiered benchmarking.

## Phase 1: Foundation and Infrastructure [checkpoint: 3d2534c]
*Setup the core classes, model overriding logic, and token monitoring infrastructure.*

- [x] Task: Create `experiments/evaluation_pipeline.py` with base class structure e54357d
- [x] Task: Implement dynamic model overriding for `get_llm` in `core/llm_factory.py` or within the pipeline 6849088
- [x] Task: Implement Token Monitoring logic to capture `usage_metadata` from LangChain responses 805f205
- [x] Task: Create `experiments/benchmark_questions.json` with initial set of tiered questions (Standard, Visual, Synthesis) 7b81551
- [x] Task: Conductor - User Manual Verification 'Phase 1: Foundation and Infrastructure' (Protocol in workflow.md) 3d2534c

## Phase 2: Ragas & Metric Integration [checkpoint: 7cb83ec]
*Integrate Ragas and set up the evaluator LLM.*

- [x] Task: Write tests for Ragas metric calculation using `gemini-3-pro-preview` 173ae30
- [x] Task: Implement `Faithfulness` and `Answer Correctness` calculation logic 173ae30
- [x] Task: Implement a mock/stub for RAG responses to test metrics in isolation addabf9
- [x] Task: Conductor - User Manual Verification 'Phase 2: Ragas & Metric Integration' (Protocol in workflow.md) 7cb83ec

## Phase 3: Ablation Tier Implementation
*Implement the 5 strategic tiers of the evaluation loop.*

- [x] Task: Implement Tier 1 (Naive RAG) & Tier 2 (Advanced RAG) logic db2d40e
- [ ] Task: Implement Tier 3 (Graph RAG) logic
- [ ] Task: Implement Tier 4 (Long Context Mode) - Read all PDFs and feed to context
- [ ] Task: Implement Tier 5 (Full Agentic RAG) with 1-minute rate-limit pause
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Ablation Tier Implementation' (Protocol in workflow.md)

## Phase 4: Reporting and Behavioral Validation
*Finalize report generation and automated behavioral checks.*

- [ ] Task: Implement Nested JSON report generation (including token counts and tool logs)
- [ ] Task: Implement Flattened CSV report generation with `Behavior_Pass` logic
- [ ] Task: Implement specific behavioral check for `visual_verification` tool usage in nnU-Net tests
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Reporting and Behavioral Validation' (Protocol in workflow.md)

## Phase 5: Execution & Refinement
*Run the full suite and refine based on initial results.*

- [ ] Task: Run full evaluation loop for all 5 models across all 5 tiers
- [ ] Task: Verify CSV/JSON outputs and ensure `Behavior_Pass` is correctly triggered
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Execution & Refinement' (Protocol in workflow.md)
