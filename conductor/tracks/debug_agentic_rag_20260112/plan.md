# Plan: Agentic Logic and Evaluation Logging Fix (Debugging Phase)

## Phase 1: Infrastructure & Token Tracking Fix
- [x] Task: Disable Ragas scoring in `experiments/evaluation_pipeline.py` to streamline debugging. 56a0115
- [x] Task: Refactor `extract_token_usage` in `experiments/evaluation_pipeline.py` to correctly handle newer LangChain `usage_metadata` formats. 130a109
- [x] Task: Implement token usage aggregation for "Full Agentic RAG" (Tier 5) in `experiments/evaluation_pipeline.py`. f758633
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure & Token Tracking Fix' (Protocol in workflow.md)

## Phase 2: Observability & Logging Enhancement
- [ ] Task: Modify `run_tier` in `experiments/evaluation_pipeline.py` to capture and return `thought_process` and `tool_calls`.
- [ ] Task: Update `run_tier` to include raw `retrieved_contexts` (text + metadata) in the returned dictionary.
- [ ] Task: Update `run_full_evaluation` and `save_results_json` to include these new diagnostic fields in the final output.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Observability & Logging Enhancement' (Protocol in workflow.md)

## Phase 3: Agentic Logic & Planner Prompt Tuning
- [ ] Task: Update `agents/planner.py` prompt templates (`_PLANNER_PROMPT`, `_GRAPH_PLANNER_PROMPT`, `_FOLLOWUP_PROMPT`) with a "Strict Visual Requirement" instruction.
- [ ] Task: Ensure the instruction explicitly mandates `vision_tool` usage when text retrieval is insufficient for spatial or visual details.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Agentic Logic & Planner Prompt Tuning' (Protocol in workflow.md)

## Phase 4: Automated Verification
- [ ] Task: Create a dedicated verification script `tests/verify_agentic_fix.py` targeting the "nnU-Net Fig 1" visual verification case.
- [ ] Task: Implement assertions in the script to verify: Tokens > 0, `tool_calls` contains vision tools, and required log fields are present.
- [ ] Task: Execute the verification script and confirm the "Behavior Pass" for the Agentic RAG tier.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Automated Verification' (Protocol in workflow.md)
