# Specification: Agentic Logic and Evaluation Logging Fix (Debugging Phase)

## Overview
This track addresses a "Behavior Failure" in the Agentic RAG system where the Planner fails to invoke vision tools when text retrieval is insufficient. It also fixes a bug in token usage tracking and enhances the evaluation output to support deeper manual analysis by Gemini.

## Functional Requirements

### 1. Agent Logic Enhancement
- **Planner Prompt Update:** Modify `agents/planner.py` (or the relevant system prompt) to include a "Strict Visual Requirement". If text retrieval cannot definitively answer questions about figures or spatial details, the agent MUST call the `vision_tool`.
- **Re-Act Enforcement:** Ensure the "Reflection" step specifically checks if an image is available when text-based answers are vague.

### 2. Evaluation Pipeline Refactoring (`experiments/evaluation_pipeline.py`)
- **Disable Ragas Scoring:** Temporarily comment out or flag the Ragas evaluation step to save cost and speed up iterations.
- **Fix Token Usage:** Correct the logic for extracting `usage_metadata` from LangChain/LLM responses. Ensure `total_tokens` is captured and saved.
- **Enhanced JSON Output:** Expand the evaluation result JSON to include:
    - `thought_process`: The agent's raw Chain of Thought.
    - `tool_calls`: Details of which tools were invoked and with what parameters.
    - `tool_outputs`: The raw response from tools (especially vision/OCR).
    - `retrieved_contexts`: The raw text chunks and metadata retrieved from the vector store (to diagnose context pollution).

## Non-Functional Requirements
- **Traceability:** Every evaluation run must generate a JSON file that is sufficient for root-cause analysis without needing to re-run the code.
- **Performance:** Removing Ragas should significantly reduce the time per evaluation iteration.

## Acceptance Criteria
- [ ] `total_tokens` in the generated JSON is greater than 0.
- [ ] For "visual_verification" question types, the JSON `tool_calls` contains at least one call to a vision-related tool.
- [ ] The JSON output contains `thought_process` and `retrieved_contexts` fields.
- [ ] Automated verification script executes the **Visual Verification question (nnU-Net Fig 1)** and confirms the above three points (Tokens > 0, Vision Tool used, Logs present).

## Out of Scope
- Implementing Metadata Filtering or Reranking (Reserved for the next track).
- Re-enabling Ragas (Reserved for the final validation phase).
