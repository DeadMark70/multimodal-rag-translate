# Specification: Evaluation Pipeline & Ragas Integration

## 1. Overview
This track involves refactoring the existing `experiments` code to implement a robust, automated evaluation framework. We are introducing `ragas` for objective metrics (Faithfulness, Answer Correctness) and building an `EvaluationPipeline` to perform ablation studies across multiple LLMs and RAG configurations.

## 2. Functional Requirements

### 2.1 Evaluation Pipeline Class
- Create a reusable `EvaluationPipeline` class in `experiments/evaluation_pipeline.py`.
- Support comparative testing across multiple models:
    - `gemma-3-27b`
    - `gemini-2.0-flash-lite`
    - `gemini-2.5-flash-lite`
    - `gemini-2.0-flash`
    - `gemini-2.5-flash`

### 2.2 Tiered Benchmarking (Ablation Matrix)
We will use 5 strategic tiers to fully evaluate the system:

1.  **Naive RAG (Weak Baseline)**
    -   `deep_research=False`, `graph=False`
    -   `rerank=False`, `hyde=False`, `multi_query=False`
    -   *Purpose:* Establish the absolute performance floor.

2.  **Advanced RAG (Strong Baseline)**
    -   `deep_research=False`, `graph=False`
    -   `rerank=True`, `hyde=True`, `multi_query=True`
    -   *Purpose:* Represent the current state-of-the-art in traditional RAG.

3.  **Graph RAG (Structured Enhanced)**
    -   `deep_research=False`, `graph=True`
    -   `rerank=True`, `hyde=True`, `multi_query=True`
    -   *Purpose:* Isolate the contribution of Knowledge Graph structure.

4.  **Long Context Mode (The "Context Stuffing" Baseline)**
    -   **Mechanism:** Concatenate all available user PDFs (converted to text/markdown) and pass them directly into the LLM's context window.
    -   **RAG:** Disabled.
    -   *Purpose:* Compare RAG against simply passing all context.

5.  **Full Agentic RAG (Ours - Ultimate)**
    -   `deep_research=True`, `graph=True`
    -   `rerank=True`, `hyde=True`, `multi_query=True`
    -   *Purpose:* Demonstrate maximum capability of Agentic Reasoning.

### 2.3 Ragas Integration
- Integrate the `ragas` library to calculate:
    - **Faithfulness:** Does the answer derive strictly from the retrieved context?
    - **Answer Correctness:** How accurate is the answer compared to the ground truth?
- Use `gemini-3-pro-preview` as the evaluator LLM for Ragas metrics.

### 2.4 Benchmark Execution
- Load questions from `benchmark_questions.json`.
- Support a `type` field for specialized logging.
- **Rate Limit Management:** 1-minute pause before executing **Full Agentic RAG** (Tier 5).

### 2.5 Token & Usage Monitoring
- **Extraction:** Capture `usage_metadata` for every LLM call.
- **Aggregation:** Track tokens per run, per model, and per tier.
- **Reporting:** Include token counts in final reports.

### 2.6 Reporting
- **Nested JSON:** Full hierarchical data.
- **Flattened CSV:** Optimized for analysis.
- **Behavior_Pass Column:** Boolean field indicating if expected tool usage (e.g., visual verification) was observed.

## 3. Specialized Test Scenarios
- **Deep Research Tool Verification:** `nnU-Net` Figure 1 details to verify `visual_verification` tool usage.
- **Synthesis Ability:** Multi-model comparison (SwinUNETR vs nnU-Net).

## 4. Non-Functional Requirements
- **Dynamic Model Overriding:** The pipeline must be able to override the `_DEFAULT_MODEL` in `llm_factory.py`.
- **Resilience:** Handle API errors gracefully.

## 5. Acceptance Criteria
- [ ] `EvaluationPipeline` loops through 5 models and 5 tiers.
- [ ] **Long Context Mode** is implemented and working.
- [ ] Ragas scores calculated via `gemini-3-pro-preview`.
- [ ] **Token usage is accurately recorded.**
- [ ] **`Behavior_Pass` column is present and accurate in the CSV.**
- [ ] Final output saved to `experiments/results/`.
