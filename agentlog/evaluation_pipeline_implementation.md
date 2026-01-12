# Evaluation Pipeline & Ragas Integration Log

**Date:** 2026-01-12
**Track:** Evaluation Pipeline & Ragas Integration

## Summary
Successfully implemented a comprehensive automated evaluation pipeline for the Multimodal Agentic RAG system. This pipeline enables tiered benchmarking across multiple LLM models (Gemini 2.0 Flash Lite, Gemini 2.5 Flash Lite) and RAG configurations.

## Key Achievements

### 1. Evaluation Pipeline (`experiments/evaluation_pipeline.py`)
- **Tiered Benchmarking:** Implemented 5 tiers of evaluation:
    1.  **Naive RAG:** Baseline (no reranking, no hybrid search).
    2.  **Advanced RAG:** Reranking + HyDE + Multi-Query.
    3.  **Graph RAG:** Advanced RAG + Knowledge Graph context.
    4.  **Long Context Mode:** Full document context stuffing (baseline).
    5.  **Full Agentic RAG:** Deep Research Service (Planner + Drill-down).
- **Metric Integration:** Integrated `ragas` library (v0.4.2) for objective metrics:
    - **Faithfulness:** Measures grounding in context.
    - **Answer Correctness:** Measures accuracy against ground truth.
- **Evaluator Model:** Configured to use `gemini-2.5-pro` for high-quality metric judgement.
- **Reporting:** Generates both nested JSON (detailed) and flattened CSV (analysis-ready) reports.
- **Behavioral Checks:** Automated verification of tool usage (e.g., visual verification) for specific question types.

### 2. Deep Research Enhancements
- **Context Transparency:** Updated `DeepResearchService` and `RAG_QA_service` to bubble up retrieved text chunks (`contexts`) to the final response.
- **Schema Updates:**
    - Added `contexts` to `SubTaskExecutionResult`.
    - Added `contexts` to `TaskDoneData` (SSE events).
- **Impact:** Enables the frontend to display the exact evidence used for each sub-task, improving transparency.

### 3. Bug Fixes & Refinements
- **Rate Limits:** Implemented 1-minute pause for Deep Research tiers to respect API quotas.
- **Ragas Initialization:** Fixed initialization issues with Ragas 0.4.x by explicitly passing LLM and Embedding wrappers to metrics.
- **Model Selection:** Refined test set to `gemini-2.0-flash-lite` and `gemini-2.5-flash-lite` to avoid rate limits on `gemma`.

## Next Steps
- **Frontend Update:** Update the UI to display the new `contexts` field in the Deep Research results.
- **Long-Running Tests:** Execute full benchmark suites overnight to gather statistically significant data.
