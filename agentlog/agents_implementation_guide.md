# AI Agents (agents) Technical Documentation

## 1. Technical Implementation Details

### Core Logic
The `agents` module implements specialized AI roles using the "Agentic RAG" pattern. It breaks down complex reasoning tasks into specialized components: Planning, Evaluation, and Synthesis.

1.  **Task Planner (`planner.py`)**:
    -   **Function**: Decomposes complex research questions into 2-5 atomic sub-tasks.
    -   **Multi-Strategy**: Supports standard RAG (vector search) and GraphRAG (relation analysis) task types.
    -   **Visual Requirement**: Proactively identifies needs for "Visual Verification" when queries involve charts or figures.
    -   **Smart Refinement**: Can refine search queries based on previous evaluation failures (Smart Retry).
    -   **Drill-down**: Generates follow-up tasks based on current findings to explore knowledge gaps.

2.  **Self-RAG Evaluator (`evaluator.py`)**:
    -   **Phase 4 Academic Engine**: Evaluates answers on a 1-10 scale across three dimensions:
        -   **Accuracy (50%)**: Data precision and citation correctness.
        -   **Completeness (30%)**: Coverage of sub-aspects.
        -   **Clarity (20%)**: Logical structure.
    -   **Conflict Awareness**: Explicitly checks for conflicting information between documents and penalizes "hedging" (vague conclusions).
    -   **Grounding**: Verifies if answers are "Grounded", "Hallucinated", or "Uncertain".
    -   **Comparison**: Supports "Arena" mode to compare RAG output against Pure LLM output.

3.  **Result Synthesizer (`synthesizer.py`)**:
    -   **Multi-Source Integration**: Merges results from multiple sub-tasks into a unified report.
    -   **Conflict Arbitration**: Uses a `<think>` block to weight evidence (Benchmark > Single Experiment, New > Old) before writing the final response.
    -   **Academic Formatting**: Produces structured reports with Executive Summaries, Key Findings, and Detailed Analysis.
    -   **Confidence Calibration**: Adjusts final confidence scores based on detected conflicts in the source material.

## 2. Codebase Map

| File Path | Responsibility |
| :--- | :--- |
| `agents/planner.py` | Task decomposition, query refinement, and follow-up generation. |
| `agents/evaluator.py` | 1-10 academic evaluation, faithfulness checking, and conflict awareness. |
| `agents/synthesizer.py` | Result merging, conflict arbitration, and report generation. |

## 3. Usage Guide

**⚠️ IMPORTANT: All commands must be executed within the project's virtual environment (`.venv`).**

### Logic Flow
1.  **Plan**: `ResearchPlan = await TaskPlanner().plan(question)`
2.  **Execute**: (Handled by `data_base.deep_research_service`)
3.  **Evaluate**: `DetailedEvaluationResult = await RAGEvaluator().evaluate_detailed(question, docs, answer)`
4.  **Synthesize**: `ResearchReport = await ResultSynthesizer().synthesize(question, sub_results)`

## 4. Dependencies

### Internal Modules
-   `core.llm_factory`: Provides optimized LLM instances for each agent role.

### External Libraries
-   `langchain`: Orchestration of LLM calls.
-   `pydantic`: Schema definition and validation.
