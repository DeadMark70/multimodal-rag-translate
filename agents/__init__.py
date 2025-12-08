"""
Agents Package

Provides agentic capabilities for the RAG system:
- Self-RAG evaluation (retrieval relevance, faithfulness)
- Plan-and-Solve (task decomposition, result synthesis)
"""

from agents.evaluator import (
    RetrievalGrade,
    FaithfulnessGrade,
    EvaluationResult,
    RAGEvaluator,
    evaluate_rag_result,
)
from agents.planner import (
    SubTask,
    ResearchPlan,
    TaskPlanner,
    plan_research,
)
from agents.synthesizer import (
    SubTaskResult,
    ResearchReport,
    ResultSynthesizer,
    synthesize_results,
)

__all__ = [
    # Evaluator
    "RetrievalGrade",
    "FaithfulnessGrade",
    "EvaluationResult",
    "RAGEvaluator",
    "evaluate_rag_result",
    # Planner
    "SubTask",
    "ResearchPlan",
    "TaskPlanner",
    "plan_research",
    # Synthesizer
    "SubTaskResult",
    "ResearchReport",
    "ResultSynthesizer",
    "synthesize_results",
]
