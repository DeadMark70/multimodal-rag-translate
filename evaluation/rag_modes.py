"""Reusable benchmark execution helpers for evaluation campaigns."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.documents import Document

from core.llm_factory import llm_runtime_override
from data_base.RAG_QA_service import RAGResult, rag_answer_question
from evaluation.agentic_evaluation_service import AgenticEvaluationService
from evaluation.retry import run_with_retry
from evaluation.schemas import TestCase

RAG_MODES: dict[str, dict[str, Any]] = {
    "naive": {
        "enable_reranking": False,
        "enable_hyde": False,
        "enable_multi_query": False,
        "enable_graph_rag": False,
        "enable_visual_verification": False,
    },
    "advanced": {
        "enable_reranking": True,
        "enable_hyde": True,
        "enable_multi_query": True,
        "enable_graph_rag": False,
        "enable_visual_verification": False,
    },
    "graph": {
        "enable_reranking": True,
        "enable_hyde": True,
        "enable_multi_query": True,
        "enable_graph_rag": True,
        "graph_search_mode": "generic",
        "enable_visual_verification": False,
    },
    "agentic": {
        "enable_reranking": True,
        "enable_hyde": True,
        "enable_multi_query": True,
        "enable_graph_rag": True,
        "graph_search_mode": "generic",
        "enable_visual_verification": True,
    },
}


@dataclass
class BenchmarkExecutionResult:
    """Normalized result payload consumed by campaign persistence."""

    question_id: str
    question: str
    ground_truth: str
    mode: str
    answer: str
    contexts: list[str]
    source_doc_ids: list[str]
    expected_sources: list[str]
    latency_ms: float
    token_usage: dict[str, int]
    category: Optional[str]
    difficulty: Optional[str]
    error_message: Optional[str] = None
    execution_profile: Optional[str] = None
    agent_trace: Optional[dict[str, Any]] = None


def _runtime_overrides(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": model_config.get("model_name"),
        "temperature": model_config.get("temperature"),
        "top_p": model_config.get("top_p"),
        "top_k": model_config.get("top_k"),
        "max_output_tokens": model_config.get("max_output_tokens"),
    }


async def run_campaign_case(
    *,
    test_case: TestCase,
    user_id: str,
    mode: str,
    model_config: dict[str, Any],
    run_number: int = 1,
) -> BenchmarkExecutionResult:
    """Execute one test case under one RAG mode."""
    if mode not in RAG_MODES:
        raise ValueError(f"Unsupported RAG mode: {mode}")

    with llm_runtime_override(**_runtime_overrides(model_config)):
        start_time = time.perf_counter()
        if mode == "agentic":
            result = await _run_agentic_case(
                question_id=test_case.id,
                question=test_case.question,
                user_id=user_id,
                run_number=run_number,
            )
        else:
            rag_result = await run_with_retry(
                rag_answer_question,
                question=test_case.question,
                user_id=user_id,
                return_docs=True,
                **RAG_MODES[mode],
            )
            assert isinstance(rag_result, RAGResult)
            result = rag_result
        latency_ms = (time.perf_counter() - start_time) * 1000

    contexts = _extract_contexts(result.documents)
    return BenchmarkExecutionResult(
        question_id=test_case.id,
        question=test_case.question,
        ground_truth=test_case.ground_truth,
        mode=mode,
        answer=result.answer,
        contexts=contexts,
        source_doc_ids=list(result.source_doc_ids),
        expected_sources=list(test_case.source_docs),
        latency_ms=latency_ms,
        token_usage=dict(result.usage or {}),
        category=test_case.category,
        difficulty=test_case.difficulty,
        execution_profile=(result.agent_trace or {}).get("execution_profile"),
        agent_trace=result.agent_trace,
    )


async def _run_agentic_case(
    *,
    question_id: str,
    question: str,
    user_id: str,
    run_number: int,
) -> RAGResult:
    service = AgenticEvaluationService(max_concurrent_tasks=3)
    return await service.run_case(
        question_id=question_id,
        question=question,
        user_id=user_id,
        run_number=run_number,
    )


def _extract_contexts(documents: list[Document]) -> list[str]:
    contexts: list[str] = []
    for document in documents:
        if hasattr(document, "page_content"):
            contexts.append(document.page_content[:500])
    return contexts
