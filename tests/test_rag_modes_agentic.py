from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from data_base.RAG_QA_service import RAGResult
from evaluation.rag_modes import (
    CONTEXT_POLICY_VERSION,
    EVALUATOR_MAX_CONTEXTS,
    EVALUATOR_MAX_CONTEXT_CHARS,
    _extract_contexts,
    run_campaign_case,
)
from evaluation.schemas import TestCase as EvaluationCase


@pytest.mark.asyncio
async def test_run_campaign_case_graph_uses_generic_graph_mode() -> None:
    test_case = EvaluationCase(
        id="Q1",
        question="What changed?",
        ground_truth="A generic graph path",
        source_docs=[],
        requires_multi_doc_reasoning=False,
    )
    mock_result = RAGResult(
        answer="graph answer",
        source_doc_ids=["doc-1"],
        documents=[Document(page_content="ctx-1")],
        usage={"total_tokens": 21},
    )

    with patch("evaluation.rag_modes.run_with_retry", new=AsyncMock(return_value=mock_result)) as mock_retry:
        result = await run_campaign_case(
            test_case=test_case,
            user_id="user-1",
            mode="graph",
            model_config={
                "model_name": "gemini-2.5-flash",
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
            run_number=1,
        )

    mock_retry.assert_awaited_once()
    _, kwargs = mock_retry.await_args
    assert kwargs["enable_graph_rag"] is True
    assert kwargs["graph_search_mode"] == "generic"
    assert result.answer == "graph answer"
    assert result.contexts == ["ctx-1"]
    assert result.context_policy_version == CONTEXT_POLICY_VERSION


@pytest.mark.asyncio
async def test_run_campaign_case_agentic_uses_evaluation_service_and_profile() -> None:
    test_case = EvaluationCase(
        id="Q1",
        question="What changed?",
        ground_truth="A forked agentic flow",
        source_docs=[],
        requires_multi_doc_reasoning=False,
    )
    mock_result = RAGResult(
        answer="agentic answer",
        source_doc_ids=["doc-1"],
        documents=[Document(page_content="ctx-1", metadata={"task_id": 1})],
        usage={"total_tokens": 55},
        thought_process="summary",
        tool_calls=[],
        agent_trace={
            "trace_id": "trace-1",
            "question_id": "Q1",
            "question": "What changed?",
            "mode": "agentic",
            "execution_profile": "agentic_eval_v4",
            "question_intent": "comparison_disambiguation",
            "strategy_tier": "tier_2_structured_compare",
            "route_profile": "hybrid_compare",
            "required_coverage": ["direct_difference"],
            "coverage_gaps": [],
            "subtask_coverage_status": {"direct_difference": True},
            "run_number": 1,
            "trace_status": "completed",
            "summary": "summary",
            "step_count": 1,
            "tool_call_count": 0,
            "total_tokens": 55,
            "created_at": "2026-03-10T00:00:00+00:00",
            "steps": [],
        },
    )

    with patch("evaluation.rag_modes.AgenticEvaluationService") as mock_service_cls:
        mock_service = mock_service_cls.return_value
        mock_service.run_case = AsyncMock(return_value=mock_result)

        result = await run_campaign_case(
            test_case=test_case,
            user_id="user-1",
            mode="agentic",
            model_config={
                "model_name": "gemini-2.5-flash",
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
            run_number=1,
        )

    mock_service_cls.assert_called_once_with(max_concurrent_tasks=3)
    mock_service.run_case.assert_awaited_once()
    assert result.answer == "agentic answer"
    assert result.contexts == ["ctx-1"]
    assert result.execution_profile == "agentic_eval_v4"
    assert result.context_policy_version == CONTEXT_POLICY_VERSION


def test_extract_contexts_uses_answer_aware_policy_and_preserves_task_coverage() -> None:
    oversized = "x" * (EVALUATOR_MAX_CONTEXT_CHARS + 200)
    documents = [
        Document(
            page_content=f"background only {index} {oversized}",
            metadata={"task_id": f"task-{index}"},
        )
        for index in range(EVALUATOR_MAX_CONTEXTS + 2)
    ]
    documents.insert(
        3,
        Document(
            page_content="medsam domain gap makes generic PEFT assumptions misleading",
            metadata={"task_id": "task-critical"},
        ),
    )
    documents.insert(
        5,
        Document(
            page_content="samed uses lora style adapters for segmentation adaptation",
            metadata={"task_id": "task-support"},
        ),
    )

    contexts = _extract_contexts(
        question="How are MedSAM and SAMed different?",
        answer="MedSAM addresses the domain gap, while SAMed is closer to LoRA or adapter style PEFT.",
        documents=documents,
    )

    assert len(contexts) == EVALUATOR_MAX_CONTEXTS
    assert any("medsam domain gap" in context for context in contexts)
    assert any("lora style adapters" in context for context in contexts)
    assert all("\n" not in context for context in contexts)
    assert all(len(context) <= EVALUATOR_MAX_CONTEXT_CHARS for context in contexts)
