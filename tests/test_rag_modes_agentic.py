from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from data_base.RAG_QA_service import RAGResult
from evaluation.agentic_evaluation_service import AGENTIC_EVAL_PROFILE
from evaluation.campaign_engine import CampaignEngine
from evaluation.retrieval_profiles import (
    ADVANCED_EVAL_PROFILE,
    GRAPH_ABLATION_MODES,
    GRAPH_EVAL_PROFILE,
    evaluation_execution_profile,
)
from evaluation.rag_modes import (
    AGENTIC_CONTEXT_POLICY_VERSION,
    CONTEXT_POLICY_VERSION,
    EVALUATOR_MAX_CONTEXTS,
    EVALUATOR_MAX_CONTEXT_CHARS,
    RAG_MODES,
    _extract_contexts,
    run_campaign_case,
)
from evaluation.schemas import TestCase as EvaluationCase


@pytest.mark.asyncio
async def test_run_campaign_case_naive_uses_plain_mode() -> None:
    test_case = EvaluationCase(
        id="Q0",
        question="Explain plain baseline",
        ground_truth="Plain mode should be enabled",
        source_docs=[],
        requires_multi_doc_reasoning=False,
    )
    mock_result = RAGResult(
        answer="naive answer",
        source_doc_ids=["doc-1"],
        documents=[Document(page_content="ctx-naive")],
        usage={"total_tokens": 12},
    )

    with patch(
        "evaluation.rag_modes.run_with_retry", new=AsyncMock(return_value=mock_result)
    ) as mock_retry:
        await run_campaign_case(
            test_case=test_case,
            user_id="user-1",
            mode="naive",
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
    assert kwargs["plain_mode"] is True
    assert kwargs["enable_hyde"] is False
    assert kwargs["enable_multi_query"] is False
    assert kwargs["enable_graph_rag"] is False


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

    with patch(
        "evaluation.rag_modes.run_with_retry", new=AsyncMock(return_value=mock_result)
    ) as mock_retry:
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
@pytest.mark.parametrize(
    ("mode", "expected_evidence_mode", "expected_flags"),
    [
        (
            "graph_raw_current",
            "raw_current",
            {"graph_raw_current_enabled": True, "graph_auto_gate_enabled": False},
        ),
        (
            "graph_locator_to_chunk",
            "locator_to_chunk",
            {"graph_to_chunk_enabled": True, "graph_auto_gate_enabled": False},
        ),
        (
            "router_auto_graph",
            "router_auto",
            {"graph_to_chunk_enabled": True, "graph_auto_gate_enabled": True},
        ),
    ],
)
async def test_graph_evaluation_modes_pass_explicit_execution_snapshots(
    mode: str,
    expected_evidence_mode: str,
    expected_flags: dict[str, bool],
) -> None:
    test_case = EvaluationCase(
        id="Q-graph-mode",
        question="Compare the claim scope across papers",
        ground_truth="Graph mode snapshot",
        source_docs=[],
        requires_multi_doc_reasoning=False,
    )
    mock_result = RAGResult(
        answer="answer",
        source_doc_ids=["doc-1"],
        documents=[Document(page_content="ctx")],
    )

    with patch(
        "evaluation.rag_modes.run_with_retry", new=AsyncMock(return_value=mock_result)
    ) as mock_retry:
        await run_campaign_case(
            test_case=test_case,
            user_id="user-1",
            mode=mode,
            model_config={"model_name": "gemini-2.5-flash"},
        )

    _, kwargs = mock_retry.await_args
    hints = kwargs["graph_execution_hints"]
    assert hints["graph_evidence_mode"] == expected_evidence_mode
    assert all(
        hints["graph_feature_flags"][key] is value
        for key, value in expected_flags.items()
    )


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
            "execution_profile": AGENTIC_EVAL_PROFILE,
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
    assert result.execution_profile == AGENTIC_EVAL_PROFILE
    assert result.context_policy_version == AGENTIC_CONTEXT_POLICY_VERSION


@pytest.mark.asyncio
async def test_v9_campaign_case_uses_the_typed_v9_runtime_not_the_v8_service() -> None:
    test_case = EvaluationCase(
        id="Q-v9",
        question="What is the reported score?",
        ground_truth="0.91",
        source_docs=["doc-1"],
        requires_multi_doc_reasoning=False,
    )
    v9_result = RAGResult(
        answer="0.91",
        source_doc_ids=["doc-1"],
        documents=[Document(page_content="0.91", metadata={"doc_id": "doc-1"})],
        agent_trace={
            "agentic_execution_version": "v9",
            "response_status": "complete",
        },
    )
    with (
        patch("evaluation.rag_modes.AgenticV9CampaignRuntime") as runtime_cls,
        patch("evaluation.rag_modes.AgenticEvaluationService") as v8_service_cls,
    ):
        runtime_cls.return_value.execute = AsyncMock(return_value=v9_result)
        result = await run_campaign_case(
            test_case=test_case,
            user_id="user-1",
            mode="agentic-v9",
            model_config={"max_input_tokens": 4096, "max_output_tokens": 256},
            run_number=1,
            agentic_execution_version="v9",
        )

    runtime_cls.return_value.execute.assert_awaited_once()
    v8_service_cls.assert_not_called()
    assert result.agentic_execution_version == "v9"


@pytest.mark.asyncio
async def test_mixed_v9_campaign_keeps_naive_baseline_on_v8_identity() -> None:
    """A campaign-wide v9 setting must not relabel a naive baseline as v9."""
    test_case = EvaluationCase(
        id="Q-mixed-naive",
        question="What is the baseline answer?",
        ground_truth="baseline",
        source_docs=[],
        requires_multi_doc_reasoning=False,
    )
    naive_result = RAGResult(
        answer="baseline",
        source_doc_ids=[],
        documents=[Document(page_content="baseline")],
        usage={"total_tokens": 4},
    )

    with patch(
        "evaluation.rag_modes.run_with_retry", new=AsyncMock(return_value=naive_result)
    ):
        result = await run_campaign_case(
            test_case=test_case,
            user_id="user-1",
            mode="naive",
            model_config={"max_output_tokens": 64},
            agentic_execution_version="v9",
        )

    assert result.mode == "naive"
    assert result.execution_identity == "naive"
    assert result.agentic_execution_version == "v8"


def test_mixed_v9_campaign_builds_v8_baselines_and_v9_agentic_unit() -> None:
    test_case = EvaluationCase(
        id="Q-mixed-units",
        question="Which identity is used?",
        ground_truth="identity",
        source_docs=[],
        requires_multi_doc_reasoning=False,
    )

    units = CampaignEngine._build_units(
        test_cases=[test_case],
        modes=["agentic-v9", "naive", "graph"],
        repeat_count=1,
        agentic_execution_version="v9",
    )

    assert {
        unit.mode: unit.agentic_execution_version for unit in units
    } == {"agentic-v9": "v9", "naive": "v8", "graph": "v8"}


def test_extract_contexts_uses_answer_aware_policy_and_preserves_task_coverage() -> (
    None
):
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


def test_all_changed_evaluation_modes_disable_hyde() -> None:
    changed_modes = {"advanced", "graph", "agentic", *GRAPH_ABLATION_MODES}
    assert changed_modes.issubset(RAG_MODES)
    assert all(RAG_MODES[mode]["enable_hyde"] is False for mode in changed_modes)


def test_advanced_and_main_graph_use_multi_query() -> None:
    assert RAG_MODES["advanced"]["enable_multi_query"] is True
    assert RAG_MODES["graph"]["enable_multi_query"] is True


def test_main_graph_uses_locator_to_chunk_policy() -> None:
    hints = RAG_MODES["graph"]["graph_execution_hints"]
    assert hints["graph_evidence_mode"] == "locator_to_chunk"
    assert hints["graph_feature_flags"] == {
        "graph_raw_current_enabled": False,
        "graph_evidence_locator_enabled": True,
        "graph_provenance_gate_enabled": True,
        "graph_to_chunk_enabled": True,
        "graph_auto_gate_enabled": False,
    }
    assert (
        RAG_MODES["graph_raw_current"]["graph_execution_hints"]["graph_evidence_mode"]
        == "raw_current"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_profile"),
    [
        ("advanced", ADVANCED_EVAL_PROFILE),
        ("graph", GRAPH_EVAL_PROFILE),
        (
            "graph_locator_to_chunk",
            evaluation_execution_profile("graph_locator_to_chunk"),
        ),
    ],
)
async def test_changed_standard_modes_persist_execution_profile(
    mode: str,
    expected_profile: str | None,
) -> None:
    assert expected_profile is not None
    test_case = EvaluationCase(
        id="Q-profile",
        question="Compare models",
        ground_truth="comparison",
        source_docs=[],
        requires_multi_doc_reasoning=False,
    )
    mock_result = RAGResult(
        answer="answer",
        source_doc_ids=["doc-1"],
        documents=[Document(page_content="context")],
    )
    with patch(
        "evaluation.rag_modes.run_with_retry",
        new=AsyncMock(return_value=mock_result),
    ):
        result = await run_campaign_case(
            test_case=test_case,
            user_id="user-1",
            mode=mode,
            model_config={"model_name": "gemini-2.5-flash"},
        )

    assert result.execution_profile == expected_profile
