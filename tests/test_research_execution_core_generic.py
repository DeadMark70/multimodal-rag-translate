from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from data_base.research_execution_core import ResearchExecutionCore
from data_base.schemas_deep_research import AtomicFact, EditableSubTask, SubTaskExecutionResult


@pytest.mark.asyncio
async def test_execute_tasks_uses_generic_mode_for_initial_graph_tasks() -> None:
    core = ResearchExecutionCore()
    task = EditableSubTask(
        id=1,
        question="Analyze the relationship between model A and model B",
        task_type="graph_analysis",
        enabled=True,
    )

    mock_result = SimpleNamespace(
        answer="answer",
        source_doc_ids=["doc-1"],
        documents=[],
        usage={"total_tokens": 10},
        thought_process=None,
        tool_calls=[],
    )

    with patch(
        "data_base.research_execution_core.rag_answer_question",
        new=AsyncMock(return_value=mock_result),
    ) as mock_rag:
        await core._execute_tasks(
            tasks=[task],
            user_id="test-user",
            doc_ids=None,
            enable_reranking=True,
            iteration=0,
        )

    kwargs = mock_rag.await_args.kwargs
    assert kwargs["enable_crag"] is True
    assert kwargs["graph_search_mode"] == "generic"
    assert kwargs["graph_execution_hints"]["stage_hint"] == "exploration"
    assert kwargs["graph_execution_hints"]["task_type_hint"] == "graph_analysis"
    assert kwargs["graph_execution_hints"]["prefer_global"] is True


@pytest.mark.asyncio
async def test_execute_single_task_uses_verification_hint_for_followups() -> None:
    core = ResearchExecutionCore()
    task = EditableSubTask(
        id=2,
        question="Compare the quantitative evidence for model A and model B",
        task_type="graph_analysis",
        enabled=True,
    )

    mock_result = SimpleNamespace(
        answer="answer",
        source_doc_ids=["doc-1"],
        documents=[],
        usage={"total_tokens": 12},
        thought_process=None,
        tool_calls=[],
    )

    with patch(
        "data_base.research_execution_core.rag_answer_question",
        new=AsyncMock(return_value=mock_result),
    ) as mock_rag:
        await core._execute_single_task(
            task=task,
            user_id="test-user",
            doc_ids=None,
            enable_reranking=True,
            iteration=1,
        )

    kwargs = mock_rag.await_args.kwargs
    assert kwargs["enable_crag"] is True
    assert kwargs["graph_search_mode"] == "generic"
    assert kwargs["graph_execution_hints"]["stage_hint"] == "verification"
    assert kwargs["graph_execution_hints"]["task_type_hint"] == "graph_analysis"
    assert kwargs["graph_execution_hints"]["prefer_local"] is False


@pytest.mark.asyncio
async def test_execute_tasks_preserves_visual_verification_metadata() -> None:
    core = ResearchExecutionCore()
    task = EditableSubTask(
        id=3,
        question="Inspect figure details",
        task_type="rag",
        enabled=True,
    )
    mock_result = SimpleNamespace(
        answer="The chart shows XYZ.",
        source_doc_ids=["doc-visual"],
        documents=[],
        usage={"total_tokens": 8},
        thought_process=None,
        tool_calls=[
            {
                "action": "VERIFY_IMAGE",
                "question": "What does XYZ mean?",
                "success": True,
                "result": "XYZ means Cross-Year Zonal Yield.",
            }
        ],
        visual_verification_meta={
            "visual_verification_attempted": True,
            "visual_tool_call_count": 1,
            "visual_force_fallback_used": False,
        },
    )

    with patch(
        "data_base.research_execution_core.rag_answer_question",
        new=AsyncMock(return_value=mock_result),
    ):
        results = await core._execute_tasks(
            tasks=[task],
            user_id="test-user",
            doc_ids=None,
            enable_reranking=True,
            iteration=0,
        )

    assert len(results) == 1
    assert results[0].visual_verification_meta["visual_verification_attempted"] is True
    assert results[0].tool_calls[0]["action"] == "VERIFY_IMAGE"


def test_build_findings_summary_prefers_structured_fact_state() -> None:
    core = ResearchExecutionCore()
    results = [
        SubTaskExecutionResult(
            id=1,
            question="What is LoRA?",
            answer="LoRA is a PEFT technique.",
            sources=["doc-1"],
            contexts=["ctx"],
        )
    ]
    fact_state = [
        AtomicFact(
            claim="LoRA reduces trainable parameters via low-rank adapters.",
            source_doc_ids=["doc-1"],
        )
    ]

    summary = core._build_findings_summary(results, fact_state=fact_state)

    assert "Structured Fact State" in summary
    assert "LoRA reduces trainable parameters via low-rank adapters." in summary
    assert "Task Coverage Snapshot" in summary


def test_build_findings_summary_includes_visual_findings_for_multimodal_followup() -> None:
    core = ResearchExecutionCore()
    results = [
        SubTaskExecutionResult(
            id=2,
            question="What does XYZ mean in Figure 2?",
            answer="Figure-based answer.",
            sources=["doc-2"],
            contexts=["ctx"],
            tool_calls=[
                {
                    "action": "VERIFY_IMAGE",
                    "question": "Identify XYZ in Figure 2",
                    "success": True,
                    "result": "Figure 2 labels XYZ as Cross-Year Zonal Yield with 12.4% improvement.",
                }
            ],
            visual_verification_meta={
                "visual_verification_attempted": True,
                "visual_tool_call_count": 1,
                "visual_force_fallback_used": False,
            },
        )
    ]
    fact_state = [
        AtomicFact(
            claim="The figure reports an XYZ metric with 12.4% gain.",
            source_doc_ids=["doc-2"],
        )
    ]

    summary = core._build_findings_summary(results, fact_state=fact_state)

    assert "Visual Verification Findings" in summary
    assert "potential_terms=XYZ" in summary
    assert "Multimodal Follow-up Requirement" in summary


@pytest.mark.asyncio
async def test_drill_down_loop_passes_structured_fact_state_to_planner() -> None:
    core = ResearchExecutionCore()
    current_results = [
        SubTaskExecutionResult(
            id=1,
            question="What is LoRA?",
            answer="LoRA is a PEFT technique.",
            sources=["doc-1"],
            contexts=["ctx"],
        )
    ]

    with patch("data_base.research_execution_core.TaskPlanner") as mock_planner_cls, patch(
        "data_base.research_execution_core.RAGEvaluator"
    ), patch.object(
        core,
        "_extract_atomic_facts",
        new=AsyncMock(
            return_value=[
                AtomicFact(
                    claim="LoRA uses low-rank adapters to reduce trainable parameters.",
                    source_doc_ids=["doc-1"],
                )
            ]
        ),
    ):
        mock_planner = mock_planner_cls.return_value
        mock_planner.create_followup_tasks = AsyncMock(return_value=[])

        iterations = await core._drill_down_loop(
            original_question="Explain LoRA",
            current_results=current_results,
            user_id="user-1",
            doc_ids=None,
            enable_reranking=True,
            max_iterations=1,
        )

    assert iterations == 0
    current_findings = mock_planner.create_followup_tasks.await_args.kwargs["current_findings"]
    assert "Structured Fact State" in current_findings
    assert "LoRA uses low-rank adapters to reduce trainable parameters." in current_findings


@pytest.mark.asyncio
async def test_drill_down_loop_passes_visual_findings_to_planner_context() -> None:
    core = ResearchExecutionCore()
    current_results = [
        SubTaskExecutionResult(
            id=1,
            question="Explain the chart abbreviation",
            answer="The chart references XYZ.",
            sources=["doc-9"],
            contexts=["ctx"],
            tool_calls=[
                {
                    "action": "VERIFY_IMAGE",
                    "question": "What does XYZ stand for?",
                    "success": True,
                    "result": "XYZ stands for Cross-Year Zonal Yield in Figure 4.",
                }
            ],
            visual_verification_meta={
                "visual_verification_attempted": True,
                "visual_tool_call_count": 1,
                "visual_force_fallback_used": False,
            },
        )
    ]

    with patch("data_base.research_execution_core.TaskPlanner") as mock_planner_cls, patch(
        "data_base.research_execution_core.RAGEvaluator"
    ), patch.object(
        core,
        "_extract_atomic_facts",
        new=AsyncMock(
            return_value=[
                AtomicFact(
                    claim="The visual evidence introduces acronym XYZ.",
                    source_doc_ids=["doc-9"],
                )
            ]
        ),
    ):
        mock_planner = mock_planner_cls.return_value
        mock_planner.create_followup_tasks = AsyncMock(return_value=[])

        iterations = await core._drill_down_loop(
            original_question="Analyze the chart evidence",
            current_results=current_results,
            user_id="user-1",
            doc_ids=None,
            enable_reranking=True,
            max_iterations=1,
        )

    assert iterations == 0
    current_findings = mock_planner.create_followup_tasks.await_args.kwargs["current_findings"]
    assert "Visual Verification Findings" in current_findings
    assert "potential_terms=XYZ" in current_findings
