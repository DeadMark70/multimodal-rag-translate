from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from data_base.research_execution_core import ResearchExecutionCore
from data_base.schemas_deep_research import EditableSubTask


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
    assert kwargs["graph_search_mode"] == "generic"
    assert kwargs["graph_execution_hints"]["stage_hint"] == "verification"
    assert kwargs["graph_execution_hints"]["task_type_hint"] == "graph_analysis"
    assert kwargs["graph_execution_hints"]["prefer_local"] is False
