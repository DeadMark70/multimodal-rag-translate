from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_base.RAG_QA_service import RAGResult
from data_base.deep_research_service import DeepResearchService
from data_base.schemas_deep_research import (
    EditableSubTask,
    ExecutePlanRequest,
    SubTaskExecutionResult,
)


def _build_request() -> ExecutePlanRequest:
    return ExecutePlanRequest(
        original_question="What is the meaning of life?",
        sub_tasks=[EditableSubTask(id=1, question="Life?", enabled=True)],
        conversation_id="test-conv-id",
        enable_drilldown=False,
    )


def _build_report() -> MagicMock:
    mock_report = MagicMock()
    mock_report.summary = "Test Summary"
    mock_report.detailed_answer = "Test Detail"
    mock_report.confidence = 0.9
    return mock_report


def _assert_persist_payload(mock_persist: AsyncMock) -> None:
    assert mock_persist.await_count == 1, "persist_research_conversation was not called"
    kwargs = mock_persist.await_args.kwargs
    assert kwargs["conversation_id"] == "test-conv-id"
    assert kwargs["user_id"] == "test-user"
    assert kwargs["title"] == "What is the meaning of life?"
    assert kwargs["metadata"]["original_question"] == "What is the meaning of life?"
    assert kwargs["metadata"]["result"]["summary"] == "Test Summary"
    assert kwargs["metadata"]["result"]["detailed_answer"] == "Test Detail"
    assert kwargs["metadata"]["result"]["question"] == "What is the meaning of life?"
    assert kwargs["metadata"]["result"]["sub_tasks"][0]["question"] == "Life?"
    assert kwargs["metadata"]["result"]["all_sources"] == ["doc1"]

@pytest.mark.asyncio
async def test_execute_plan_persists_to_supabase():
    """
    Test that execute_plan calls supabase.update when conversation_id is provided.
    """
    service = DeepResearchService()
    
    # Mock execute_tasks and synthesize_results to avoid real LLM calls
    mock_subtask_result = SubTaskExecutionResult(
        id=1,
        question="Life?",
        answer="42",
        sources=["doc1"]
    )
    
    with patch.object(service, '_execute_tasks', return_value=[mock_subtask_result]), \
         patch('data_base.research_execution_core.synthesize_results') as mock_synth, \
         patch('data_base.deep_research_service.persist_research_conversation', new=AsyncMock()) as mock_persist:
        mock_synth.return_value = _build_report()

        request = _build_request()

        await service.execute_plan(request, user_id="test-user")

        _assert_persist_payload(mock_persist)


@pytest.mark.asyncio
async def test_execute_plan_streaming_persists_to_supabase():
    service = DeepResearchService()

    mock_subtask_result = SubTaskExecutionResult(
        id=1,
        question="Life?",
        answer="42",
        sources=["doc1"],
    )

    with patch.object(service, "_execute_single_task", new=AsyncMock(return_value=mock_subtask_result)), \
         patch("data_base.research_execution_core.synthesize_results") as mock_synth, \
         patch("data_base.deep_research_service.persist_research_conversation", new=AsyncMock()) as mock_persist:
        mock_synth.return_value = _build_report()

        request = _build_request()
        events = [event async for event in service.execute_plan_streaming(request, user_id="test-user")]

        assert events[-1]["event"] == "complete"
        _assert_persist_payload(mock_persist)


@pytest.mark.asyncio
async def test_execute_plan_streaming_forwards_deep_image_analysis_flag():
    service = DeepResearchService()

    request = _build_request()
    request.enable_deep_image_analysis = True
    mock_subtask_result = SubTaskExecutionResult(
        id=1,
        question="Life?",
        answer="42",
        sources=["doc1"],
    )

    with patch.object(service, "_execute_single_task", new=AsyncMock(return_value=mock_subtask_result)) as mock_execute_single, \
         patch("data_base.research_execution_core.synthesize_results") as mock_synth, \
         patch("data_base.deep_research_service.persist_research_conversation", new=AsyncMock()):
        mock_synth.return_value = _build_report()

        _ = [event async for event in service.execute_plan_streaming(request, user_id="test-user")]

        assert mock_execute_single.await_count == 1
        kwargs = mock_execute_single.await_args.kwargs
        assert kwargs["enable_deep_image_analysis"] is True


@pytest.mark.asyncio
async def test_execute_plan_streaming_emits_task_phase_updates():
    service = DeepResearchService()

    async def fake_rag_answer_question(**kwargs):
        progress_callback = kwargs["progress_callback"]
        await progress_callback("retrieval", {"query_count": 1})
        await progress_callback("reranking", {"document_count": 5})
        return RAGResult(
            answer="42",
            source_doc_ids=["doc1"],
            documents=[],
            usage={},
        )

    with patch(
        "data_base.deep_research_service.rag_answer_question",
        new=AsyncMock(side_effect=fake_rag_answer_question),
    ), patch("data_base.research_execution_core.synthesize_results") as mock_synth, patch(
        "data_base.deep_research_service.persist_research_conversation",
        new=AsyncMock(),
    ):
        mock_synth.return_value = _build_report()

        request = _build_request()
        events = [event async for event in service.execute_plan_streaming(request, user_id="test-user")]

    phase_events = [event for event in events if event["event"] == "task_phase_update"]
    assert len(phase_events) == 2
    assert '"stage": "retrieval"' in phase_events[0]["data"]
    assert '"stage": "reranking"' in phase_events[1]["data"]
    assert '"id": 1' in phase_events[0]["data"]

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_execute_plan_persists_to_supabase())
