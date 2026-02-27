import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from data_base.deep_research_service import DeepResearchService
from data_base.schemas_deep_research import ExecutePlanRequest, EditableSubTask, SubTaskExecutionResult

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
         patch('data_base.deep_research_service.synthesize_results') as mock_synth, \
         patch('data_base.deep_research_service.persist_research_conversation', new=AsyncMock()) as mock_persist:
        
        # Mock synthesis result
        mock_report = MagicMock()
        mock_report.summary = "Test Summary"
        mock_report.detailed_answer = "Test Detail"
        mock_report.confidence = 0.9
        mock_synth.return_value = mock_report
        
        request = ExecutePlanRequest(
            original_question="What is the meaning of life?",
            sub_tasks=[EditableSubTask(id=1, question="Life?", enabled=True)],
            conversation_id="test-conv-id",
            enable_drilldown=False,
        )
        
        await service.execute_plan(request, user_id="test-user")
        
        # Verify repository persistence was called
        assert mock_persist.await_count == 1, "persist_research_conversation was not called"

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_execute_plan_persists_to_supabase())
