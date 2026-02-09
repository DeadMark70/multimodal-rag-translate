import pytest
from unittest.mock import MagicMock, patch
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
         patch('data_base.deep_research_service.supabase') as mock_supabase:
        
        # Mock synthesis result
        mock_report = MagicMock()
        mock_report.summary = "Test Summary"
        mock_report.detailed_answer = "Test Detail"
        mock_report.confidence = 0.9
        mock_synth.return_value = mock_report
        
        # Mock supabase update response
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_update = MagicMock()
        mock_table.update.return_value = mock_update
        mock_eq1 = MagicMock()
        mock_update.eq.return_value = mock_eq1
        mock_eq2 = MagicMock()
        mock_eq1.eq.return_value = mock_eq2
        
        request = ExecutePlanRequest(
            original_question="What is the meaning of life?",
            sub_tasks=[EditableSubTask(id=1, question="Life?", enabled=True)],
            conversation_id="test-conv-id"
        )
        
        await service.execute_plan(request, user_id="test-user")
        
        # Verify supabase.update was called
        mock_supabase.table.assert_any_call("conversations")
        
        # This test will fail if we haven't implemented the logic yet
        assert mock_table.update.called, "supabase.table('conversations').update() was not called"

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_execute_plan_persists_to_supabase())