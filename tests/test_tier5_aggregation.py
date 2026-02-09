import pytest
from unittest.mock import patch, MagicMock
from experiments.evaluation_pipeline import EvaluationPipeline

@pytest.mark.asyncio
async def test_run_tier_5_aggregation():
    """Test that Tier 5 correctly aggregates token usage from sub-tasks"""
    pipeline = EvaluationPipeline()
    
    # Mocking deep research service and its methods
    with patch("experiments.evaluation_pipeline.get_deep_research_service") as mock_get_service, \
         patch("experiments.evaluation_pipeline.asyncio.sleep"):
        
        from unittest.mock import AsyncMock
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        
        # Mock generate_plan as AsyncMock
        mock_service.generate_plan = AsyncMock(return_value=MagicMock(sub_tasks=[]))
        
        # Mock execute_plan as AsyncMock
        mock_exec_res = MagicMock()
        mock_service.execute_plan = AsyncMock(return_value=mock_exec_res)
        
        # Mock sub-tasks
        mock_subtask1 = MagicMock()
        mock_subtask1.contexts = ["context1"]
        mock_subtask1.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        
        mock_subtask2 = MagicMock()
        mock_subtask2.contexts = ["context2"]
        mock_subtask2.usage = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        
        mock_exec_res = MagicMock()
        mock_exec_res.sub_tasks = [mock_subtask1, mock_subtask2]
        mock_exec_res.detailed_answer = "detailed answer"
        mock_exec_res.summary = "summary"
        mock_exec_res.all_sources = ["source1"]
        
        mock_service.execute_plan.return_value = mock_exec_res
        
        # Run Tier 5
        result = await pipeline.run_tier("Full Agentic RAG", "test question", "test-model")
        
        # Assertions
        assert result["usage"]["input_tokens"] == 30
        assert result["usage"]["output_tokens"] == 15
        assert result["usage"]["total_tokens"] == 45
        assert len(result["contexts"]) == 2
