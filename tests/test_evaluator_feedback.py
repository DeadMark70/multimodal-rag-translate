import pytest
from unittest.mock import MagicMock, patch
from agents.evaluator import RAGEvaluator, DetailedEvaluationResult
from agents.planner import TaskPlanner

@pytest.mark.asyncio
async def test_evaluator_scoring_logic():
    """
    Test the weighted scoring logic of DetailedEvaluationResult.
    """
    # High accuracy, high completeness, high clarity
    res = DetailedEvaluationResult(
        accuracy=9.0,
        completeness=8.0,
        clarity=10.0,
        reason="Excellent answer",
        confidence=0.9
    )
    # Weighted = 9*0.5 + 8*0.3 + 10*0.2 = 4.5 + 2.4 + 2.0 = 8.9
    assert abs(res.weighted_score - 8.9) < 0.01
    assert res.is_passing is True
    assert res.is_reliable is True

    # Low accuracy (Hallucination)
    res_fail = DetailedEvaluationResult(
        accuracy=4.0,
        completeness=10.0,
        clarity=10.0,
        reason="Good structure but contains wrong data",
        confidence=0.5
    )
    assert res_fail.is_passing is False

@pytest.mark.asyncio
async def test_planner_query_refinement():
    """
    Test that the planner generates a different query when evaluation fails.
    """
    planner = TaskPlanner()
    original_q = "Transformer 的參數量是多少？"
    failed_ans = "文中沒有提到參數量。"
    reason = "缺少具體數據，讀者需要確切的模型大小資訊。"
    
    # We'll mock the LLM response for query refinement
    with patch("agents.planner.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value
        # Use AsyncMock for ainvoke
        from unittest.mock import AsyncMock
        mock_llm.ainvoke = AsyncMock()
        
        mock_response = MagicMock()
        mock_response.content = "Transformer 參數量 數據 統計"
        mock_llm.ainvoke.return_value = mock_response
        
        refined = await planner.refine_query_from_evaluation(
            original_q, reason, failed_ans
        )
        
        assert refined != original_q
        assert "數據" in refined or "參數量" in refined
        print(f"Refined query: {refined}")

def test_should_skip_drilldown_logic():
    """
    Verify smart termination logic in deep research.
    """
    from data_base.deep_research_service import DeepResearchService
    from data_base.schemas_deep_research import SubTaskExecutionResult
    
    service = DeepResearchService()
    
    # Mock results: one long/good, one short/bad
    results = [
        SubTaskExecutionResult(id=1, question="Q1", answer="Long answer "*50, sources=["s1"], is_drilldown=False, iteration=0),
        SubTaskExecutionResult(id=2, question="Q2", answer="Too short.", sources=["s2"], is_drilldown=False, iteration=0)
    ]
    
    # Iteration 0 should NEVER skip (Phase 6.1B)
    assert service._should_skip_drilldown(results, current_iteration=0) is False
    
    # Later iterations might skip if ratio is met
    # With 1/2 complete, ratio is 0.5. Default min_complete_ratio is 0.8
    assert service._should_skip_drilldown(results, current_iteration=1) is False
    
    # If all are good
    good_results = [
        SubTaskExecutionResult(id=1, question="Q1", answer="Long answer "*50, sources=["s1"], is_drilldown=False, iteration=0),
        SubTaskExecutionResult(id=2, question="Q2", answer="Another long answer "*50, sources=["s2"], is_drilldown=False, iteration=0)
    ]
    assert service._should_skip_drilldown(good_results, current_iteration=1) is True
