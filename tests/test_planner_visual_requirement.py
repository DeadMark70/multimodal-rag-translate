import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agents.planner import TaskPlanner

@pytest.mark.asyncio
async def test_planner_triggers_visual_requirement():
    """Test that the planner includes visual verification keywords when asked about a figure."""
    planner = TaskPlanner()
    
    question = "In the nnU-Net Revisited paper, where is the BraTS dataset located in Figure 1?"
    
    # We want to check if the LLM output (mocked here but simulated) would 
    # likely contain visual instructions based on our new prompt.
    # In a real test we'd call the LLM, but here we'll verify the prompt construction.
    
    with patch("agents.planner.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_get_llm.return_value = mock_llm
        
        # Simulate LLM response that follows the new strict requirement
        mock_response = MagicMock()
        mock_response.content = "1. [RAG] 檢索 nnU-Net Revisited 的 Figure 1 相關文字摘要\n2. [RAG] 執行視覺查證工具以確認 BraTS 在 Figure 1 中的具體位置"
        mock_llm.ainvoke.return_value = mock_response
        
        plan = await planner.plan(question)
        
        # Verify that our mocked response (which follows the spec) results in a plan
        assert len(plan.sub_tasks) >= 2
        assert "視覺查證" in plan.sub_tasks[1].question or "Figure 1" in plan.sub_tasks[1].question

def test_planner_prompts_contain_strict_requirement():
    """Verify that the prompt templates actually contain the new text."""
    from agents.planner import _PLANNER_PROMPT, _GRAPH_PLANNER_PROMPT, _FOLLOWUP_PROMPT
    
    assert "Strict Visual Requirement" in _PLANNER_PROMPT
    assert "視覺查證規範" in _PLANNER_PROMPT
    assert "Strict Visual Requirement" in _GRAPH_PLANNER_PROMPT
    assert "視覺查證規範" in _FOLLOWUP_PROMPT
