import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agents.planner import TaskPlanner

@pytest.mark.asyncio
async def test_max_subtasks_enforcement():
    """Verify that the planner strictly enforces max_subtasks limit."""
    # Test with 3 subtasks
    planner_3 = TaskPlanner(max_subtasks=3)
    
    with patch("agents.planner.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock LLM returning 10 subtasks
        mock_response = MagicMock()
        mock_response.content = "\n".join([f"{i}. Subtask {i}" for i in range(1, 11)])
        mock_llm.ainvoke.return_value = mock_response
        
        plan = await planner_3.plan("Complex question")
        
        # Should be exactly 3
        assert len(plan.sub_tasks) == 3
        assert plan.sub_tasks[0].id == 1
        assert plan.sub_tasks[2].id == 3

@pytest.mark.asyncio
async def test_subtask_count_1():
    """Verify that the planner works with max_subtasks=1."""
    planner_1 = TaskPlanner(max_subtasks=1)

    with patch("agents.planner.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_get_llm.return_value = mock_llm

        mock_response = MagicMock()
        mock_response.content = "1. First\n2. Second"
        mock_llm.ainvoke.return_value = mock_response

        plan = await planner_1.plan("Question")
        print(f"DEBUG: LLM Content: {mock_response.content!r}")
        print(f"DEBUG: plan.sub_tasks: {[t.question for t in plan.sub_tasks]}")
        assert len(plan.sub_tasks) == 1
        assert plan.sub_tasks[0].question == "First"

def test_internal_parsing_debug():
    """Directly test _parse_subtasks."""
    planner = TaskPlanner(max_subtasks=1)
    response = "1. First\n2. Second"
    subtasks = planner._parse_subtasks(response)
    print(f"DEBUG internal: {[t.question for t in subtasks]}")
    assert len(subtasks) == 1
    assert subtasks[0].question == "First"
@pytest.mark.asyncio
async def test_subtask_parsing_logic():
    """Test the internal _parse_subtasks method for various formats."""
    planner = TaskPlanner(max_subtasks=5)
    
    # Test different number formats and tags
    response = """
    1. [RAG] First question
    2) [GRAPH] Second question
    3. Third question without tag
    4. [Something] Fourth question
    """
    
    subtasks = planner._parse_subtasks(response)
    
    assert len(subtasks) == 4
    assert subtasks[0].task_type == "rag"
    assert subtasks[1].task_type == "graph_analysis"
    assert subtasks[2].task_type == "rag" # Default
    assert subtasks[3].task_type == "rag" # Unrecognized tag defaults to rag
    
    assert subtasks[0].question == "First question"
    assert subtasks[1].question == "Second question"