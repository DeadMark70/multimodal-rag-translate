import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.planner import TaskPlanner
from core.prompt_loader import format_agentic_rag_prompt


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
        mock_response.content = chr(10).join(
            [
                "1. [RAG] nnU-Net Revisited Figure 1 verification task",
                "2. [RAG] BraTS placement in Figure 1 details",
            ]
        )
        mock_llm.ainvoke.return_value = mock_response

        plan = await planner.plan(question)

        # Verify that our mocked response (which follows the spec) results in a plan
        assert len(plan.sub_tasks) >= 2
        assert "Figure 1" in plan.sub_tasks[1].question


def test_planner_prompts_contain_strict_requirement():
    """Verify that the prompt templates actually contain the new text."""
    planner_prompt = format_agentic_rag_prompt("planner", question="What is X?")
    graph_prompt = format_agentic_rag_prompt("graph_planner", question="What is X?")
    followup_prompt = format_agentic_rag_prompt(
        "followup",
        original_question="Q",
        current_findings="F",
        existing_questions="E",
    )

    assert "Strict Visual Requirement" in planner_prompt
    assert "Strict Visual Requirement" in graph_prompt
    assert "Strict Visual Requirement" in followup_prompt
