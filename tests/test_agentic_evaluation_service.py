from unittest.mock import AsyncMock, patch

import pytest

from agents.planner import ResearchPlan, SubTask
from data_base.RAG_QA_service import RAGResult
from data_base.schemas_deep_research import EditableSubTask, ExecutePlanResponse, ResearchPlanResponse
from evaluation.agentic_evaluation_service import (
    AGENTIC_EVAL_PROFILE,
    AGENTIC_IMAGE_ANALYSIS_ENABLED,
    AGENTIC_MAX_DRILLDOWN_ITERATIONS,
    AgenticEvaluationService,
)


@pytest.mark.asyncio
async def test_generate_agentic_plan_caps_initial_plan_to_three_tasks() -> None:
    service = AgenticEvaluationService()
    generated_plan = ResearchPlan(
        original_question="What changed?",
        sub_tasks=[
            SubTask(id=1, question="Task 1", task_type="rag"),
            SubTask(id=2, question="Task 2", task_type="graph_analysis"),
            SubTask(id=3, question="Task 3", task_type="rag"),
            SubTask(id=4, question="Task 4", task_type="rag"),
        ],
        estimated_complexity="complex",
    )

    with patch("evaluation.agentic_evaluation_service.TaskPlanner") as mock_planner_cls:
        mock_planner = mock_planner_cls.return_value
        mock_planner.plan = AsyncMock(return_value=generated_plan)

        response = await service.generate_agentic_plan(question="What changed?", user_id="user-1")

    mock_planner_cls.assert_called_once_with(max_subtasks=3, enable_graph_planning=True)
    assert response.status == "waiting_confirmation"
    assert len(response.sub_tasks) == 3
    assert [task.id for task in response.sub_tasks] == [1, 2, 3]
    assert response.estimated_complexity == "complex"


@pytest.mark.asyncio
async def test_run_case_uses_dedicated_agentic_execution_constraints() -> None:
    service = AgenticEvaluationService(max_concurrent_tasks=3)
    plan_response = ResearchPlanResponse(
        original_question="What changed?",
        sub_tasks=[
            EditableSubTask(id=1, question="Task 1", task_type="rag"),
            EditableSubTask(id=2, question="Task 2", task_type="graph_analysis"),
            EditableSubTask(id=3, question="Task 3", task_type="rag"),
        ],
        estimated_complexity="medium",
        doc_ids=None,
    )
    execute_response = ExecutePlanResponse(
        question="What changed?",
        summary="summary",
        detailed_answer="final answer",
        sub_tasks=[],
        all_sources=[],
        confidence=1.0,
        total_iterations=1,
    )

    service.generate_agentic_plan = AsyncMock(return_value=plan_response)
    service.run_execute_plan = AsyncMock(return_value=execute_response)

    async def passthrough(func, **kwargs):
        return await func(**kwargs)

    with patch("evaluation.agentic_evaluation_service.run_with_retry", new=AsyncMock(side_effect=passthrough)):
        result = await service.run_case(
            question_id="Q1",
            question="What changed?",
            user_id="user-1",
            run_number=1,
        )

    request = service.run_execute_plan.await_args.kwargs["request"]
    assert request.enable_drilldown is True
    assert request.max_iterations == AGENTIC_MAX_DRILLDOWN_ITERATIONS
    assert request.enable_deep_image_analysis is AGENTIC_IMAGE_ANALYSIS_ENABLED
    assert len(request.sub_tasks) == 3
    assert result.answer == "final answer"
    assert result.agent_trace["execution_profile"] == AGENTIC_EVAL_PROFILE
