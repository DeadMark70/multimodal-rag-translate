import asyncio
import pytest
from data_base.deep_research_service import DeepResearchService
from data_base.schemas_deep_research import ExecutePlanRequest, EditableSubTask

@pytest.mark.asyncio
async def test_deep_research_full_workflow():
    """
    End-to-End Test: Planning -> Execution -> Synthesis.
    Using a real DeepResearchService instance.
    """
    service = DeepResearchService()
    user_id = "test-user-id-001"
    question = "請分析 SwinUNETR 的核心架構，並說明其相較於傳統 UNet 的優勢。"
    
    # 1. Test Planning Phase
    # Note: enable_graph_planning=True to test the complex path
    plan_res = await service.generate_plan(question, user_id, enable_graph_planning=True)
    
    assert plan_res.status == "waiting_confirmation"
    assert len(plan_res.sub_tasks) >= 2
    print(f"\n[Step 1] Generated {len(plan_res.sub_tasks)} sub-tasks.")
    
    # 2. Test Execution Phase (Simulated confirmation)
    # We take the generated sub-tasks and create an execution request
    exec_request = ExecutePlanRequest(
        original_question=question,
        sub_tasks=plan_res.sub_tasks,
        enable_drilldown=True,
        max_iterations=1,
        enable_reranking=False # Disable for faster test
    )
    
    # Run execution
    # This will perform RAG lookups. If user has no docs, it will return empty/failure results,
    # which is also a valid path to test.
    print(f"[Step 2] Executing plan...")
    exec_res = await service.execute_plan(exec_request, user_id)
    
    # 3. Verify Results
    assert exec_res.question == question
    assert exec_res.summary != ""
    assert isinstance(exec_res.sub_tasks, list)
    
    print(f"[Step 3] Execution finished. Total iterations: {exec_res.total_iterations}")
    print(f"Summary: {exec_res.summary[:100]}...")
    
    # Check if confidence exists
    assert exec_res.confidence >= 0.0

@pytest.mark.asyncio
async def test_deep_research_no_tasks():
    """Verify behavior when no tasks are enabled."""
    service = DeepResearchService()
    user_id = "test-user"
    request = ExecutePlanRequest(
        original_question="Test",
        sub_tasks=[EditableSubTask(id=1, question="Q1", task_type="rag", enabled=False)],
        enable_drilldown=False
    )
    
    res = await service.execute_plan(request, user_id)
    assert "沒有啟用的子任務" in res.summary
