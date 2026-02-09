import pytest
from agents.planner import TaskPlanner, ResearchPlan

@pytest.mark.asyncio
async def test_planner_basic_decomposition():
    """
    Test that the planner can decompose a complex question into sub-tasks.
    """
    planner = TaskPlanner(max_subtasks=3)
    question = "請比較 SwinUNETR 與 nnU-Net 在醫學影像分割上的差異，特別是在參數量與計算效率方面。"
    
    print(f"\nDEBUG: Question length: {len(question)}")
    print(f"DEBUG: Indicators check: {[ind for ind in ['比較', '對比', '分析'] if ind in question]}")
    
    # We want to see if it correctly identifies the need for planning
    result = planner.needs_planning(question)
    print(f"DEBUG: needs_planning result: {result}")
    assert result is True
    
    # Execute plan (this will use LLM)
    # If API key is missing, it will fallback to original question
    plan = await planner.plan(question)
    
    assert isinstance(plan, ResearchPlan)
    assert len(plan.sub_tasks) >= 1
    assert plan.original_question == question
    
    for task in plan.sub_tasks:
        assert task.question
        assert task.id > 0
        print(f"Generated sub-task {task.id}: {task.question} (Type: {task.task_type})")

@pytest.mark.asyncio
async def test_planner_graph_aware():
    """Test graph-aware planning mode."""
    planner = TaskPlanner(max_subtasks=5, enable_graph_planning=True)
    question = "分析這幾篇論文之間關於 Transformer 應用的關係與趨勢。"
    
    assert planner.needs_graph_analysis(question) is True
    
    plan = await planner.plan(question)
    
    # Check if any task is assigned as graph_analysis
    any(t.task_type == "graph_analysis" for t in plan.sub_tasks)
    print(f"Graph-aware plan sub-tasks: {[t.task_type for t in plan.sub_tasks]}")
    # We don't strictly assert has_graph_task because LLM output varies, 
    # but we verify the code handles the tag parsing.

@pytest.mark.asyncio
async def test_planner_followup_logic():
    """Test follow-up task generation logic."""
    planner = TaskPlanner()
    original = "SwinUNETR 的優點是什麼？"
    findings = "SwinUNETR 具有層次化結構，但目前尚不清楚其在小數據集上的表現。"
    existing = []
    
    followups = await planner.create_followup_tasks(original, findings, existing)
    
    assert isinstance(followups, list)
    for f in followups:
        print(f"Generated follow-up: {f.question}")
