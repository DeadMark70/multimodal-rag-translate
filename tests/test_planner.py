"""
Unit Tests for Planner Module

Tests the TaskPlanner for research question decomposition.
"""

# Standard library
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestSubTask:
    """Tests for SubTask model."""

    def test_create_subtask(self):
        """Tests creating a sub-task."""
        from agents.planner import SubTask
        
        task = SubTask(id=1, question="子問題")
        
        assert task.id == 1
        assert task.question == "子問題"
        assert task.priority == 1


class TestResearchPlan:
    """Tests for ResearchPlan model."""

    def test_create_plan(self):
        """Tests creating a research plan."""
        from agents.planner import ResearchPlan, SubTask
        
        plan = ResearchPlan(
            original_question="原始問題",
            sub_tasks=[SubTask(id=1, question="子問題")],
            estimated_complexity="medium",
        )
        
        assert plan.original_question == "原始問題"
        assert len(plan.sub_tasks) == 1


class TestTaskPlannerParsing:
    """Tests for sub-task parsing."""

    def test_parse_numbered_list(self):
        """Tests parsing numbered list response."""
        from agents.planner import TaskPlanner
        
        planner = TaskPlanner()
        
        response = """1. 第一個子問題是什麼？
2. 第二個子問題是什麼？
3. 第三個子問題是什麼？"""
        
        subtasks = planner._parse_subtasks(response)
        
        assert len(subtasks) == 3
        assert subtasks[0].id == 1
        assert "第一個" in subtasks[0].question

    def test_parse_empty_response(self):
        """Tests parsing empty response."""
        from agents.planner import TaskPlanner
        
        planner = TaskPlanner()
        subtasks = planner._parse_subtasks("")
        
        assert subtasks == []

    def test_parse_respects_max_subtasks(self):
        """Tests that max_subtasks limit is respected."""
        from agents.planner import TaskPlanner
        
        planner = TaskPlanner(max_subtasks=2)
        
        response = """1. 子問題一的內容
2. 子問題二的內容
3. 子問題三的內容
4. 子問題四的內容"""
        
        subtasks = planner._parse_subtasks(response)
        
        assert len(subtasks) == 2


class TestTaskPlannerPlanning:
    """Tests for the planning functionality."""

    @pytest.mark.asyncio
    async def test_plan_success(self):
        """Tests successful planning with mock LLM."""
        from agents.planner import TaskPlanner
        
        mock_response = MagicMock()
        mock_response.content = """1. 子問題一的完整內容
2. 子問題二的完整內容"""
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("agents.planner.get_llm", return_value=mock_llm):
            planner = TaskPlanner()
            plan = await planner.plan("複雜的研究問題")
        
        assert plan.original_question == "複雜的研究問題"
        assert len(plan.sub_tasks) >= 1

    @pytest.mark.asyncio
    async def test_plan_fallback_on_error(self):
        """Tests fallback to original question on error."""
        from agents.planner import TaskPlanner
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        with patch("agents.planner.get_llm", return_value=mock_llm):
            planner = TaskPlanner()
            plan = await planner.plan("問題")
        
        # Should fallback to original question
        assert len(plan.sub_tasks) == 1
        assert plan.sub_tasks[0].question == "問題"


class TestTaskPlannerComplexityCheck:
    """Tests for complexity heuristics."""

    def test_needs_planning_simple_question(self):
        """Tests that simple questions don't need planning."""
        from agents.planner import TaskPlanner
        
        planner = TaskPlanner()
        
        assert planner.needs_planning("什麼是 Python？") == False

    def test_needs_planning_complex_question(self):
        """Tests that complex questions need planning."""
        from agents.planner import TaskPlanner
        
        planner = TaskPlanner()
        
        complex_q = "比較 Python 和 JavaScript 的優缺點，並分析它們在網頁開發中的應用"
        assert planner.needs_planning(complex_q) == True


class TestPlanResearchConvenience:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_disabled_returns_single_task(self):
        """Tests that disabled planning returns single task."""
        from agents.planner import plan_research
        
        plan = await plan_research("問題", enabled=False)
        
        assert len(plan.sub_tasks) == 1
        assert plan.sub_tasks[0].question == "問題"


class TestRefineQueryFromEvaluation:
    """Tests for smart query refinement based on evaluation reason."""

    @pytest.mark.asyncio
    async def test_refine_query_outdated_data(self):
        """Tests that 'outdated data' reason adds time qualifiers."""
        from agents.planner import TaskPlanner
        
        mock_response = MagicMock()
        mock_response.content = "最新的 Transformer 架構發展"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("agents.planner.get_llm", return_value=mock_llm):
            planner = TaskPlanner()
            refined = await planner.refine_query_from_evaluation(
                original_question="Transformer 架構是什麼？",
                evaluation_reason="資料太舊，缺乏最新發展",
                failed_answer="Transformer 是 2017 年提出的..."
            )
        
        assert "最新" in refined or "recent" in refined.lower()

    @pytest.mark.asyncio
    async def test_refine_query_lack_of_data(self):
        """Tests that 'lack of data' reason focuses on statistics."""
        from agents.planner import TaskPlanner
        
        mock_response = MagicMock()
        mock_response.content = "深度學習模型效能數據比較"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("agents.planner.get_llm", return_value=mock_llm):
            planner = TaskPlanner()
            refined = await planner.refine_query_from_evaluation(
                original_question="深度學習模型的效能？",
                evaluation_reason="缺乏具體數據，沒有量化指標",
                failed_answer="深度學習模型效能很好..."
            )
        
        assert "數據" in refined or "statistics" in refined.lower()

    @pytest.mark.asyncio
    async def test_refine_query_different_from_original(self):
        """Tests that refined query is different from original."""
        from agents.planner import TaskPlanner
        
        mock_response = MagicMock()
        mock_response.content = "CNN 和 RNN 的差異比較"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("agents.planner.get_llm", return_value=mock_llm):
            planner = TaskPlanner()
            original = "神經網路類型？"
            refined = await planner.refine_query_from_evaluation(
                original_question=original,
                evaluation_reason="範圍太廣，需要更具體",
                failed_answer="神經網路有很多類型..."
            )
        
        assert refined != original

    @pytest.mark.asyncio
    async def test_refine_query_fallback_on_error(self):
        """Tests that refinement falls back to original on error."""
        from agents.planner import TaskPlanner
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API Error"))
        
        with patch("agents.planner.get_llm", return_value=mock_llm):
            planner = TaskPlanner()
            original = "原始問題"
            refined = await planner.refine_query_from_evaluation(
                original_question=original,
                evaluation_reason="任何原因",
                failed_answer="失敗的答案"
            )
        
        assert refined == original
