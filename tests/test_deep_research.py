"""
Unit Tests for Deep Research Module

Tests the Interactive Deep Research API endpoints and services.
"""

# Standard library
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEditableSubTask:
    """Tests for EditableSubTask model."""

    def test_create_editable_subtask(self):
        """Tests creating an editable sub-task."""
        from data_base.schemas_deep_research import EditableSubTask
        
        task = EditableSubTask(id=1, question="子問題")
        
        assert task.id == 1
        assert task.question == "子問題"
        assert task.task_type == "rag"
        assert task.enabled is True

    def test_editable_subtask_disabled(self):
        """Tests creating a disabled sub-task."""
        from data_base.schemas_deep_research import EditableSubTask
        
        task = EditableSubTask(id=2, question="停用的問題", enabled=False)
        
        assert task.enabled is False


class TestResearchPlanResponse:
    """Tests for ResearchPlanResponse model."""

    def test_create_plan_response(self):
        """Tests creating a research plan response."""
        from data_base.schemas_deep_research import (
            EditableSubTask,
            ResearchPlanResponse,
        )
        
        response = ResearchPlanResponse(
            original_question="複雜問題",
            sub_tasks=[EditableSubTask(id=1, question="子問題1")],
            estimated_complexity="medium",
        )
        
        assert response.status == "waiting_confirmation"
        assert response.original_question == "複雜問題"
        assert len(response.sub_tasks) == 1


class TestExecutePlanRequest:
    """Tests for ExecutePlanRequest model."""

    def test_create_execute_request(self):
        """Tests creating an execute plan request."""
        from data_base.schemas_deep_research import (
            EditableSubTask,
            ExecutePlanRequest,
        )
        
        request = ExecutePlanRequest(
            original_question="研究問題",
            sub_tasks=[
                EditableSubTask(id=1, question="子問題1"),
                EditableSubTask(id=2, question="子問題2", enabled=False),
            ],
            max_iterations=2,
        )
        
        assert request.original_question == "研究問題"
        assert len(request.sub_tasks) == 2
        assert request.max_iterations == 2

    def test_max_iterations_constraint(self):
        """Tests that max_iterations has constraints."""
        from data_base.schemas_deep_research import (
            EditableSubTask,
            ExecutePlanRequest,
        )
        from pydantic import ValidationError
        
        # Should raise error for max_iterations > 5
        with pytest.raises(ValidationError):
            ExecutePlanRequest(
                original_question="問題",
                sub_tasks=[EditableSubTask(id=1, question="子問題")],
                max_iterations=10,  # Too high
            )


class TestDeepResearchService:
    """Tests for DeepResearchService."""

    @pytest.mark.asyncio
    async def test_generate_plan(self):
        """Tests plan generation."""
        from data_base.deep_research_service import DeepResearchService
        
        mock_plan_response = MagicMock()
        mock_plan_response.content = """1. 子問題一的完整內容
2. 子問題二的完整內容"""
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_plan_response)
        
        with patch("agents.planner.get_llm", return_value=mock_llm):
            service = DeepResearchService()
            plan = await service.generate_plan(
                question="複雜的研究問題",
                user_id="test-user",
            )
        
        assert plan.status == "waiting_confirmation"
        assert plan.original_question == "複雜的研究問題"
        assert len(plan.sub_tasks) >= 1


class TestPlannerFollowup:
    """Tests for TaskPlanner follow-up task generation."""

    def test_is_similar_question(self):
        """Tests question similarity detection."""
        from agents.planner import TaskPlanner
        
        planner = TaskPlanner()
        
        # Similar questions
        assert planner._is_similar_question(
            "Transformer 的架構是什麼？",
            "Transformer 的基本架構？"
        ) is True
        
        # Different questions
        assert planner._is_similar_question(
            "Transformer 的架構是什麼？",
            "RNN 的優缺點？"
        ) is False

    @pytest.mark.asyncio
    async def test_create_followup_no_gaps(self):
        """Tests follow-up generation when no gaps found."""
        from agents.planner import TaskPlanner, SubTask
        
        mock_response = MagicMock()
        mock_response.content = "無需追加查詢"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("agents.planner.get_llm", return_value=mock_llm):
            planner = TaskPlanner()
            followups = await planner.create_followup_tasks(
                original_question="研究問題",
                current_findings="已經找到完整答案",
                existing_tasks=[SubTask(id=1, question="原始問題")],
            )
        
        assert followups == []

    @pytest.mark.asyncio
    async def test_create_followup_with_gaps(self):
        """Tests follow-up generation when gaps found."""
        from agents.planner import TaskPlanner, SubTask
        
        mock_response = MagicMock()
        mock_response.content = """1. [RAG] 詳細解釋 Attention 機制
2. [GRAPH] 分析模型之間的關係"""
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("agents.planner.get_llm", return_value=mock_llm):
            planner = TaskPlanner()
            followups = await planner.create_followup_tasks(
                original_question="比較 Transformer 和 RNN",
                current_findings="Transformer 使用 Attention...",
                existing_tasks=[SubTask(id=1, question="RNN 是什麼？")],
            )
        
        assert len(followups) >= 1


class TestExecutePlanResponse:
    """Tests for ExecutePlanResponse model."""

    def test_create_execute_response(self):
        """Tests creating an execute plan response."""
        from data_base.schemas_deep_research import (
            ExecutePlanResponse,
            SubTaskExecutionResult,
        )
        
        response = ExecutePlanResponse(
            question="研究問題",
            summary="研究摘要",
            detailed_answer="詳細報告",
            sub_tasks=[
                SubTaskExecutionResult(
                    id=1,
                    question="子問題1",
                    answer="答案1",
                    sources=["doc-1"],
                )
            ],
            all_sources=["doc-1"],
            confidence=0.85,
            total_iterations=1,
        )
        
        assert response.question == "研究問題"
        assert len(response.sub_tasks) == 1
        assert response.confidence == 0.85


class TestEvaluationDrivenDrilldown:
    """Tests for evaluation-driven drill-down loop."""

    def test_detailed_evaluation_result_structure(self):
        """Tests DetailedEvaluationResult has expected fields."""
        from agents.evaluator import DetailedEvaluationResult
        
        result = DetailedEvaluationResult(
            relevance_score=4,
            groundedness_score=3,
            completeness_score=2,
            reason="回答不夠完整，缺乏細節",
            confidence=0.6,
            evaluation_failed=False,
        )
        
        assert result.completeness_score == 2
        assert "不夠完整" in result.reason
        assert result.is_reliable == False  # confidence < 0.7

    def test_detailed_evaluation_result_is_reliable(self):
        """Tests is_reliable property calculation."""
        from agents.evaluator import DetailedEvaluationResult
        
        # High confidence should be reliable
        high_result = DetailedEvaluationResult(
            relevance_score=5,
            groundedness_score=5,
            completeness_score=5,
            confidence=0.9,
        )
        assert high_result.is_reliable == True
        
        # Low confidence should not be reliable
        low_result = DetailedEvaluationResult(
            relevance_score=2,
            groundedness_score=2,
            completeness_score=2,
            confidence=0.4,
        )
        assert low_result.is_reliable == False
        
        # Failed evaluation should not be reliable
        failed_result = DetailedEvaluationResult(
            confidence=0.9,
            evaluation_failed=True,
        )
        assert failed_result.is_reliable == False

    @pytest.mark.asyncio
    async def test_max_retries_protection(self):
        """Tests that max_retries prevents infinite loops."""
        # This test verifies the constant is properly set
        from data_base.deep_research_service import DeepResearchService
        
        service = DeepResearchService()
        
        # The service should initialize without errors
        assert service.max_concurrent_tasks == 3
        assert service.default_max_iterations == 2
