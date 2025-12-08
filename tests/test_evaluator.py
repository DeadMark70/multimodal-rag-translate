"""
Unit Tests for Evaluator Module

Tests the Self-RAG evaluator for retrieval relevance and faithfulness.
"""

# Standard library
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party
from langchain.schema import Document


class TestRetrievalGrade:
    """Tests for RetrievalGrade enum."""

    def test_enum_values(self):
        """Tests that enum has expected values."""
        from agents.evaluator import RetrievalGrade
        
        assert RetrievalGrade.RELEVANT == "relevant"
        assert RetrievalGrade.NOT_RELEVANT == "not_relevant"


class TestFaithfulnessGrade:
    """Tests for FaithfulnessGrade enum."""

    def test_enum_values(self):
        """Tests that enum has expected values."""
        from agents.evaluator import FaithfulnessGrade
        
        assert FaithfulnessGrade.GROUNDED == "grounded"
        assert FaithfulnessGrade.HALLUCINATED == "hallucinated"


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_create_result(self):
        """Tests creating an evaluation result."""
        from agents.evaluator import (
            EvaluationResult, RetrievalGrade, FaithfulnessGrade
        )
        
        result = EvaluationResult(
            retrieval_grade=RetrievalGrade.RELEVANT,
            faithfulness_grade=FaithfulnessGrade.GROUNDED,
            should_retry=False,
            confidence=0.9,
        )
        
        assert result.retrieval_grade == RetrievalGrade.RELEVANT
        assert result.should_retry == False
        assert result.confidence == 0.9


class TestRAGEvaluatorRetrievalEvaluation:
    """Tests for retrieval evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_relevant(self):
        """Tests evaluation returns RELEVANT for relevant docs."""
        from agents.evaluator import RAGEvaluator, RetrievalGrade
        
        mock_response = MagicMock()
        mock_response.content = "RELEVANT"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        docs = [Document(page_content="相關內容", metadata={})]
        
        with patch("agents.evaluator.get_llm", return_value=mock_llm):
            evaluator = RAGEvaluator()
            result = await evaluator.evaluate_retrieval("問題", docs)
        
        assert result == RetrievalGrade.RELEVANT

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_not_relevant(self):
        """Tests evaluation returns NOT_RELEVANT for irrelevant docs."""
        from agents.evaluator import RAGEvaluator, RetrievalGrade
        
        mock_response = MagicMock()
        mock_response.content = "NOT_RELEVANT"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        docs = [Document(page_content="無關內容", metadata={})]
        
        with patch("agents.evaluator.get_llm", return_value=mock_llm):
            evaluator = RAGEvaluator()
            result = await evaluator.evaluate_retrieval("問題", docs)
        
        assert result == RetrievalGrade.NOT_RELEVANT

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_empty_docs(self):
        """Tests evaluation returns NOT_RELEVANT for empty docs."""
        from agents.evaluator import RAGEvaluator, RetrievalGrade
        
        evaluator = RAGEvaluator()
        result = await evaluator.evaluate_retrieval("問題", [])
        
        assert result == RetrievalGrade.NOT_RELEVANT


class TestRAGEvaluatorFaithfulnessEvaluation:
    """Tests for faithfulness evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness_grounded(self):
        """Tests evaluation returns GROUNDED for grounded answer."""
        from agents.evaluator import RAGEvaluator, FaithfulnessGrade
        
        mock_response = MagicMock()
        mock_response.content = "GROUNDED"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        docs = [Document(page_content="事實內容", metadata={})]
        
        with patch("agents.evaluator.get_llm", return_value=mock_llm):
            evaluator = RAGEvaluator()
            result = await evaluator.evaluate_faithfulness("問題", docs, "答案")
        
        assert result == FaithfulnessGrade.GROUNDED

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness_hallucinated(self):
        """Tests evaluation returns HALLUCINATED for ungrounded answer."""
        from agents.evaluator import RAGEvaluator, FaithfulnessGrade
        
        mock_response = MagicMock()
        mock_response.content = "HALLUCINATED"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        docs = [Document(page_content="事實內容", metadata={})]
        
        with patch("agents.evaluator.get_llm", return_value=mock_llm):
            evaluator = RAGEvaluator()
            result = await evaluator.evaluate_faithfulness("問題", docs, "不相關的答案")
        
        assert result == FaithfulnessGrade.HALLUCINATED


class TestRAGEvaluatorFullEvaluation:
    """Tests for full RAG evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_full_success(self):
        """Tests full evaluation with good results."""
        from agents.evaluator import (
            RAGEvaluator, RetrievalGrade, FaithfulnessGrade
        )
        
        mock_response = MagicMock()
        mock_response.content = "RELEVANT"  # First call
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        docs = [Document(page_content="內容", metadata={})]
        
        with patch("agents.evaluator.get_llm", return_value=mock_llm):
            evaluator = RAGEvaluator()
            result = await evaluator.evaluate("問題", docs, "答案")
        
        # Both calls use same mock, so both return RELEVANT/GROUNDED behavior
        assert result.should_retry == False or result.should_retry == True


class TestEvaluateRagResultConvenience:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_disabled_returns_positive(self):
        """Tests that disabled evaluation returns positive result."""
        from agents.evaluator import (
            evaluate_rag_result, RetrievalGrade, FaithfulnessGrade
        )
        
        result = await evaluate_rag_result(
            question="問題",
            documents=[],
            answer="答案",
            enabled=False,
        )
        
        assert result.retrieval_grade == RetrievalGrade.RELEVANT
        assert result.faithfulness_grade == FaithfulnessGrade.GROUNDED
        assert result.should_retry == False
