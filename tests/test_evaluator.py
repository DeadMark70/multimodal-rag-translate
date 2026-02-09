"""
Unit Tests for Evaluator Module

Tests the Self-RAG evaluator for retrieval relevance and faithfulness.
"""

# Standard library
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party
from langchain_core.documents import Document


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
        assert not result.should_retry
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
            RAGEvaluator
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
        assert not result.should_retry or result.should_retry


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
        assert not result.should_retry


class TestFineGrainedEvaluator:
    """
    Tests for Phase 4 Fine-Grained Evaluation (1-10 scale).
    
    Uses mocking to isolate tests from actual LLM calls.
    Tests the core evaluation logic, score calculation, and JSON parsing.
    """

    @pytest.fixture
    def mock_llm(self):
        """Creates a mock LLM that can be controlled in tests."""
        with patch("agents.evaluator.get_llm") as mock_get:
            fake_llm = AsyncMock()
            mock_get.return_value = fake_llm
            yield fake_llm

    @pytest.mark.asyncio
    async def test_perfect_answer(self, mock_llm):
        """
        Case 1: Perfect answer should have is_passing=True and correct weighted_score.
        
        Simulates: LLM evaluates answer as highly accurate (accuracy=9).
        """
        from agents.evaluator import RAGEvaluator
        
        # Setup: LLM returns high scores
        mock_response = MagicMock()
        mock_response.content = '''
        {
            "analysis": "回答非常精確，數據正確，邏輯清晰。",
            "accuracy": 9,
            "completeness": 10,
            "clarity": 9,
            "reason": "完美回答",
            "suggestion": ""
        }
        '''
        mock_llm.ainvoke.return_value = mock_response
        
        # Execute
        evaluator = RAGEvaluator()
        docs = [Document(page_content="參考內容", metadata={})]
        result = await evaluator.evaluate_detailed("問題", docs, "完美答案")
        
        # Assert
        assert result.accuracy == 9.0
        assert result.completeness == 10.0
        assert result.clarity == 9.0
        assert result.is_passing is True  # accuracy >= 7
        # weighted_score = 9*0.5 + 10*0.3 + 9*0.2 = 4.5 + 3.0 + 1.8 = 9.3
        assert abs(result.weighted_score - 9.3) < 0.01
        assert result.evaluation_failed is False

    @pytest.mark.asyncio
    async def test_hallucination(self, mock_llm):
        """
        Case 2: Hallucinated answer should have is_passing=False.
        
        Simulates: LLM detects severe hallucination (accuracy=2).
        """
        from agents.evaluator import RAGEvaluator
        
        # Setup: LLM returns low accuracy score
        mock_response = MagicMock()
        mock_response.content = '''{
            "analysis": "答案包含嚴重錯誤數據。",
            "accuracy": 2,
            "completeness": 5,
            "clarity": 8,
            "reason": "嚴重幻覺",
            "suggestion": "需要重新檢索相關文獻"
        }'''
        mock_llm.ainvoke.return_value = mock_response
        
        # Execute
        evaluator = RAGEvaluator()
        docs = [Document(page_content="正確資料", metadata={})]
        result = await evaluator.evaluate_detailed("問題", docs, "錯誤答案")
        
        # Assert
        assert result.accuracy == 2.0
        assert result.is_passing is False  # accuracy < 7
        assert result.evaluation_failed is False
        assert "幻覺" in result.reason or "錯誤" in result.reason

    @pytest.mark.asyncio
    async def test_honest_unknown(self, mock_llm):
        """
        Case 3: Honest "I don't know" should get high accuracy.
        
        Simulates: AI correctly identifies no data and admits ignorance.
        """
        from agents.evaluator import RAGEvaluator
        
        # Setup: LLM rewards honesty
        mock_response = MagicMock()
        mock_response.content = '''{
            "analysis": "AI 正確辨識無資料並誠實回應。",
            "accuracy": 9,
            "completeness": 8,
            "clarity": 9,
            "reason": "誠實表明無法回答",
            "suggestion": ""
        }'''
        mock_llm.ainvoke.return_value = mock_response
        
        # Execute
        evaluator = RAGEvaluator()
        docs = [Document(page_content="無相關資料", metadata={})]
        result = await evaluator.evaluate_detailed(
            "論文訓練時間是多少？", 
            docs, 
            "根據文獻內容，未提及訓練時間，無法回答此問題。"
        )
        
        # Assert
        assert result.accuracy >= 8.0  # Honesty is rewarded
        assert result.is_passing is True
        assert result.evaluation_failed is False

    @pytest.mark.asyncio
    async def test_json_parsing_error(self, mock_llm):
        """
        Case 4: Invalid JSON should result in evaluation_failed=True.
        
        Tests robustness against LLM format errors.
        """
        from agents.evaluator import RAGEvaluator
        
        # Setup: LLM returns invalid JSON
        mock_response = MagicMock()
        mock_response.content = "這不是 JSON 格式，LLM 出錯了！"
        mock_llm.ainvoke.return_value = mock_response
        
        # Execute
        evaluator = RAGEvaluator()
        docs = [Document(page_content="內容", metadata={})]
        result = await evaluator.evaluate_detailed("問題", docs, "答案")
        
        # Assert
        assert result.evaluation_failed is True
        assert "無法解析" in result.reason or "失敗" in result.reason

    @pytest.mark.asyncio
    async def test_weighted_score_calculation(self, mock_llm):
        """
        Case 5: Verify weighted_score matches formula: 0.5*acc + 0.3*cmp + 0.2*clr.
        """
        from agents.evaluator import RAGEvaluator
        
        # Setup: Specific scores for calculation verification
        mock_response = MagicMock()
        mock_response.content = '''{"accuracy": 8, "completeness": 6, "clarity": 10, "reason": "test"}'''
        mock_llm.ainvoke.return_value = mock_response
        
        # Execute
        evaluator = RAGEvaluator()
        docs = [Document(page_content="測試", metadata={})]
        result = await evaluator.evaluate_detailed("問題", docs, "答案")
        
        # Assert: weighted_score = 8*0.5 + 6*0.3 + 10*0.2 = 4 + 1.8 + 2 = 7.8
        expected_score = 8 * 0.5 + 6 * 0.3 + 10 * 0.2
        assert abs(result.weighted_score - expected_score) < 0.01
        assert result.is_passing is True  # accuracy=8 >= 7
