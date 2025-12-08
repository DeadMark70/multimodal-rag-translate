"""
Unit Tests for Synthesizer Module

Tests the ResultSynthesizer for combining sub-task results.
"""

# Standard library
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestSubTaskResult:
    """Tests for SubTaskResult model."""

    def test_create_result(self):
        """Tests creating a sub-task result."""
        from agents.synthesizer import SubTaskResult
        
        result = SubTaskResult(
            task_id=1,
            question="問題",
            answer="答案",
            sources=["doc1"],
        )
        
        assert result.task_id == 1
        assert result.answer == "答案"


class TestResearchReport:
    """Tests for ResearchReport model."""

    def test_create_report(self):
        """Tests creating a research report."""
        from agents.synthesizer import ResearchReport, SubTaskResult
        
        sub_result = SubTaskResult(
            task_id=1, question="問題", answer="答案"
        )
        
        report = ResearchReport(
            original_question="原始問題",
            summary="摘要",
            detailed_answer="詳細答案",
            sub_results=[sub_result],
            all_sources=["doc1"],
        )
        
        assert report.summary == "摘要"
        assert len(report.sub_results) == 1


class TestResultSynthesizerFormatting:
    """Tests for result formatting."""

    def test_format_sub_results(self):
        """Tests formatting sub-results for prompt."""
        from agents.synthesizer import ResultSynthesizer, SubTaskResult
        
        synthesizer = ResultSynthesizer()
        
        results = [
            SubTaskResult(task_id=1, question="問題1", answer="答案1"),
            SubTaskResult(task_id=2, question="問題2", answer="答案2"),
        ]
        
        formatted = synthesizer._format_sub_results(results)
        
        assert "問題 1" in formatted
        assert "問題 2" in formatted
        assert "答案1" in formatted


class TestResultSynthesizerSynthesis:
    """Tests for result synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_empty_results(self):
        """Tests synthesis with empty results."""
        from agents.synthesizer import ResultSynthesizer
        
        synthesizer = ResultSynthesizer()
        report = await synthesizer.synthesize("問題", [])
        
        assert report.confidence == 0.0
        assert "沒有子任務結果" in report.summary

    @pytest.mark.asyncio
    async def test_synthesize_single_result(self):
        """Tests synthesis with single result (no LLM call)."""
        from agents.synthesizer import ResultSynthesizer, SubTaskResult
        
        synthesizer = ResultSynthesizer()
        
        result = SubTaskResult(
            task_id=1,
            question="問題",
            answer="這是詳細的答案內容",
            sources=["doc1"],
        )
        
        report = await synthesizer.synthesize("原始問題", [result])
        
        assert report.detailed_answer == "這是詳細的答案內容"
        assert "doc1" in report.all_sources

    @pytest.mark.asyncio
    async def test_synthesize_multiple_results(self):
        """Tests synthesis with multiple results using mock LLM."""
        from agents.synthesizer import ResultSynthesizer, SubTaskResult
        
        mock_response = MagicMock()
        mock_response.content = """## 摘要
這是綜合摘要。

## 詳細分析
這是詳細的綜合分析內容。"""
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        results = [
            SubTaskResult(task_id=1, question="問題1", answer="答案1", sources=["doc1"]),
            SubTaskResult(task_id=2, question="問題2", answer="答案2", sources=["doc2"]),
        ]
        
        with patch("agents.synthesizer.get_llm", return_value=mock_llm):
            synthesizer = ResultSynthesizer()
            report = await synthesizer.synthesize("原始問題", results)
        
        assert "綜合摘要" in report.summary
        assert len(report.all_sources) == 2

    @pytest.mark.asyncio
    async def test_synthesize_error_fallback(self):
        """Tests fallback on LLM error."""
        from agents.synthesizer import ResultSynthesizer, SubTaskResult
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        results = [
            SubTaskResult(task_id=1, question="問題1", answer="答案1"),
            SubTaskResult(task_id=2, question="問題2", answer="答案2"),
        ]
        
        with patch("agents.synthesizer.get_llm", return_value=mock_llm):
            synthesizer = ResultSynthesizer()
            report = await synthesizer.synthesize("原始問題", results)
        
        # Should fallback to concatenation
        assert report.confidence == 0.5
        assert "答案1" in report.detailed_answer
        assert "答案2" in report.detailed_answer


class TestSynthesizeResultsConvenience:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_disabled_returns_first_result(self):
        """Tests that disabled synthesis returns first result."""
        from agents.synthesizer import synthesize_results, SubTaskResult
        
        result = SubTaskResult(
            task_id=1, question="問題", answer="答案", sources=["doc1"]
        )
        
        report = await synthesize_results("問題", [result], enabled=False)
        
        assert report.detailed_answer == "答案"

    @pytest.mark.asyncio
    async def test_single_result_no_synthesis(self):
        """Tests that single result doesn't trigger synthesis."""
        from agents.synthesizer import synthesize_results, SubTaskResult
        
        result = SubTaskResult(
            task_id=1, question="問題", answer="單一答案"
        )
        
        # Even with enabled=True, single result should not need LLM
        report = await synthesize_results("問題", [result], enabled=True)
        
        assert report.detailed_answer == "單一答案"
