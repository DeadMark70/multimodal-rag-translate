"""
Unit Tests for EvaluationPipeline Module

Tests the initialization and core structure of the EvaluationPipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

class TestEvaluationPipelineStructure:
    """Tests for EvaluationPipeline structure."""

    def test_initialization(self):
        """Tests that EvaluationPipeline can be initialized."""
        try:
            from experiments.evaluation_pipeline import EvaluationPipeline
        except ImportError:
            pytest.fail("Could not import EvaluationPipeline from experiments.evaluation_pipeline")
        
        pipeline = EvaluationPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, "models")
        assert hasattr(pipeline, "tiers")
        assert len(pipeline.models) == 5
        assert len(pipeline.tiers) == 5

    def test_models_list(self):
        """Tests that the pipeline supports the required models."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline()
        expected_models = [
            "gemma-3-27b",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash"
        ]
        
        for model in expected_models:
            assert model in pipeline.models

    def test_tiers_list(self):
        """Tests that the pipeline supports the required tiers."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline()
        expected_tiers = [
            "Naive RAG",
            "Advanced RAG",
            "Graph RAG",
            "Long Context Mode",
            "Full Agentic RAG"
        ]
        
        for tier in expected_tiers:
            assert tier in pipeline.tiers

class TestTokenMonitoring:
    """Tests for token monitoring logic."""

    def test_extract_token_usage_success(self):
        """Tests that token usage is correctly extracted from a response object."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from langchain_core.messages import AIMessage
        
        pipeline = EvaluationPipeline()
        
        # Mock response with usage_metadata
        mock_response = AIMessage(
            content="Test response",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30
            }
        )
        
        usage = pipeline.extract_token_usage(mock_response)
        
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_extract_token_usage_missing(self):
        """Tests that missing usage_metadata returns zeros."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from langchain_core.messages import AIMessage
        
        pipeline = EvaluationPipeline()
        
        mock_response = AIMessage(content="No usage info")
        
        usage = pipeline.extract_token_usage(mock_response)
        
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0

class TestBenchmarkQuestions:
    """Tests for benchmark questions loading."""

    def test_benchmark_file_exists(self):
        """Tests that experiments/benchmark_questions.json exists and is valid JSON."""
        import json
        import os
        
        file_path = "experiments/benchmark_questions.json"
        assert os.path.exists(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            
            # Check structure of first item
            item = data[0]
            assert "question" in item
            assert "ground_truth" in item
            assert "type" in item

class TestMockRAG:
    """Tests for mock RAG functionality."""

    @pytest.mark.asyncio
    async def test_mock_rag_answer(self):
        """Tests that mock_rag_answer returns consistent mocked results."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        pipeline = EvaluationPipeline()
        
        question = "Test question"
        answer, contexts = await pipeline.mock_rag_answer(question)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(contexts, list)
        assert len(contexts) > 0
        assert "Mocked context" in contexts[0]

class TestTierExecution:
    """Tests for individual tier execution."""

    @pytest.mark.asyncio
    async def test_run_tier_naive_rag(self):
        """Tests execution of Naive RAG tier."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from data_base.RAG_QA_service import RAGResult
        from langchain_core.documents import Document
        
        pipeline = EvaluationPipeline()
        mock_result = RAGResult(
            answer="Naive answer",
            source_doc_ids=["doc1"],
            documents=[Document(page_content="Naive context")]
        )
        
        with patch("experiments.evaluation_pipeline.rag_answer_question", return_value=mock_result) as mock_rag:
            res = await pipeline.run_tier("Naive RAG", "Question", "gemini-2.0-flash")
            
            assert res["answer"] == "Naive answer"
            assert res["contexts"] == ["Naive context"]
            mock_rag.assert_called_once()
            # Check params
            args, kwargs = mock_rag.call_args
            assert kwargs["enable_reranking"] is False
            assert kwargs["enable_hyde"] is False

    @pytest.mark.asyncio
    async def test_run_tier_advanced_rag(self):
        """Tests execution of Advanced RAG tier."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from data_base.RAG_QA_service import RAGResult
        from langchain_core.documents import Document
        
        pipeline = EvaluationPipeline()
        mock_result = RAGResult(
            answer="Advanced answer",
            source_doc_ids=["doc1"],
            documents=[Document(page_content="Advanced context")]
        )
        
        with patch("experiments.evaluation_pipeline.rag_answer_question", return_value=mock_result) as mock_rag:
            res = await pipeline.run_tier("Advanced RAG", "Question", "gemini-2.0-flash")
            
            assert res["answer"] == "Advanced answer"
            mock_rag.assert_called_once()
            args, kwargs = mock_rag.call_args
            assert kwargs["enable_reranking"] is True
            assert kwargs["enable_hyde"] is True

    @pytest.mark.asyncio
    async def test_run_tier_graph_rag(self):
        """Tests execution of Graph RAG tier."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from data_base.RAG_QA_service import RAGResult
        from langchain_core.documents import Document
        
        pipeline = EvaluationPipeline()
        mock_result = RAGResult(
            answer="Graph answer",
            source_doc_ids=["doc1"],
            documents=[Document(page_content="Graph context")]
        )
        
        with patch("experiments.evaluation_pipeline.rag_answer_question", return_value=mock_result) as mock_rag:
            res = await pipeline.run_tier("Graph RAG", "Question", "gemini-2.0-flash")
            
            assert res["answer"] == "Graph answer"
            mock_rag.assert_called_once()
            args, kwargs = mock_rag.call_args
            assert kwargs["enable_graph_rag"] is True
