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
