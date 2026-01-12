"""
Unit Tests for Ragas Integration

Tests the basic functionality of Ragas metrics with Gemini.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from experiments.evaluation_pipeline import EvaluationPipeline

class TestRagasIntegration:
    """Tests for Ragas integration."""

    @pytest.mark.asyncio
    async def test_calculate_metrics_success(self):
        """Tests that calculate_metrics returns expected scores."""
        pipeline = EvaluationPipeline()
        
        # Mocking Ragas evaluate function to avoid actual LLM calls
        mock_result = {
            "faithfulness": 0.9,
            "answer_correctness": 0.8
        }
        
        with patch("experiments.evaluation_pipeline.evaluate", return_value=mock_result):
            with patch("experiments.evaluation_pipeline.get_embeddings") as mock_get_embeddings:
                mock_embeddings = MagicMock()
                mock_get_embeddings.return_value = mock_embeddings
                
                # These are stubs/mocks for the parameters
                question = "What is nnU-Net?"
                answer = "nnU-Net is a medical imaging framework."
                contexts = ["nnU-Net is a framework for medical image segmentation."]
                ground_truth = "nnU-Net is a self-configuring framework for deep learning-based medical image segmentation."
                
                scores = await pipeline.calculate_ragas_metrics(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth
                )
                assert scores["faithfulness"] == 0.9
                assert scores["answer_correctness"] == 0.8

    def test_ragas_imports(self):
        """Tests that ragas can be imported."""
        try:
            import ragas
            from ragas.metrics import faithfulness, answer_correctness
            from ragas import evaluate
        except ImportError as e:
            pytest.fail(f"Could not import ragas components: {e}")
