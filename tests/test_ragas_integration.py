"""
Unit Tests for Ragas Integration

Tests the basic functionality of Ragas metrics with Gemini.
"""

import pytest
from unittest.mock import patch
from experiments.evaluation_pipeline import EvaluationPipeline
from importlib.util import find_spec

class TestRagasIntegration:
    """Tests for Ragas integration."""

    @pytest.mark.asyncio
    async def test_calculate_metrics_success(self):
        """Tests that disabled Ragas path returns zero scores and skips expensive calls."""
        pipeline = EvaluationPipeline()

        with patch("experiments.evaluation_pipeline.evaluate") as mock_evaluate:
            with patch("experiments.evaluation_pipeline.get_embeddings") as mock_get_embeddings:
                scores = await pipeline.calculate_ragas_metrics(
                    question="What is nnU-Net?",
                    answer="nnU-Net is a medical imaging framework.",
                    contexts=["nnU-Net is a framework for medical image segmentation."],
                    ground_truth=(
                        "nnU-Net is a self-configuring framework "
                        "for deep learning-based medical image segmentation."
                    ),
                )

        assert scores["faithfulness"] == 0.0
        assert scores["answer_correctness"] == 0.0
        mock_evaluate.assert_not_called()
        mock_get_embeddings.assert_not_called()

    def test_ragas_imports(self):
        """Tests that ragas can be imported."""
        if find_spec("ragas") is None:
            pytest.fail("Could not import ragas components: ragas package not found")

        from ragas import evaluate
        from ragas.metrics import answer_correctness, faithfulness

        assert evaluate is not None
        assert faithfulness is not None
        assert answer_correctness is not None
