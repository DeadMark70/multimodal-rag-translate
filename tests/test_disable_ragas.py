import pytest
from unittest.mock import patch, MagicMock
from experiments.evaluation_pipeline import EvaluationPipeline

@pytest.mark.asyncio
async def test_calculate_ragas_metrics_disabled():
    """
    Test that calculate_ragas_metrics is temporarily disabled and does not call ragas.evaluate.
    """
    pipeline = EvaluationPipeline()
    
    # Mock data
    question = "test question"
    answer = "test answer"
    contexts = ["context1"]
    ground_truth = "test ground truth"
    
    # Mock dependencies to pass checks
    with patch("experiments.evaluation_pipeline.get_embeddings") as mock_get_embeddings, \
         patch("experiments.evaluation_pipeline.get_llm") as mock_get_llm, \
         patch("experiments.evaluation_pipeline.evaluate") as mock_evaluate:
        
        # Setup mocks
        mock_get_embeddings.return_value = MagicMock()
        mock_get_llm.return_value = MagicMock()
        
        # Configure mock to return something if it were called
        mock_evaluate.return_value = {"faithfulness": [0.5], "answer_correctness": [0.5]}
        
        # Call the method
        result = await pipeline.calculate_ragas_metrics(question, answer, contexts, ground_truth)
        
        # Assertions
        # 1. It should NOT call evaluate (This is what we want to achieve)
        mock_evaluate.assert_not_called()
        
        # 2. It should return default scores
        assert result["faithfulness"] == 0.0
        assert result["answer_correctness"] == 0.0

