import pytest
from unittest.mock import MagicMock
from experiments.evaluation_pipeline import EvaluationPipeline

def test_extract_token_usage_new_format():
    """Test extraction from 'usage_metadata' (LangChain 0.2+)"""
    pipeline = EvaluationPipeline()
    
    response = MagicMock()
    response.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150
    }
    # Ensure llm_output is absent to simulate new format exclusivity or precedence
    del response.llm_output 
    
    usage = pipeline.extract_token_usage(response)
    
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 150

def test_extract_token_usage_old_format():
    """Test extraction from 'llm_output' (LangChain Legacy)"""
    pipeline = EvaluationPipeline()
    
    response = MagicMock()
    # Simulate missing usage_metadata
    del response.usage_metadata
    
    response.llm_output = {
        "token_usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100
        }
    }
    
    usage = pipeline.extract_token_usage(response)
    
    assert usage["input_tokens"] == 80
    assert usage["output_tokens"] == 20
    assert usage["total_tokens"] == 100

def test_extract_token_usage_missing():
    """Test when no usage info is available"""
    pipeline = EvaluationPipeline()
    
    response = MagicMock()
    del response.usage_metadata
    del response.llm_output
    
    usage = pipeline.extract_token_usage(response)
    
    assert usage["total_tokens"] == 0
