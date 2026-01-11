"""
Unit Tests for LLM Factory Model Overriding

Tests the ability to dynamically override the model in get_llm.
"""

import pytest
from core.llm_factory import get_llm, clear_llm_cache

def test_get_llm_default_model():
    """Tests that get_llm returns the default model when no override is provided."""
    clear_llm_cache()
    llm = get_llm("rag_qa")
    # _DEFAULT_MODEL is "gemma-3-27b-it" in the current file
    assert llm.model == "gemma-3-27b-it"

def test_get_llm_override_model():
    """Tests that get_llm returns the overridden model when provided."""
    clear_llm_cache()
    # Attempting to use a model name that is not the default
    override_model = "gemini-2.0-flash"
    
    # This should fail if the parameter doesn't exist yet
    try:
        llm = get_llm("rag_qa", model_name=override_model)
        assert llm.model == override_model
    except TypeError as e:
        if "unexpected keyword argument 'model_name'" in str(e):
            pytest.fail("get_llm does not yet support 'model_name' override")
        raise e
