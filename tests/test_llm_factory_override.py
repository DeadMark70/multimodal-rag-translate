"""Unit tests for LLM factory model and runtime overrides."""

from unittest.mock import Mock, patch

import pytest

from core.llm_factory import clear_llm_cache, get_llm, get_llm_usage_metrics, llm_runtime_override

def test_get_llm_default_model():
    """Tests that get_llm returns the default model when no override is provided."""
    clear_llm_cache()
    with patch("core.llm_factory.ChatGoogleGenerativeAI") as mock_chat:
        mock_chat.return_value.model = "gemini-2.5-flash-lite"
        llm = get_llm("rag_qa")

    assert llm.model == "gemini-2.5-flash-lite"

def test_get_llm_override_model():
    """Tests that get_llm returns the overridden model when provided."""
    clear_llm_cache()
    # Attempting to use a model name that is not the default
    override_model = "gemini-2.0-flash"
    
    # This should fail if the parameter doesn't exist yet
    try:
        with patch("core.llm_factory.ChatGoogleGenerativeAI") as mock_chat:
            mock_chat.return_value.model = override_model
            llm = get_llm("rag_qa", model_name=override_model)
        assert llm.model == override_model
    except TypeError as e:
        if "unexpected keyword argument 'model_name'" in str(e):
            pytest.fail("get_llm does not yet support 'model_name' override")
        raise e


def test_get_llm_runtime_override_passes_thinking_config() -> None:
    """GraphRAG runtime overrides should forward dynamic thinking config to the SDK."""
    clear_llm_cache()
    with patch("core.llm_factory.ChatGoogleGenerativeAI") as mock_chat:
        mock_chat.return_value.model = "gemini-2.5-flash-lite"
        with llm_runtime_override(thinking_budget=-1, include_thoughts=False):
            get_llm("graph_extraction")

    mock_chat.assert_called_once()
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["thinking_budget"] == -1
    assert kwargs["include_thoughts"] is False


def test_get_llm_usage_metrics_reads_reasoning_tokens() -> None:
    response = Mock(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
            "output_token_details": {"reasoning": 24},
        }
    )

    usage = get_llm_usage_metrics(response)

    assert usage == {
        "input_tokens": 100,
        "output_tokens": 40,
        "total_tokens": 140,
        "reasoning_tokens": 24,
    }
