"""Unit tests for LLM factory model and runtime overrides."""

from unittest.mock import Mock, patch

import pytest

from core.llm_factory import (
    clear_llm_cache,
    get_graph_rag_runtime_overrides,
    get_llm,
    get_llm_usage_metrics,
    graph_rag_llm_runtime_override,
)

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


def test_get_llm_graph_rag_defaults_use_2_5_flash_lite() -> None:
    clear_llm_cache()
    with patch("core.llm_factory.ChatGoogleGenerativeAI") as mock_chat:
        mock_chat.return_value.model = "gemini-2.5-flash-lite"
        get_llm("graph_extraction")
        get_llm("community_summary")

    assert mock_chat.call_args_list[0].kwargs["model"] == "gemini-2.5-flash-lite"
    assert mock_chat.call_args_list[1].kwargs["model"] == "gemini-2.5-flash-lite"


def test_get_graph_rag_runtime_overrides_use_budget_for_2_5_models() -> None:
    assert get_graph_rag_runtime_overrides("graph_extraction") == {
        "thinking_budget": 2048,
        "include_thoughts": False,
    }
    assert get_graph_rag_runtime_overrides("community_summary") == {
        "thinking_budget": 1024,
        "include_thoughts": False,
    }


def test_get_graph_rag_runtime_overrides_use_thinking_level_for_gemini_3() -> None:
    assert get_graph_rag_runtime_overrides(
        "community_summary",
        model_name="gemini-3.1-flash-lite-preview",
    ) == {
        "thinking_level": "high",
        "include_thoughts": False,
    }


def test_graph_rag_runtime_override_passes_2_5_thinking_config() -> None:
    """GraphRAG runtime overrides should forward 2.5 thinking config to the SDK."""
    clear_llm_cache()
    with patch("core.llm_factory.ChatGoogleGenerativeAI") as mock_chat:
        mock_chat.return_value.model = "gemini-2.5-flash-lite"
        with graph_rag_llm_runtime_override("graph_extraction"):
            get_llm("graph_extraction")

    mock_chat.assert_called_once()
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["thinking_budget"] == 2048
    assert kwargs["include_thoughts"] is False
    assert "thinking_level" not in kwargs


def test_graph_rag_runtime_override_passes_summary_budget_for_2_5() -> None:
    clear_llm_cache()
    with patch("core.llm_factory.ChatGoogleGenerativeAI") as mock_chat:
        mock_chat.return_value.model = "gemini-2.5-flash-lite"
        with graph_rag_llm_runtime_override("community_summary"):
            get_llm("community_summary")

    mock_chat.assert_called_once()
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["thinking_budget"] == 1024
    assert kwargs["include_thoughts"] is False


def test_graph_rag_runtime_override_passes_3x_thinking_level() -> None:
    clear_llm_cache()
    with patch("core.llm_factory.ChatGoogleGenerativeAI") as mock_chat:
        mock_chat.return_value.model = "gemini-3.1-flash-lite-preview"
        with graph_rag_llm_runtime_override(
            "community_summary",
            model_name="gemini-3.1-flash-lite-preview",
        ):
            get_llm("community_summary", model_name="gemini-3.1-flash-lite-preview")

    mock_chat.assert_called_once()
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["thinking_level"] == "high"
    assert kwargs["include_thoughts"] is False
    assert "thinking_budget" not in kwargs


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
