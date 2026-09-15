"""Exercise the installed Gemini SDK instead of mocking its parameter validation."""

import pytest
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from core.llm_factory import clear_llm_cache, get_llm, llm_runtime_override
from evaluation.model_capabilities import normalize_model_config_for_runtime


@pytest.fixture
def local_google_credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "local-test-no-network")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    clear_llm_cache()
    yield
    clear_llm_cache()


@pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
def test_evaluation_thinking_level_reaches_google_request(
    local_google_credentials, level
):
    setup = {
        "model_name": "gemini-3.5-flash-lite",
        "thinking_mode": True,
        "thinking_level": level,
        "thinking_include_thoughts": False,
        "temperature": 1.0,
        "max_output_tokens": 65536,
    }
    with llm_runtime_override(**normalize_model_config_for_runtime(setup)):
        llm = get_llm("synthesizer")
        assert isinstance(llm, ChatGoogleGenerativeAI)
        # Build the real provider request without sending it or using API credits.
        request = llm._prepare_request([HumanMessage(content="Reply OK")])

    config = request["config"].model_dump(mode="json", exclude_none=True)
    assert request["model"] == setup["model_name"]
    assert config["thinking_config"] == {
        "thinking_level": level.upper(),
        "include_thoughts": False,
    }
    assert config["temperature"] == 1.0
    assert config["max_output_tokens"] == 65536


def test_gemini_25_explicit_zero_budget_still_reaches_google_request(
    local_google_credentials,
):
    setup = {
        "model_name": "gemini-2.5-flash-lite",
        "thinking_mode": True,
        "thinking_budget": 0,
    }
    with llm_runtime_override(**normalize_model_config_for_runtime(setup)):
        request = get_llm("rag_qa")._prepare_request(
            [HumanMessage(content="Reply OK")]
        )

    thinking = request["config"].thinking_config
    assert thinking.thinking_budget == 0
    assert thinking.thinking_level is None
