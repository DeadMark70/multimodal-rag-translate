from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agents.planner import classify_question_intent_semantic


@pytest.mark.asyncio
async def test_semantic_classifier_returns_llm_decision_on_valid_json() -> None:
    mock_llm = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=SimpleNamespace(
                content='{"intent":"benchmark_data","complexity_score":4,"confidence":0.82,"rationale":"numeric benchmark comparison"}'
            )
        )
    )
    with patch("agents.planner.get_llm", return_value=mock_llm):
        decision = await classify_question_intent_semantic("Compare FLOPs and Params across models")

    assert decision.intent == "benchmark_data"
    assert decision.complexity_score == 4
    assert decision.source == "llm"


@pytest.mark.asyncio
async def test_semantic_classifier_timeout_falls_back_to_heuristic() -> None:
    mock_llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=TimeoutError()))
    with patch("agents.planner.get_llm", return_value=mock_llm):
        decision = await classify_question_intent_semantic("What is SONO-MultiKAN?")

    assert decision.source == "timeout_fallback"
    assert decision.intent in {
        "comparison_disambiguation",
        "figure_flow",
        "benchmark_data",
        "enumeration_definition",
        "general_research",
    }
    assert 1 <= decision.complexity_score <= 5


@pytest.mark.asyncio
async def test_semantic_classifier_parse_error_falls_back_to_heuristic() -> None:
    mock_llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="not json")))
    with patch("agents.planner.get_llm", return_value=mock_llm):
        decision = await classify_question_intent_semantic("List the model components")

    assert decision.source == "parse_fallback"
    assert 1 <= decision.complexity_score <= 5
