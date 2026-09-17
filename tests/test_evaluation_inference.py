import asyncio
from unittest.mock import AsyncMock

import pytest
from google.genai import types
from google.genai.errors import ServerError
from langchain_core.messages import HumanMessage

from core.evaluation_inference import EvaluationGoogleChat, evaluation_inference_scope
from evaluation.retry import RateBudget
from evaluation.token_normalizers import normalize_provider_usage


def response():
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="ok")]),
                finish_reason="STOP",
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=10,
            thoughts_token_count=5,
            cached_content_token_count=80,
            total_token_count=115,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_count, expected", [(None, 0), (0, 0), (80, 80)])
async def test_measured_gemini_response_keeps_cache_misses(monkeypatch, cache_count, expected):
    model = EvaluationGoogleChat(model="gemini-3.1-flash-lite", google_api_key="unit-test")
    reply = response()
    reply.usage_metadata.cached_content_token_count = cache_count
    monkeypatch.setattr(model.async_client.models, "generate_content", AsyncMock(return_value=reply))
    with evaluation_inference_scope({"ragas_service_tier": "flex"}, limiter=RateBudget(100)):
        result = await model._agenerate([HumanMessage(content="question")])
    usage = normalize_provider_usage("google", result.generations[0].message.usage_metadata)
    assert usage.cached_input_tokens == expected
    assert usage.reconciliation_status == "balanced"


@pytest.mark.asyncio
async def test_missing_response_usage_stays_unknown(monkeypatch):
    model = EvaluationGoogleChat(model="gemini-3.1-flash-lite", google_api_key="unit-test")
    reply = response()
    reply.usage_metadata = None
    monkeypatch.setattr(model.async_client.models, "generate_content", AsyncMock(return_value=reply))
    with evaluation_inference_scope({}, limiter=RateBudget(100)):
        result = await model._agenerate([HumanMessage(content="question")])
    usage = normalize_provider_usage("google", result.generations[0].message.usage_metadata)
    assert usage.usage_status == "missing"
    assert usage.cached_input_tokens is None


@pytest.mark.asyncio
async def test_flex_reaches_sdk_with_correct_timeout_and_single_sdk_attempt(
    monkeypatch,
):
    model = EvaluationGoogleChat(
        model="gemini-3.5-flash-lite", google_api_key="unit-test"
    )
    call = AsyncMock(return_value=response())
    monkeypatch.setattr(model.async_client.models, "generate_content", call)
    budget = RateBudget(100)
    with evaluation_inference_scope(
        {"ragas_service_tier": "flex", "ragas_request_timeout_seconds": 900},
        limiter=budget,
    ):
        result = await model._agenerate([HumanMessage(content="question")])
    config = call.call_args.kwargs["config"]
    assert config.service_tier == types.ServiceTier.FLEX
    assert config.http_options.timeout == 900000
    assert config.http_options.retry_options.attempts == 1
    assert len(budget._timestamps) == 1
    assert (
        result.generations[0].message.response_metadata["requested_service_tier"]
        == "flex"
    )


@pytest.mark.asyncio
async def test_flex_fallback_is_explicit_and_each_request_is_counted(monkeypatch):
    model = EvaluationGoogleChat(
        model="gemini-3.5-flash-lite", google_api_key="unit-test"
    )
    call = AsyncMock(
        side_effect=[ServerError(503, {"error": {"message": "capacity"}}), response()]
    )
    monkeypatch.setattr(model.async_client.models, "generate_content", call)
    budget = RateBudget(100)
    with evaluation_inference_scope(
        {
            "ragas_service_tier": "flex",
            "ragas_max_attempts": 1,
            "ragas_standard_fallback": True,
        },
        limiter=budget,
    ):
        await model._agenerate([HumanMessage(content="question")])
    assert [
        str(c.kwargs["config"].service_tier.value) for c in call.call_args_list
    ] == ["flex", "standard"]
    assert len(budget._timestamps) == 2


@pytest.mark.asyncio
async def test_flex_does_not_fallback_without_opt_in(monkeypatch):
    model = EvaluationGoogleChat(
        model="gemini-3.5-flash-lite", google_api_key="unit-test"
    )
    call = AsyncMock(side_effect=ServerError(503, {"error": {"message": "capacity"}}))
    monkeypatch.setattr(model.async_client.models, "generate_content", call)
    with evaluation_inference_scope(
        {"ragas_service_tier": "flex", "ragas_max_attempts": 1}, limiter=RateBudget(100)
    ):
        with pytest.raises(Exception):
            await model._agenerate([HumanMessage(content="question")])
    assert call.await_count == 1


@pytest.mark.asyncio
async def test_request_slots_cover_parallel_relevancy_samples(monkeypatch):
    model = EvaluationGoogleChat(
        model="gemini-3.5-flash-lite", google_api_key="unit-test"
    )
    active = peak = 0

    async def generate(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return response()

    monkeypatch.setattr(model.async_client.models, "generate_content", generate)
    with evaluation_inference_scope(
        {}, limiter=RateBudget(100), slots=asyncio.Semaphore(2)
    ):
        await asyncio.gather(
            *(model._agenerate([HumanMessage(content="q")]) for _ in range(6))
        )
    assert peak == 2
