"""Exercise real SDK serialization/parsing against in-memory HTTP responses."""

import json

import httpx
import pytest
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage

from core.evaluation_inference import EvaluationGoogleChat, evaluation_inference_scope
from core.llm_response import final_answer_text
from core.llm_factory import (
    chat_llm_settings,
    clear_llm_cache,
    get_llm,
    graph_rag_llm_runtime_override,
    llm_runtime_override,
)
from evaluation.retry import RateBudget
from evaluation.token_normalizers import normalize_provider_usage


def _reply():
    return {
        "modelVersion": "gemini-3.1-flash-lite",
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": '{"score": 1}'}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 5,
            "cachedContentTokenCount": 80,
            "totalTokenCount": 115,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", ["minimal", "medium"])
@pytest.mark.parametrize("tier", ["standard", "flex"])
async def test_scoring_serializes_controls_and_parses_usage(thinking, tier):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=_reply())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        sdk = genai.Client(
            api_key="unit-test",
            http_options=types.HttpOptions(httpx_async_client=http),
        )
        model = EvaluationGoogleChat(
            model="gemini-3.1-flash-lite",
            google_api_key="unit-test",
            thinking_level=thinking,
        )
        model.client.close()
        model.client = sdk
        try:
            with evaluation_inference_scope(
                {"ragas_service_tier": tier, "ragas_request_timeout_seconds": 900},
                limiter=RateBudget(100),
            ):
                result = await model.ainvoke(
                    [HumanMessage(content="score this answer")],
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {"score": {"type": "number"}},
                    },
                )
            assert len(requests) == 1
            assert requests[0].url.path.endswith(":generateContent")
            body = json.loads(requests[0].content)
            config = body["generationConfig"]
            thinking_config = types.ThinkingConfig.model_validate(
                config["thinkingConfig"]
            )
            assert thinking_config.thinking_level.value.lower() == thinking
            assert config["responseMimeType"] == "application/json"
            assert body["serviceTier"].lower() == tier
            assert requests[0].extensions["timeout"]["read"] == 900
            assert json.loads(final_answer_text(result)) == {"score": 1}
            usage = normalize_provider_usage("google", result.usage_metadata)
            assert usage.cached_input_tokens == 80
            assert usage.reconciliation_status == "balanced"
        finally:
            await sdk.aio.aclose()
            sdk.close()


@pytest.mark.asyncio
async def test_ordinary_chat_stream_parses_sdk_sse():
    def handle(request):
        assert request.url.path.endswith(":streamGenerateContent")
        return httpx.Response(
            200,
            text="data: " + json.dumps(_reply()) + "\n\n",
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        sdk = genai.Client(
            api_key="unit-test",
            http_options=types.HttpOptions(httpx_async_client=http),
        )
        model = EvaluationGoogleChat(
            model="gemini-3.1-flash-lite", google_api_key="unit-test"
        )
        model.client.close()
        model.client = sdk
        try:
            chunks = [chunk async for chunk in model.astream("hello")]
            assert (
                "".join(final_answer_text(chunk) for chunk in chunks) == '{"score": 1}'
            )
        finally:
            await sdk.aio.aclose()
            sdk.close()


def test_control_plane_sdk_model_listing():
    def handle(request):
        assert request.url.path.endswith("/models")
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-3.1-flash-lite",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        with genai.Client(
            api_key="unit-test",
            http_options=types.HttpOptions(httpx_client=http),
        ) as sdk:
            models = list(sdk.models.list())
            assert models[0].name == "models/gemini-3.1-flash-lite"
            assert models[0].supported_actions == ["generateContent"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "purpose",
    ["rag_qa", "query_rewrite", "planner", "synthesizer", "community_summary"],
)
async def test_chat_requests_use_medium_without_deprecated_parameters(
    monkeypatch, purpose
):
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=_reply())

    clear_llm_cache()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        sdk = genai.Client(
            api_key="unit-test", http_options=types.HttpOptions(httpx_async_client=http)
        )
        try:
            async for _ in chat_llm_settings():
                # Old nested phase defaults must not reintroduce deprecated fields.
                with llm_runtime_override(
                    temperature=0.1, top_p=0.8, top_k=10, thinking_budget=1024
                ):
                    with graph_rag_llm_runtime_override("community_summary"):
                        model = get_llm(purpose, model_name="gemini-2.5-flash-lite")
                        model.client.close()
                        model.client = sdk
                        result = await model.ainvoke("question")
            assert isinstance(result.content, list)
            assert final_answer_text(result) == '{"score": 1}'
            assert len(requests) == 1
            assert "/gemini-3.1-flash-lite:" in requests[0].url.path
            config = json.loads(requests[0].content)["generationConfig"]
            assert not {"temperature", "topP", "topK", "top_p", "top_k"}.intersection(
                config
            )
            thinking = config["thinkingConfig"]
            assert (
                "thinkingBudget" not in thinking and "thinking_budget" not in thinking
            )
            assert (
                types.ThinkingConfig.model_validate(thinking).thinking_level
                == types.ThinkingLevel.MEDIUM
            )
        finally:
            clear_llm_cache()
            await sdk.aio.aclose()
            sdk.close()
