"""Chat settings cover HTTP streams without changing evaluation requests."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from core.llm_factory import (
    chat_llm_settings,
    clear_llm_cache,
    current_llm_runtime_overrides,
    get_llm,
    llm_runtime_override,
)
from data_base import router as rag_router
from data_base.sse_events import SSEEventType, format_sse_event


def _assert_chat_settings():
    settings = current_llm_runtime_overrides()
    assert settings["model_name"] == "gemini-3.1-flash-lite"
    assert settings["thinking_level"] == "medium"
    assert settings["chat_request"] is True


@pytest.mark.parametrize("path", ["/ask", "/ask/stream", "/agentic/stream"])
def test_http_chat_scope_reaches_async_answer_and_stream(monkeypatch, path):
    app = FastAPI()
    app.include_router(rag_router.router, prefix="/rag")
    app.dependency_overrides[get_current_user_id] = lambda: "unit-test-user"

    async def answer(**kwargs):
        await asyncio.sleep(0)
        _assert_chat_settings()
        return "answer", []

    async def agentic_stream(**kwargs):
        await asyncio.sleep(0)
        _assert_chat_settings()
        yield format_sse_event(SSEEventType.COMPLETE, {"answer": "answer"})

    monkeypatch.setattr(rag_router, "rag_answer_question", answer)
    monkeypatch.setattr(rag_router, "build_source_details", AsyncMock(return_value=[]))
    monkeypatch.setattr(rag_router, "insert_chat_log", AsyncMock())
    monkeypatch.setattr(rag_router, "insert_query_log", AsyncMock())
    monkeypatch.setattr(
        rag_router,
        "get_agentic_chat_service",
        lambda: Mock(execute_stream=agentic_stream),
    )
    with TestClient(app) as client:
        response = client.post("/rag" + path, json={"question": "question"})
    assert response.status_code == 200
    assert "answer" in response.text
    assert current_llm_runtime_overrides() == {}


@pytest.mark.asyncio
async def test_concurrent_evaluation_keeps_its_model_sampling_and_thinking(monkeypatch):
    created = Mock()
    monkeypatch.setattr("core.llm_factory.ChatGoogleGenerativeAI", created)
    clear_llm_cache()
    chat_started = asyncio.Event()
    evaluation_finished = asyncio.Event()

    async def chat():
        async for _ in chat_llm_settings():
            chat_started.set()
            await evaluation_finished.wait()
            get_llm("rag_qa")
            settings = created.call_args.kwargs
            assert settings["model"] == "gemini-3.1-flash-lite"
            assert settings["temperature"] is None
            assert settings["thinking_level"] == "medium"

    async def evaluation():
        await chat_started.wait()
        with llm_runtime_override(
            model_name="gemini-2.5-flash-lite", temperature=0.2, thinking_budget=8192
        ):
            get_llm("rag_qa")
            settings = created.call_args.kwargs
            assert settings["model"] == "gemini-2.5-flash-lite"
            assert settings["temperature"] == 0.2
            assert settings["thinking_budget"] == 8192
            assert "thinking_level" not in settings
        evaluation_finished.set()

    try:
        await asyncio.wait_for(asyncio.gather(chat(), evaluation()), timeout=5)
        assert current_llm_runtime_overrides() == {}
    finally:
        clear_llm_cache()
