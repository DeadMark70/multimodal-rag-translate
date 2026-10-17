"""V10 chat reuses the pipeline and ordinary chat's stream/citation contract."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from core.auth import get_current_user_id
from core.llm_factory import current_llm_runtime_overrides
from data_base import router as chat_router
from data_base.agentic_v10 import chat
from data_base.rag_pipeline_schemas import RAGResult
from data_base.schemas import AskRequest


@pytest.mark.parametrize("streaming", [False, True])
def test_v10_chat_reuses_pipeline_history_and_original_references(
    monkeypatch, streaming
):
    documents = [
        Document(
            page_content="source one",
            metadata={"doc_id": "a", "file_name": "one.pdf", "page": 2},
        ),
        Document(
            page_content="source five",
            metadata={"doc_id": "b", "file_name": "two.pdf", "page": 8},
        ),
    ]
    original = RAGResult(
        answer="Answer [Ref 1] [Ref 5]",
        source_doc_ids=["a", "b"],
        documents=documents,
        agent_trace={
            "agentic_v10": {
                "context_pack": {"final_reference_ids": ["[Ref 1]", "[Ref 5]"]}
            }
        },
    )

    async def execute(**kwargs):
        assert kwargs["user_id"] == "user-a"
        assert kwargs["authorized_doc_ids"] == ["a", "b"]
        assert kwargs["question"] == "What about its speed?"
        assert "assistant: Model A" in kwargs["decomposition_question"]
        assert "UNTRUSTED CONVERSATION CONTEXT" in kwargs["decomposition_question"]
        assert current_llm_runtime_overrides()["model_name"] == "gemini-3.1-flash-lite"
        assert current_llm_runtime_overrides()["max_output_tokens"] == 8192
        if kwargs["progress_callback"]:
            for stage in (
                "query_expansion",
                "coverage_check",
                "drilldown",
                "answer_generation",
            ):
                await kwargs["progress_callback"](stage, None)
        return original

    pipeline = AsyncMock(side_effect=execute)
    monkeypatch.setattr(chat.AgenticV10PipelineService, "execute", pipeline)
    legacy = AsyncMock(
        side_effect=AssertionError("V10 must not execute the legacy RAG pipeline")
    )
    monkeypatch.setattr(chat_router, "rag_answer_question", legacy)
    monkeypatch.setattr(chat_router, "insert_chat_log", AsyncMock())
    monkeypatch.setattr(chat_router, "insert_query_log", AsyncMock())
    monkeypatch.setattr(
        "data_base.citations.fetch_document_filenames", AsyncMock(return_value={})
    )
    app = FastAPI()
    app.include_router(chat_router.router, prefix="/rag")
    app.dependency_overrides[get_current_user_id] = lambda: "user-a"
    with TestClient(app) as client:
        reply = client.post(
            "/rag/ask/stream" if streaming else "/rag/ask",
            json={
                "rag_mode": "agentic_v10",
                "question": "What about its speed?",
                "doc_ids": ["a", "b"],
                "history": [{"role": "assistant", "content": "Model A"}],
                "enable_evaluation": True,
            },
        )
    assert reply.status_code == 200
    if streaming:
        assert '"stage": "coverage_check"' in reply.text
        assert '"stage": "drilldown"' in reply.text
        events = reply.text.replace("\r\n", "\n").split("\n\n")
        complete = next(
            event for event in events if event.startswith("event: complete")
        )
        payload = json.loads(
            next(
                line[6:] for line in complete.splitlines() if line.startswith("data: ")
            )
        )
    else:
        payload = reply.json()
    assert payload["answer"] == original.answer
    assert payload["agentic_execution_version"] == "v10"
    assert payload["metrics"] is None
    assert [
        (item["reference_id"], item["filename"], item["page"])
        for item in payload["sources"]
    ] == [
        ("[Ref 1]", "one.pdf", 2),
        ("[Ref 5]", "two.pdf", 8),
    ]
    assert "reference_id" not in documents[0].metadata
    legacy.assert_not_awaited()
    pipeline.assert_awaited_once()


@pytest.mark.asyncio
async def test_closing_v10_stream_cancels_the_active_pipeline(monkeypatch):
    cancelled = asyncio.Event()

    async def answer(request, user_id, *, progress_callback):
        try:
            await progress_callback("retrieval", None)
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(chat_router, "answer_agentic_chat", answer)
    response = await chat_router.ask_question_with_context_stream(
        AskRequest(question="question", rag_mode="agentic_v10"),
        BackgroundTasks(),
        "user-a",
    )
    iterator = response.body_iterator
    event = await asyncio.wait_for(anext(iterator), timeout=2)
    assert event["event"] == "phase_update"
    await iterator.aclose()
    assert cancelled.is_set()
