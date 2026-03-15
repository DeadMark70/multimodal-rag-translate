"""Regression tests for ordinary ask SSE streaming."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from data_base.sse_events import PhaseUpdateData, SSEEventType, format_sse_event
from main import app

TEST_USER_ID = "test-user-123"


@contextmanager
def _build_client():
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
    ):
        app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
        with TestClient(app) as client:
            yield client
    app.dependency_overrides = {}


def test_format_sse_event_serializes_phase_update_payload() -> None:
    event = format_sse_event(
        SSEEventType.PHASE_UPDATE,
        PhaseUpdateData(stage="retrieval", label="正在檢索文件"),
    )

    assert event["event"] == "phase_update"
    payload = json.loads(event["data"])
    assert payload == {"stage": "retrieval", "label": "正在檢索文件"}


def test_rag_ask_stream_emits_phase_updates_and_complete_payload() -> None:
    async def fake_rag_answer_question(**kwargs):
        progress_callback = kwargs["progress_callback"]
        await progress_callback("query_expansion", {"mode": "multi_query"})
        await progress_callback("retrieval", {"query_count": 2})
        await progress_callback("reranking", {"document_count": 5})
        await progress_callback("graph_context", {"search_mode": "generic"})
        await progress_callback("answer_generation", {"document_count": 3})
        return ("streamed answer", ["doc-1"])

    with _build_client() as client, patch(
        "data_base.router.rag_answer_question",
        new=AsyncMock(side_effect=fake_rag_answer_question),
    ), patch("data_base.router.insert_chat_log", new=AsyncMock()), patch(
        "data_base.router.insert_query_log", new=AsyncMock()
    ):
        with client.stream(
            "POST",
            "/rag/ask/stream",
            json={
                "question": "Explain the paper",
                "enable_multi_query": True,
                "enable_reranking": True,
                "enable_graph_rag": True,
                "graph_search_mode": "generic",
            },
        ) as stream_response:
            stream_body = "".join(stream_response.iter_text())

    assert "event: phase_update" in stream_body
    assert '"stage": "query_expansion"' in stream_body
    assert '"stage": "retrieval"' in stream_body
    assert '"stage": "reranking"' in stream_body
    assert '"stage": "graph_context"' in stream_body
    assert '"stage": "answer_generation"' in stream_body
    assert "event: complete" in stream_body
    assert '"answer": "streamed answer"' in stream_body
    assert '"doc_id": "doc-1"' in stream_body


def test_rag_ask_stream_emits_error_event_on_failure() -> None:
    with _build_client() as client, patch(
        "data_base.router.rag_answer_question",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        with client.stream(
            "POST",
            "/rag/ask/stream",
            json={"question": "Explain the paper"},
            ) as stream_response:
            stream_body = "".join(stream_response.iter_text())

    assert "event: error" in stream_body
    assert "Failed to answer question with context" in stream_body
