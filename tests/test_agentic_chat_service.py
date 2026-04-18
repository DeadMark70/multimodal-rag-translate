from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from data_base.RAG_QA_service import RAGResult
from data_base.agentic_chat_service import AgenticChatService
from data_base.schemas_agentic_chat import AgenticBenchmarkStreamRequest
from evaluation.agentic_evaluation_service import AGENTIC_EVAL_PROFILE
from main import app

TEST_USER_ID = "test-user-123"


@pytest.mark.asyncio
async def test_execute_stream_emits_agentic_events_and_persists_trace() -> None:
    service = AgenticChatService()

    async def fake_rag_answer_question(**kwargs):
        assert kwargs["plain_mode"] is False
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            await progress_callback("retrieval", {"query_count": 1})
            await progress_callback("answer_generation", {"chunk_count": 2})
        return RAGResult(
            answer="LoRA keeps most base weights frozen.",
            source_doc_ids=["doc-1"],
            documents=[],
            usage={"total_tokens": 42},
            thought_process="retrieval -> synthesis",
            tool_calls=[{"action": "retrieve_docs", "status": "completed"}],
            visual_verification_meta={},
        )

    mock_report = SimpleNamespace(
        summary="Benchmark summary",
        detailed_answer="Benchmark detailed answer",
        confidence=0.91,
    )

    async def passthrough(func, *args, **kwargs):  # noqa: ANN001
        return await func(*args, **kwargs)

    with (
        patch(
            "data_base.agentic_chat_service.classify_question_intent",
            return_value="enumeration_definition",
        ),
        patch(
            "data_base.agentic_chat_service._strategy_tier_for_intent",
            return_value="tier_1_detail_lookup",
        ),
        patch(
            "data_base.agentic_chat_service._drilldown_iterations_for_strategy",
            return_value=0,
        ),
        patch(
            "data_base.agentic_chat_service.rag_answer_question",
            new=AsyncMock(side_effect=fake_rag_answer_question),
        ),
        patch(
            "evaluation.agentic_evaluation_service.synthesize_results",
            new=AsyncMock(return_value=mock_report),
        ),
        patch(
            "data_base.agentic_chat_service.run_with_retry",
            new=AsyncMock(side_effect=passthrough),
        ),
        patch(
            "data_base.agentic_chat_service.persist_research_conversation",
            new=AsyncMock(),
        ) as mock_persist,
    ):
        request = AgenticBenchmarkStreamRequest(
            question="What is LoRA?",
            conversation_id="conv-123",
        )
        events = [event async for event in service.execute_stream(request=request, user_id=TEST_USER_ID)]

    event_names = [event["event"] for event in events]
    assert "plan_ready" in event_names
    assert "task_start" in event_names
    assert "task_phase_update" in event_names
    assert "task_done" in event_names
    assert "evaluation_update" in event_names
    assert "trace_step" in event_names
    assert "synthesis_start" in event_names
    assert "complete" in event_names
    assert event_names.index("plan_ready") < event_names.index("task_start")
    assert event_names.index("task_start") < event_names.index("task_done")
    assert event_names.index("task_done") < event_names.index("complete")

    complete_event = next(event for event in events if event["event"] == "complete")
    payload = json.loads(complete_event["data"])
    assert payload["result"]["summary"] == "Benchmark summary"
    assert payload["agent_trace"]["execution_profile"] == AGENTIC_EVAL_PROFILE

    assert mock_persist.await_count == 1
    persisted = mock_persist.await_args.kwargs
    assert persisted["conversation_id"] == "conv-123"
    assert persisted["user_id"] == TEST_USER_ID
    assert persisted["metadata"]["research_engine"] == "agentic_benchmark"
    assert persisted["metadata"]["result"]["summary"] == "Benchmark summary"
    assert persisted["metadata"]["agent_trace"]["mode"] == "agentic"


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


def test_agentic_stream_endpoint_returns_sse_response() -> None:
    async def fake_stream(*, request, user_id):  # noqa: ANN001
        assert user_id == TEST_USER_ID
        assert request.question == "test"
        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "result": {
                        "question": "test",
                        "summary": "done",
                        "detailed_answer": "done",
                        "sub_tasks": [],
                        "all_sources": [],
                        "confidence": 1.0,
                        "total_iterations": 0,
                    },
                    "agent_trace": {"mode": "agentic"},
                }
            ),
        }

    fake_service = SimpleNamespace(execute_stream=fake_stream)
    with _build_client() as client, patch(
        "data_base.router.get_agentic_chat_service",
        return_value=fake_service,
    ):
        response = client.post("/rag/agentic/stream", json={"question": "test"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "event: complete" in response.text
