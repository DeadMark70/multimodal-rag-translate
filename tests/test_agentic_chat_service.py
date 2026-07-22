from __future__ import annotations

import json
import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from data_base.RAG_QA_service import RAGResult
from data_base.agentic_chat_service import (
    AgenticChatService,
    StreamingAgenticEvaluationService,
    build_query_resolution_question,
)
from data_base.agentic_v9.token_estimator import estimate_text_tokens
from data_base.indexing_service import DEFAULT_PRODUCTION_INDEXING_PROFILE
from data_base.schemas_deep_research import EditableSubTask
from data_base.schemas_agentic_chat import (
    MAX_AGENTIC_HISTORY_TOKENS,
    AgenticBenchmarkStreamRequest,
    AgenticHistoryMessage,
)
from data_base.sse_events import TaskDoneData
from main import app

TEST_USER_ID = "test-user-123"
AGENTIC_LEGACY_CHAT_PROFILE = (
    f"agentic_eval_v7_semantic_router_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
)


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
        events = [
            event
            async for event in service.execute_stream(
                request=request, user_id=TEST_USER_ID
            )
        ]

    event_names = [event["event"] for event in events]
    assert "plan_ready" in event_names
    assert "plan_confirmed" in event_names
    assert "task_start" in event_names
    assert "task_phase_update" in event_names
    assert "task_done" in event_names
    assert "evaluation_update" in event_names
    assert "trace_step" in event_names
    assert "synthesis_start" in event_names
    assert "complete" in event_names
    assert event_names.index("plan_ready") < event_names.index("plan_confirmed")
    assert event_names.index("plan_confirmed") < event_names.index("task_start")
    assert event_names.index("task_start") < event_names.index("task_done")
    assert event_names.index("task_done") < event_names.index("complete")

    task_done_event = next(event for event in events if event["event"] == "task_done")
    assert json.loads(task_done_event["data"])["answer"] == (
        "LoRA keeps most base weights frozen."
    )

    complete_event = next(event for event in events if event["event"] == "complete")
    payload = json.loads(complete_event["data"])
    assert payload["result"]["summary"] == "Benchmark summary"
    assert payload["agent_trace"]["execution_profile"] == AGENTIC_LEGACY_CHAT_PROFILE

    assert mock_persist.await_count == 1
    persisted = mock_persist.await_args.kwargs
    assert persisted["conversation_id"] == "conv-123"
    assert persisted["user_id"] == TEST_USER_ID
    assert persisted["metadata"]["research_engine"] == "agentic_benchmark"
    assert persisted["metadata"]["execution_profile"] == AGENTIC_LEGACY_CHAT_PROFILE
    assert persisted["metadata"]["result"]["summary"] == "Benchmark summary"
    assert persisted["metadata"]["agent_trace"]["mode"] == "agentic"


def test_task_done_data_allows_v9_evidence_progress_without_an_answer() -> None:
    payload = TaskDoneData(
        id=7,
        question="Which source supports the claim?",
        evidence_count=3,
        target_slots=["method", "result"],
        sources=["doc-7"],
    )

    assert payload.answer is None
    assert payload.evidence_count == 3
    assert payload.target_slots == ["method", "result"]
    assert payload.sources == ["doc-7"]


def test_agentic_stream_request_rejects_history_over_message_limit() -> None:
    with pytest.raises(ValueError, match="at most 10"):
        AgenticBenchmarkStreamRequest(
            question="Follow up",
            history=[{"role": "user", "content": "x"}] * 11,
        )


def test_history_is_bounded_and_used_only_for_query_resolution() -> None:
    history = [
        {"role": "user", "content": "old context " * 900},
        {"role": "assistant", "content": "recent answer " * 900},
    ]

    resolution_question = build_query_resolution_question(
        question="What about its limitation?",
        history=history,
    )

    assert "UNTRUSTED CONVERSATION CONTEXT" in resolution_question
    assert "What about its limitation?" in resolution_question
    assert len(resolution_question) <= (MAX_AGENTIC_HISTORY_TOKENS * 4) + 300


def test_history_resolution_accepts_validated_history_messages() -> None:
    resolution_question = build_query_resolution_question(
        question="What about its limitation?",
        history=[
            AgenticHistoryMessage(role="user", content="Explain LoRA."),
            AgenticHistoryMessage(role="assistant", content="LoRA is a low-rank adapter."),
        ],
    )

    assert "user: Explain LoRA." in resolution_question
    assert "assistant: LoRA is a low-rank adapter." in resolution_question


def test_history_resolution_enforces_the_token_cap_for_cjk_history() -> None:
    resolution_question = build_query_resolution_question(
        question="繼續說明。",
        history=[AgenticHistoryMessage(role="user", content="醫" * 2000)],
    )
    history_block = resolution_question.split("\n\nCURRENT USER QUESTION:", 1)[0]
    history_block = history_block.split("\n", 1)[1]

    assert estimate_text_tokens(history_block) <= MAX_AGENTIC_HISTORY_TOKENS


@pytest.mark.asyncio
async def test_v9_stream_emits_evidence_progress_without_legacy_subtask_answer() -> None:
    service = AgenticChatService()

    async def fake_rag_answer_question(**kwargs):
        return RAGResult(
            answer="Legacy subtask answer must not be emitted by v9.",
            source_doc_ids=["doc-v9"],
            documents=[],
            usage={"total_tokens": 5},
            thought_process=None,
            tool_calls=[],
            visual_verification_meta={},
        )

    mock_report = SimpleNamespace(
        summary="v9 summary",
        detailed_answer="v9 final answer",
        confidence=0.9,
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
        ),
    ):
        events = [
            event
            async for event in service.execute_stream(
                request=AgenticBenchmarkStreamRequest(
                    question="What is LoRA?",
                    agentic_execution_version="v9",
                ),
                user_id=TEST_USER_ID,
            )
        ]

    task_done = next(event for event in events if event["event"] == "task_done")
    task_payload = json.loads(task_done["data"])
    assert task_payload["answer"] is None
    assert task_payload["evidence_count"] == 0
    assert task_payload["target_slots"] == []
    assert task_payload["sources"] == ["doc-v9"]

    complete = next(event for event in events if event["event"] == "complete")
    assert json.loads(complete["data"])["agent_trace"]["agentic_execution_version"] == "v9"


@pytest.mark.asyncio
async def test_stream_disconnect_cancels_the_active_agentic_pipeline() -> None:
    service = AgenticChatService()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def never_finishing_rag(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def passthrough(func, *args, **kwargs):  # noqa: ANN001
        return await func(*args, **kwargs)

    plan = SimpleNamespace(
        estimated_complexity="simple",
        sub_tasks=[
            EditableSubTask(id=1, question="Find the evidence", task_type="rag")
        ],
    )
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
            "data_base.agentic_chat_service.StreamingAgenticEvaluationService.generate_agentic_plan",
            new=AsyncMock(return_value=plan),
        ),
        patch(
            "data_base.agentic_chat_service.rag_answer_question",
            new=AsyncMock(side_effect=never_finishing_rag),
        ),
        patch(
            "data_base.agentic_chat_service.run_with_retry",
            new=AsyncMock(side_effect=passthrough),
        ),
    ):
        stream = service.execute_stream(
            request=AgenticBenchmarkStreamRequest(question="Find the evidence"),
            user_id=TEST_USER_ID,
        )
        assert (await anext(stream))["event"] == "plan_ready"
        event = await anext(stream)
        while event["event"] != "plan_confirmed":
            event = await anext(stream)
        event = await anext(stream)
        while event["event"] != "task_start":
            event = await anext(stream)
        await asyncio.wait_for(started.wait(), timeout=1)
        await stream.aclose()

    await asyncio.wait_for(cancelled.wait(), timeout=1)


@pytest.mark.parametrize(
    ("route_profile", "expected_hyde", "expected_multi_query", "expected_graph"),
    [
        ("hybrid_exact", False, False, False),
        ("hybrid_compare", True, True, False),
        ("graph_global", False, False, True),
        ("visual_verify", False, True, False),
        ("generic_graph", True, True, True),
    ],
)
def test_public_agentic_chat_explicitly_retains_legacy_v7_retrieval_policy(
    route_profile: str,
    expected_hyde: bool,
    expected_multi_query: bool,
    expected_graph: bool,
) -> None:
    service = StreamingAgenticEvaluationService(emit_event=AsyncMock())

    kwargs = service._route_kwargs(
        route_profile=route_profile,
        micro_route="broad_context_rag",
        enable_reranking=True,
        enable_visual_verification=False,
        task_type="graph_analysis" if expected_graph else "rag",
        stage_hint="exploration",
    )

    assert service.execution_profile == AGENTIC_LEGACY_CHAT_PROFILE
    assert kwargs["enable_hyde"] is expected_hyde
    assert kwargs["enable_multi_query"] is expected_multi_query
    assert "crag_rewrite_mode" not in kwargs
    assert kwargs["enable_graph_rag"] is expected_graph
    if expected_graph:
        assert kwargs["graph_execution_hints"] == {
            "stage_hint": "exploration",
            "task_type_hint": "graph_analysis",
            "prefer_global": True,
            "prefer_local": False,
        }


@contextmanager
def _build_client(database_path: Path):
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("evaluation.db.EVALUATION_DB_PATH", database_path),
    ):
        app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
        with TestClient(app) as client:
            yield client
    app.dependency_overrides = {}


def test_agentic_stream_endpoint_returns_sse_response(tmp_path: Path) -> None:
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
    with (
        _build_client(tmp_path / "evaluation.db") as client,
        patch(
            "data_base.router.get_agentic_chat_service",
            return_value=fake_service,
        ),
    ):
        response = client.post("/rag/agentic/stream", json={"question": "test"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "event: complete" in response.text
