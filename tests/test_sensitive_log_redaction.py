"""Behavior regressions for user-controlled production log content."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from conversations.schemas import ConversationCreate
from core.errors import AppError, ErrorCode
from data_base.schemas_deep_research import EditableSubTask

_SECRET_MARKER = "PRIVATE-MARKER-9f72c1"
_RAW_USER_ID = "raw-user-81d4"


class _RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def _isolated_log_messages(logger: logging.Logger):
    """Capture one module logger without propagating sensitive RED fixtures."""
    handler = _RecordHandler()
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def _assert_sensitive_values_absent(messages: list[str]) -> None:
    combined = "\n".join(messages)
    if _SECRET_MARKER in combined or _RAW_USER_ID in combined:
        raise AssertionError("Sensitive user-controlled content reached production logs")


@pytest.mark.asyncio
async def test_rag_router_logs_only_safe_research_metadata() -> None:
    from data_base import router

    planner_service = SimpleNamespace(
        generate_plan=AsyncMock(return_value=SimpleNamespace(sub_tasks=[]))
    )
    with (
        _isolated_log_messages(router.logger) as messages,
        patch.object(router, "get_deep_research_service", return_value=planner_service),
    ):
        await router.generate_research_plan(
            SimpleNamespace(
                question=_SECRET_MARKER,
                doc_ids=[],
                enable_graph_planning=False,
            ),
            user_id=_RAW_USER_ID,
        )

    _assert_sensitive_values_absent(messages)


@pytest.mark.asyncio
async def test_rag_router_does_not_log_database_error_content() -> None:
    from data_base import router

    error = AppError(
        code=ErrorCode.DATABASE_ERROR,
        message=_SECRET_MARKER,
        status_code=503,
    )
    with (
        _isolated_log_messages(router.logger) as messages,
        patch.object(router, "insert_chat_log", new=AsyncMock(side_effect=error)),
        patch.object(router, "insert_query_log", new=AsyncMock()),
    ):
        await router._log_query_to_supabase(
            user_id=_RAW_USER_ID,
            question="question",
            answer="answer",
            has_history=False,
        )

    _assert_sensitive_values_absent(messages)


@pytest.mark.asyncio
async def test_deep_research_service_logs_only_safe_plan_metadata() -> None:
    from data_base import deep_research_service

    plan = SimpleNamespace(sub_tasks=[], estimated_complexity="simple")
    planner = SimpleNamespace(plan=AsyncMock(return_value=plan))
    with (
        _isolated_log_messages(deep_research_service.logger) as messages,
        patch.object(deep_research_service, "TaskPlanner", return_value=planner),
    ):
        await deep_research_service.DeepResearchService().generate_plan(
            question=_SECRET_MARKER,
            user_id=_RAW_USER_ID,
        )

    _assert_sensitive_values_absent(messages)


@pytest.mark.asyncio
async def test_deep_research_service_does_not_log_task_error_content() -> None:
    from data_base import deep_research_service

    service = deep_research_service.DeepResearchService()
    task = EditableSubTask(id=1, question="question")
    with (
        _isolated_log_messages(deep_research_service.logger) as messages,
        patch.object(
            deep_research_service,
            "rag_answer_question",
            new=AsyncMock(side_effect=RuntimeError(_SECRET_MARKER)),
        ),
    ):
        await service._execute_single_task(
            task=task,
            user_id=_RAW_USER_ID,
            doc_ids=None,
            enable_reranking=False,
            iteration=0,
        )

    _assert_sensitive_values_absent(messages)


@pytest.mark.asyncio
async def test_research_execution_core_logs_only_safe_plan_metadata() -> None:
    from data_base import research_execution_core

    plan = SimpleNamespace(sub_tasks=[], estimated_complexity="simple")
    planner = SimpleNamespace(plan=AsyncMock(return_value=plan))
    with (
        _isolated_log_messages(research_execution_core.logger) as messages,
        patch.object(research_execution_core, "TaskPlanner", return_value=planner),
    ):
        await research_execution_core.ResearchExecutionCore().generate_plan(
            question=_SECRET_MARKER,
            user_id=_RAW_USER_ID,
        )

    _assert_sensitive_values_absent(messages)


@pytest.mark.asyncio
async def test_research_execution_core_does_not_log_task_error_content() -> None:
    from data_base import research_execution_core

    task = EditableSubTask(id=1, question="question")
    with (
        _isolated_log_messages(research_execution_core.logger) as messages,
        patch.object(
            research_execution_core,
            "rag_answer_question",
            new=AsyncMock(side_effect=RuntimeError(_SECRET_MARKER)),
        ),
    ):
        await research_execution_core.ResearchExecutionCore()._execute_tasks(
            tasks=[task],
            user_id=_RAW_USER_ID,
            doc_ids=None,
            enable_reranking=False,
            iteration=0,
        )

    _assert_sensitive_values_absent(messages)


@pytest.mark.asyncio
async def test_crag_logs_only_safe_correction_metadata() -> None:
    from data_base import rag_crag

    with _isolated_log_messages(rag_crag.logger) as messages:
        await rag_crag.run_corrective_retrieval(
            question=_SECRET_MARKER,
            documents=[Document(page_content="evidence")],
            retriever=object(),
            judge=AsyncMock(return_value=False),
            rewrite_mode="none",
            query_executor=AsyncMock(return_value=[[]]),
        )

    _assert_sensitive_values_absent(messages)


@pytest.mark.asyncio
async def test_conversation_router_logs_only_safe_creation_metadata() -> None:
    router = import_module("conversations.router")

    with (
        _isolated_log_messages(router.logger) as messages,
        patch.object(
            router,
            "create_user_conversation",
            new=AsyncMock(return_value=object()),
        ),
    ):
        await router.create_conversation(
            ConversationCreate(title=_SECRET_MARKER),
            user_id=_RAW_USER_ID,
        )

    _assert_sensitive_values_absent(messages)
