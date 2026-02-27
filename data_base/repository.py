"""Persistence helpers for RAG domain Supabase access."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.concurrency import run_in_threadpool
from postgrest.exceptions import APIError as PostgrestAPIError

from core.errors import AppError, ErrorCode
from supabase_client import get_supabase


def _get_client_or_raise():
    client = get_supabase()
    if not client:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Database service unavailable",
            status_code=500,
        )
    return client


async def insert_chat_log(
    *,
    user_id: str,
    question: str,
    answer: str,
) -> None:
    """Inserts one row into chat_logs."""
    client = _get_client_or_raise()
    payload = {"user_id": user_id, "question": question, "answer": answer}
    try:
        await run_in_threadpool(lambda: client.table("chat_logs").insert(payload).execute())
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to save chat log",
            status_code=500,
            details={"operation": "insert_chat_log"},
        ) from exc


async def insert_query_log(
    *,
    user_id: str,
    question: str,
    answer: str | None,
    has_history: bool,
    faithfulness: str | None,
    confidence: float | None,
    response_time_ms: int | None,
    doc_ids: list[str] | None,
) -> None:
    """Inserts one row into query_logs."""
    client = _get_client_or_raise()
    payload = {
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "has_history": has_history,
        "faithfulness": faithfulness,
        "confidence": confidence,
        "response_time_ms": response_time_ms,
        "doc_ids": doc_ids,
    }
    try:
        await run_in_threadpool(lambda: client.table("query_logs").insert(payload).execute())
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to save query log",
            status_code=500,
            details={"operation": "insert_query_log"},
        ) from exc


async def fetch_document_filenames(doc_ids: list[str]) -> dict[str, str]:
    """Returns a map of document id to file_name."""
    if not doc_ids:
        return {}

    client = _get_client_or_raise()
    try:
        response = await run_in_threadpool(
            lambda: client.table("documents")
            .select("id, file_name")
            .in_("id", doc_ids)
            .execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to fetch document metadata",
            status_code=500,
            details={"operation": "fetch_document_filenames"},
        ) from exc

    name_map: dict[str, str] = {}
    for row in response.data or []:
        doc_id = row.get("id")
        if not doc_id:
            continue
        name_map[doc_id] = row.get("file_name") or f"文件-{doc_id[:8]}"
    return name_map


async def persist_research_conversation(
    *,
    conversation_id: str,
    user_id: str,
    title: str | None,
    metadata: dict,
) -> None:
    """Persists deep-research result payload to conversations metadata."""
    client = _get_client_or_raise()
    payload = {
        "metadata": metadata,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if title:
        payload["title"] = title

    try:
        await run_in_threadpool(
            lambda: client.table("conversations")
            .update(payload)
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to persist research conversation",
            status_code=500,
            details={"operation": "persist_research_conversation"},
        ) from exc
