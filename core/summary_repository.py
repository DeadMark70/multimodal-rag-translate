"""Repository helpers for document summary persistence."""

from __future__ import annotations

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


async def update_document_summary_record(*, doc_id: str, summary: str) -> None:
    """Updates executive summary by document id."""
    client = _get_client_or_raise()
    try:
        await run_in_threadpool(
            lambda: client.table("documents")
            .update({"executive_summary": summary})
            .eq("id", doc_id)
            .execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to update executive summary",
            status_code=500,
            details={"operation": "update_document_summary_record"},
        ) from exc


async def get_document_summary_record(*, doc_id: str) -> str | None:
    """Returns executive summary by document id."""
    client = _get_client_or_raise()
    try:
        result = await run_in_threadpool(
            lambda: client.table("documents")
            .select("executive_summary")
            .eq("id", doc_id)
            .single()
            .execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to get executive summary",
            status_code=500,
            details={"operation": "get_document_summary_record"},
        ) from exc
    return result.data.get("executive_summary") if result.data else None
