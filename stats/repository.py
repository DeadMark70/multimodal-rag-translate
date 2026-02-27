"""Persistence layer for dashboard statistics."""

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


async def list_query_logs(*, user_id: str) -> list[dict]:
    """Returns all query log rows for the user."""
    client = _get_client_or_raise()
    try:
        response = await run_in_threadpool(
            lambda: client.table("query_logs").select("*").eq("user_id", user_id).execute()
        )
        return response.data or []
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to retrieve statistics",
            status_code=500,
            details={"operation": "list_query_logs"},
        ) from exc
