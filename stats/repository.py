"""Persistence layer for dashboard statistics."""

from __future__ import annotations

from core.supabase_repository import execute_supabase_operation


async def list_query_logs(*, user_id: str) -> list[dict]:
    """Returns all query log rows for the user."""
    response = await execute_supabase_operation(
        operation="list_query_logs",
        failure_message="Failed to retrieve statistics",
        handler=lambda client: client.table("query_logs")
        .select("*")
        .eq("user_id", user_id)
        .execute(),
    )
    return response.data or []
