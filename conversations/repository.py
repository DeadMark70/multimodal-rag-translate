"""Persistence layer for conversation-related Supabase operations."""

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


async def list_conversations(*, user_id: str) -> list[dict]:
    """Returns all conversations for a user sorted by updated time desc."""
    client = _get_client_or_raise()
    try:
        response = await run_in_threadpool(
            lambda: client.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return response.data or []
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to list conversations",
            status_code=500,
            details={"operation": "list_conversations"},
        ) from exc


async def create_conversation(
    *,
    user_id: str,
    title: str | None,
    conversation_type: str,
    metadata: dict | None,
) -> dict | None:
    """Creates a conversation and returns the created row."""
    client = _get_client_or_raise()
    payload = {
        "user_id": user_id,
        "title": title,
        "type": conversation_type,
        "metadata": metadata,
    }
    try:
        response = await run_in_threadpool(
            lambda: client.table("conversations").insert(payload).execute()
        )
        return response.data[0] if response.data else None
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to create conversation",
            status_code=500,
            details={"operation": "create_conversation"},
        ) from exc


async def get_conversation(*, conversation_id: str, user_id: str) -> dict | None:
    """Returns a conversation row if it belongs to the user."""
    client = _get_client_or_raise()
    try:
        response = await run_in_threadpool(
            lambda: client.table("conversations")
            .select("*")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return response.data[0]
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to load conversation",
            status_code=500,
            details={"operation": "get_conversation"},
        ) from exc


async def list_messages(*, conversation_id: str) -> list[dict]:
    """Returns all messages in a conversation sorted by create time asc."""
    client = _get_client_or_raise()
    try:
        response = await run_in_threadpool(
            lambda: client.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data or []
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to load conversation messages",
            status_code=500,
            details={"operation": "list_messages"},
        ) from exc


async def update_conversation(
    *,
    conversation_id: str,
    user_id: str,
    update_payload: dict,
) -> dict | None:
    """Updates a conversation and returns the updated row."""
    client = _get_client_or_raise()
    try:
        response = await run_in_threadpool(
            lambda: client.table("conversations")
            .update(update_payload)
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )
        return response.data[0] if response.data else None
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to update conversation",
            status_code=500,
            details={"operation": "update_conversation"},
        ) from exc


async def delete_conversation(*, conversation_id: str, user_id: str) -> None:
    """Deletes a conversation owned by the user."""
    client = _get_client_or_raise()
    try:
        await run_in_threadpool(
            lambda: client.table("conversations")
            .delete()
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to delete conversation",
            status_code=500,
            details={"operation": "delete_conversation"},
        ) from exc


async def create_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None,
) -> dict | None:
    """Creates a message row and returns it."""
    client = _get_client_or_raise()
    payload = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "metadata": metadata,
    }
    try:
        response = await run_in_threadpool(
            lambda: client.table("messages").insert(payload).execute()
        )
        return response.data[0] if response.data else None
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to create message",
            status_code=500,
            details={"operation": "create_message"},
        ) from exc
