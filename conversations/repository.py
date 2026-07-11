"""Persistence layer for conversation-related Supabase operations."""

from __future__ import annotations

from core.supabase_repository import execute_supabase_operation


async def list_conversations(*, user_id: str) -> list[dict]:
    """Returns all conversations for a user sorted by updated time desc."""
    response = await execute_supabase_operation(
        operation="list_conversations",
        failure_message="Failed to list conversations",
        handler=lambda client: client.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute(),
    )
    return response.data or []


async def list_conversation_page(
    *,
    user_id: str,
    limit: int,
    cursor_updated_at: str | None = None,
    cursor_id: str | None = None,
    search: str | None = None,
) -> tuple[list[dict], bool]:
    """Return bounded conversation summaries using a keyset cursor."""
    def handler(client):
        query = (
            client.table("conversations")
            .select(
                "id,title,type,created_at,updated_at,"
                "mode_preset:metadata->>mode_preset,"
                "mode_config_snapshot:metadata->mode_config_snapshot,"
                "enable_graph_planning:metadata->enable_graph_planning"
            )
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .order("id", desc=True)
        )
        if cursor_updated_at and cursor_id:
            query = query.or_(
                "updated_at.lt.{updated},and(updated_at.eq.{updated},id.lt.{id})".format(
                    updated=cursor_updated_at,
                    id=cursor_id,
                )
            )
        if search:
            query = query.ilike("title", f"%{search}%")
        return query.limit(limit + 1).execute()

    response = await execute_supabase_operation(
        operation="list_conversation_page",
        failure_message="Failed to list conversation summaries",
        handler=handler,
    )
    rows = response.data or []
    return rows[:limit], len(rows) > limit


async def create_conversation(
    *,
    user_id: str,
    title: str | None,
    conversation_type: str,
    metadata: dict | None,
) -> dict | None:
    """Creates a conversation and returns the created row."""
    payload = {
        "user_id": user_id,
        "title": title,
        "type": conversation_type,
        "metadata": metadata,
    }
    response = await execute_supabase_operation(
        operation="create_conversation",
        failure_message="Failed to create conversation",
        handler=lambda client: client.table("conversations").insert(payload).execute(),
    )
    return response.data[0] if response.data else None


async def get_conversation(*, conversation_id: str, user_id: str) -> dict | None:
    """Returns a conversation row if it belongs to the user."""
    response = await execute_supabase_operation(
        operation="get_conversation",
        failure_message="Failed to load conversation",
        handler=lambda client: client.table("conversations")
            .select("*")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute(),
    )
    if not response.data:
        return None
    return response.data[0]


async def list_messages(*, conversation_id: str) -> list[dict]:
    """Returns all messages in a conversation sorted by create time asc."""
    response = await execute_supabase_operation(
        operation="list_messages",
        failure_message="Failed to load conversation messages",
        handler=lambda client: client.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .execute(),
    )
    return response.data or []


async def list_messages_page(
    *,
    conversation_id: str,
    limit: int,
    cursor_created_at: str | None = None,
    cursor_id: str | None = None,
) -> tuple[list[dict], bool]:
    """Return the newest bounded message page, optionally before a cursor."""
    def handler(client):
        query = (
            client.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=True)
            .order("id", desc=True)
        )
        if cursor_created_at and cursor_id:
            query = query.or_(
                "created_at.lt.{created},and(created_at.eq.{created},id.lt.{id})".format(
                    created=cursor_created_at,
                    id=cursor_id,
                )
            )
        return query.limit(limit + 1).execute()

    response = await execute_supabase_operation(
        operation="list_messages_page",
        failure_message="Failed to load conversation message page",
        handler=handler,
    )
    rows = response.data or []
    return rows[:limit], len(rows) > limit


async def update_conversation(
    *,
    conversation_id: str,
    user_id: str,
    update_payload: dict,
) -> dict | None:
    """Updates a conversation and returns the updated row."""
    response = await execute_supabase_operation(
        operation="update_conversation",
        failure_message="Failed to update conversation",
        handler=lambda client: client.table("conversations")
            .update(update_payload)
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .execute(),
    )
    return response.data[0] if response.data else None


async def delete_conversation(*, conversation_id: str, user_id: str) -> None:
    """Deletes a conversation owned by the user."""
    await execute_supabase_operation(
        operation="delete_conversation",
        failure_message="Failed to delete conversation",
        handler=lambda client: client.table("conversations")
        .delete()
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .execute(),
    )


async def create_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None,
) -> dict | None:
    """Creates a message row and returns it."""
    payload = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "metadata": metadata,
    }
    response = await execute_supabase_operation(
        operation="create_message",
        failure_message="Failed to create message",
        handler=lambda client: client.table("messages").insert(payload).execute(),
    )
    return response.data[0] if response.data else None
