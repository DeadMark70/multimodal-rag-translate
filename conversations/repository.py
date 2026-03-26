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
