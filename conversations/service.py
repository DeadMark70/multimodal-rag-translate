"""Service layer for conversation APIs."""

from __future__ import annotations

from uuid import UUID

from core.errors import AppError, ErrorCode
from conversations.repository import (
    create_conversation as repo_create_conversation,
    create_message as repo_create_message,
    delete_conversation as repo_delete_conversation,
    get_conversation as repo_get_conversation,
    list_conversations as repo_list_conversations,
    list_messages as repo_list_messages,
    update_conversation as repo_update_conversation,
)
from conversations.schemas import (
    ChatMessageResponse,
    ConversationCreate,
    ConversationDetailResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageCreate,
)


async def list_user_conversations(*, user_id: str) -> list[ConversationResponse]:
    rows = await repo_list_conversations(user_id=user_id)
    return [ConversationResponse(**row) for row in rows]


async def create_user_conversation(
    *, user_id: str, data: ConversationCreate
) -> ConversationResponse:
    row = await repo_create_conversation(
        user_id=user_id,
        title=data.title,
        conversation_type=data.type,
        metadata=data.metadata,
    )
    if not row:
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to create conversation",
            status_code=500,
        )
    return ConversationResponse(**row)


async def get_user_conversation_detail(
    *, conversation_id: UUID, user_id: str
) -> ConversationDetailResponse:
    conversation_row = await repo_get_conversation(
        conversation_id=str(conversation_id), user_id=user_id
    )
    if not conversation_row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Conversation not found",
            status_code=404,
        )

    message_rows = await repo_list_messages(conversation_id=str(conversation_id))
    messages = [
        ChatMessageResponse(
            id=msg["id"],
            role=msg["role"],
            content=msg["content"],
            metadata=msg.get("metadata"),
            created_at=msg["created_at"],
        )
        for msg in message_rows
    ]
    return ConversationDetailResponse(**conversation_row, messages=messages)


async def update_user_conversation(
    *, conversation_id: UUID, user_id: str, data: ConversationUpdate
) -> ConversationResponse:
    update_payload: dict[str, object] = {"title": data.title}
    if data.metadata is not None:
        update_payload["metadata"] = data.metadata

    row = await repo_update_conversation(
        conversation_id=str(conversation_id),
        user_id=user_id,
        update_payload=update_payload,
    )
    if not row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Conversation not found",
            status_code=404,
        )
    return ConversationResponse(**row)


async def delete_user_conversation(*, conversation_id: UUID, user_id: str) -> None:
    conversation = await repo_get_conversation(
        conversation_id=str(conversation_id), user_id=user_id
    )
    if not conversation:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Conversation not found",
            status_code=404,
        )
    await repo_delete_conversation(conversation_id=str(conversation_id), user_id=user_id)


async def create_conversation_message(
    *, conversation_id: UUID, user_id: str, data: MessageCreate
) -> ChatMessageResponse:
    conversation = await repo_get_conversation(
        conversation_id=str(conversation_id), user_id=user_id
    )
    if not conversation:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Conversation not found",
            status_code=404,
        )

    row = await repo_create_message(
        conversation_id=str(conversation_id),
        role=data.role,
        content=data.content,
        metadata=data.metadata,
    )
    if not row:
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to create message",
            status_code=500,
        )

    return ChatMessageResponse(
        id=row["id"],
        role=row["role"],
        content=row["content"],
        metadata=row.get("metadata"),
        created_at=row["created_at"],
    )
