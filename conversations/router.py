"""
Conversations Router

API endpoints for conversation history management.
Supports CRUD operations for conversations.
"""

# Standard library
import logging
from typing import List
from uuid import UUID

# Third-party
from fastapi import APIRouter, Depends, Query

# Local application
from core.auth import get_current_user_id
from conversations.schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationPageResponse,
    MessagePageResponse,
    ConversationDetailResponse,
    ChatMessageResponse,
    MessageCreate,
)
from conversations.service import (
    create_conversation_message,
    create_user_conversation,
    delete_user_conversation,
    get_user_conversation_detail,
    list_user_conversations,
    list_user_conversation_page,
    list_user_message_page,
    update_user_conversation,
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    user_id: str = Depends(get_current_user_id)
) -> List[ConversationResponse]:
    """
    Lists all conversations for the authenticated user.
    
    Returns conversations ordered by updated_at descending (most recent first).
    
    Args:
        user_id: Authenticated user ID (injected).
        
    Returns:
        List of ConversationResponse.
        
    Raises:
        HTTPException: 500 if database query fails.
    """
    logger.info(f"Listing conversations for user {user_id}")
    return await list_user_conversations(user_id=user_id)


@router.get("/page", response_model=ConversationPageResponse)
async def list_conversation_page(
    limit: int = Query(default=40, ge=1, le=100),
    cursor: str | None = Query(default=None),
    search: str | None = Query(default=None, max_length=200),
    user_id: str = Depends(get_current_user_id),
) -> ConversationPageResponse:
    """Lists bounded conversation summaries using a keyset cursor."""
    return await list_user_conversation_page(
        user_id=user_id,
        limit=limit,
        cursor=cursor,
        search=search,
    )


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    data: ConversationCreate,
    user_id: str = Depends(get_current_user_id)
) -> ConversationResponse:
    """
    Creates a new conversation.
    
    Args:
        data: Conversation creation data.
        user_id: Authenticated user ID (injected).
        
    Returns:
        Created ConversationResponse.
        
    Raises:
        HTTPException: 500 if creation fails.
    """
    logger.info(f"Creating conversation for user {user_id}: {data.title}")
    return await create_user_conversation(user_id=user_id, data=data)


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: UUID,
    include_messages: bool = Query(
        default=True,
        description="Whether to include full message history in response.",
    ),
    user_id: str = Depends(get_current_user_id)
) -> ConversationDetailResponse:
    """
    Gets a conversation with all its messages.
    
    Args:
        conversation_id: Conversation UUID.
        user_id: Authenticated user ID (injected).
        
    Returns:
        ConversationDetailResponse with messages.
        
    Raises:
        HTTPException: 404 if not found, 500 if query fails.
    """
    logger.info(f"Getting conversation {conversation_id} for user {user_id}")
    return await get_user_conversation_detail(
        conversation_id=conversation_id,
        user_id=user_id,
        include_messages=include_messages,
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: UUID,
    data: ConversationUpdate,
    user_id: str = Depends(get_current_user_id)
) -> ConversationResponse:
    """
    Updates a conversation's title or metadata.
    
    Args:
        conversation_id: Conversation UUID.
        data: Update data.
        user_id: Authenticated user ID (injected).
        
    Returns:
        Updated ConversationResponse.
        
    Raises:
        HTTPException: 404 if not found, 500 if update fails.
    """
    logger.info(f"Updating conversation {conversation_id}: {data.title}")
    return await update_user_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        data=data,
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    user_id: str = Depends(get_current_user_id)
) -> None:
    """
    Deletes a conversation and all its messages.
    
    Messages are deleted via CASCADE foreign key constraint.
    
    Args:
        conversation_id: Conversation UUID.
        user_id: Authenticated user ID (injected).
        
    Raises:
        HTTPException: 404 if not found, 500 if delete fails.
    """
    logger.info(f"Deleting conversation {conversation_id}")
    await delete_user_conversation(conversation_id=conversation_id, user_id=user_id)


@router.get("/{conversation_id}/messages/page", response_model=MessagePageResponse)
async def list_message_page(
    conversation_id: UUID,
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None),
    user_id: str = Depends(get_current_user_id),
) -> MessagePageResponse:
    """Loads a bounded newest message page for a conversation."""
    return await list_user_message_page(
        conversation_id=conversation_id,
        user_id=user_id,
        limit=limit,
        cursor=cursor,
    )


@router.post("/{conversation_id}/messages", response_model=ChatMessageResponse, status_code=201)
async def create_message(
    conversation_id: UUID,
    data: MessageCreate,
    user_id: str = Depends(get_current_user_id)
) -> ChatMessageResponse:
    """
    Adds a message to a conversation.
    
    Args:
        conversation_id: Conversation UUID.
        data: Message content and role.
        user_id: Authenticated user ID (injected).
        
    Returns:
        Created ChatMessageResponse.
        
    Raises:
        HTTPException: 404 if conversation not found, 500 if insert fails.
    """
    logger.info(f"Adding message to conversation {conversation_id} for user {user_id}")
    return await create_conversation_message(
        conversation_id=conversation_id,
        user_id=user_id,
        data=data,
    )
