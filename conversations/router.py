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
from fastapi import APIRouter, Depends, HTTPException
from postgrest.exceptions import APIError as PostgrestAPIError

# Local application
from core.auth import get_current_user_id
from supabase_client import supabase
from conversations.schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationDetailResponse,
    ChatMessageResponse,
    MessageCreate,
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
    
    try:
        response = supabase.table("conversations") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .execute()
        
        return [ConversationResponse(**row) for row in response.data]
        
    except PostgrestAPIError as e:
        logger.error(f"Failed to list conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="無法取得對話列表")


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
    
    try:
        insert_data = {
            "user_id": user_id,
            "title": data.title,
            "type": data.type,
            "metadata": data.metadata,
        }
        
        response = supabase.table("conversations") \
            .insert(insert_data) \
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="建立對話失敗")
        
        return ConversationResponse(**response.data[0])
        
    except PostgrestAPIError as e:
        logger.error(f"Failed to create conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="建立對話失敗")


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: UUID,
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
    
    try:
        # Get conversation
        conv_response = supabase.table("conversations") \
            .select("*") \
            .eq("id", str(conversation_id)) \
            .eq("user_id", user_id) \
            .single() \
            .execute()
        
        if not conv_response.data:
            raise HTTPException(status_code=404, detail="對話不存在")
        
        # Get messages from 'messages' table
        msgs_response = supabase.table("messages") \
            .select("*") \
            .eq("conversation_id", str(conversation_id)) \
            .order("created_at", desc=False) \
            .execute()
        
        # Transform messages
        messages = []
        for msg in msgs_response.data:
            messages.append(ChatMessageResponse(
                id=msg["id"],
                role=msg["role"],
                content=msg["content"],
                metadata=msg.get("metadata"),
                created_at=msg["created_at"],
            ))
        
        # Merge created_at/updated_at/metadata from conversation if needed, 
        # but the query already returns them.
        
        return ConversationDetailResponse(
            **conv_response.data,
            messages=messages,
        )
        
    except PostgrestAPIError as e:
        logger.error(f"Failed to get conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="無法取得對話詳情")


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
    
    try:
        update_data = {"title": data.title}
        if data.metadata is not None:
            update_data["metadata"] = data.metadata

        response = supabase.table("conversations") \
            .update(update_data) \
            .eq("id", str(conversation_id)) \
            .eq("user_id", user_id) \
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="對話不存在")
        
        return ConversationResponse(**response.data[0])
        
    except PostgrestAPIError as e:
        logger.error(f"Failed to update conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="更新對話失敗")


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
    
    try:
        # First check if conversation exists and belongs to user
        check = supabase.table("conversations") \
            .select("id") \
            .eq("id", str(conversation_id)) \
            .eq("user_id", user_id) \
            .execute()
        
        if not check.data:
            raise HTTPException(status_code=404, detail="對話不存在")
        
        # Delete conversation (messages cascade delete)
        supabase.table("conversations") \
            .delete() \
            .eq("id", str(conversation_id)) \
            .eq("user_id", user_id) \
            .execute()
        
    except PostgrestAPIError as e:
        logger.error(f"Failed to delete conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="刪除對話失敗")


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
    
    try:
        # Verify conversation exists and belongs to user
        # (Technically RLS handles this, but explicit 404 is nicer)
        conv_check = supabase.table("conversations") \
            .select("id") \
            .eq("id", str(conversation_id)) \
            .eq("user_id", user_id) \
            .single() \
            .execute()
            
        if not conv_check.data:
            raise HTTPException(status_code=404, detail="對話不存在")
            
        insert_data = {
            "conversation_id": str(conversation_id),
            "role": data.role,
            "content": data.content,
            "metadata": data.metadata,
        }
        
        response = supabase.table("messages") \
            .insert(insert_data) \
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="建立訊息失敗")
            
        msg = response.data[0]
        return ChatMessageResponse(
            id=msg["id"],
            role=msg["role"],
            content=msg["content"],
            metadata=msg.get("metadata"),
            created_at=msg["created_at"],
        )
        
    except PostgrestAPIError as e:
        logger.error(f"Failed to add message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="建立訊息失敗")
