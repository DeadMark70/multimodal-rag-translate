"""
Conversation Schemas

Pydantic models for conversation management API.
"""

# Standard library
from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

# Third-party
from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    """
    Request model for creating a new conversation.
    
    Attributes:
        title: Conversation title (default: "新對話").
        type: Conversation type - 'chat' or 'research'.
    """
    title: Optional[str] = Field(default="新對話", max_length=200)
    type: Literal["chat", "research"] = "chat"


class ConversationUpdate(BaseModel):
    """
    Request model for updating a conversation.
    
    Attributes:
        title: New conversation title.
    """
    title: str = Field(..., min_length=1, max_length=200)


class ConversationResponse(BaseModel):
    """
    Response model for conversation basic info.
    
    Attributes:
        id: Unique conversation ID.
        title: Conversation title.
        type: Conversation type.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """
    id: UUID
    title: str
    type: Literal["chat", "research"]
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    """
    Response model for a single chat message.
    
    Attributes:
        id: Message ID.
        role: Message role ('user' or 'assistant').
        content: Message content.
        source_pages: Optional source page references.
        created_at: Message timestamp.
    """
    id: UUID
    role: Literal["user", "assistant"]
    content: str
    source_pages: Optional[dict] = None
    created_at: datetime


class ConversationDetailResponse(ConversationResponse):
    """
    Response model for conversation with messages.
    
    Extends ConversationResponse with a list of messages.
    
    Attributes:
        messages: List of chat messages in chronological order.
    """
    messages: List[ChatMessageResponse] = []
