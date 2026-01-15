# Backend Implementation Guide: Conversation Persistence

The frontend is receiving `404 Not Found` because the conversation endpoints are missing. Please implement the following Pydantic schemas and FastAPI router logic.

## 1. Update `schemas.py` (or equivalent)

Add these Pydantic models to handle conversation and message data types.

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from uuid import UUID

# --- Conversation Schemas ---

class ConversationCreate(BaseModel):
    title: str = "新對話"
    type: Literal["chat", "research"] = "chat"
    metadata: Optional[Dict[str, Any]] = {}

class ConversationUpdate(BaseModel):
    title: str

class Conversation(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    type: str
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

# --- Message Schemas ---

class MessageCreate(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    metadata: Optional[Dict[str, Any]] = {}

class Message(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationDetail(Conversation):
    messages: List[Message] = []
```

## 2. Create/Update `router.py` (e.g., `routers/conversations.py`)

Implement the endpoints. Ensure this router is included in your main `app.py` with `app.include_router(conversations.router, prefix="/api")`.

```python
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from uuid import UUID
from supabase import Client
# Import your 'get_supabase_client' dependency and schemas
# from .dependencies import get_current_user_id, get_supabase
# from .schemas import ConversationCreate, ConversationUpdate, Conversation, ConversationDetail, MessageCreate, Message

router = APIRouter(tags=["Conversations"])

@router.get("/conversations", response_model=List[Conversation])
def list_conversations(
    user_id: UUID = Depends(get_current_user_id),
    supabase: Client = Depends(get_supabase)
):
    response = supabase.table("conversations") \
        .select("*") \
        .eq("user_id", str(user_id)) \
        .order("updated_at", desc=True) \
        .execute()
    
    return response.data

@router.post("/conversations", response_model=Conversation)
def create_conversation(
    request: ConversationCreate,
    user_id: UUID = Depends(get_current_user_id),
    supabase: Client = Depends(get_supabase)
):
    data = {
        "user_id": str(user_id),
        "title": request.title,
        "type": request.type,
        "metadata": request.metadata
    }
    response = supabase.table("conversations").insert(data).execute()
    
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to create conversation")
        
    return response.data[0]

@router.post("/conversations/{conversation_id}/messages", response_model=Message)
def create_message(
    conversation_id: UUID,
    request: MessageCreate,
    user_id: UUID = Depends(get_current_user_id),
    supabase: Client = Depends(get_supabase)
):
    # Verify conversation ownership first
    conv_check = supabase.table("conversations")\
        .select("id")\
        .eq("id", str(conversation_id))\
        .eq("user_id", str(user_id))\
        .single()\
        .execute()
        
    if not conv_check.data:
        raise HTTPException(status_code=404, detail="Conversation not found")

    data = {
        "conversation_id": str(conversation_id),
        "role": request.role,
        "content": request.content,
        "metadata": request.metadata
    }
    
    response = supabase.table("messages").insert(data).execute()
    
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to create message")
        
    # Optional: Update conversation 'updated_at' timestamp
    supabase.table("conversations")\
        .update({"updated_at": "now()"})\
        .eq("id", str(conversation_id))\
        .execute()
        
    return response.data[0]
```