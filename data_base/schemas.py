"""
Pydantic Schemas for RAG API

Provides request/response models for the RAG question answering endpoints.
"""

# Standard library
from enum import Enum
from typing import List, Optional

# Third-party
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Valid roles for chat messages in conversation history."""
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """
    A single message in conversation history.

    Attributes:
        role: The role of the message sender (user or assistant).
        content: The message content.
    """
    role: MessageRole
    content: str


class AskRequest(BaseModel):
    """
    Request body for POST /ask endpoint.

    Attributes:
        question: The user's question.
        doc_ids: Optional list of document IDs to filter retrieval.
        history: Optional conversation history for context-aware responses.
        enable_hyde: Enable HyDE (Hypothetical Document Embeddings) retrieval.
        enable_multi_query: Enable multi-query fusion retrieval.
        enable_reranking: Enable Cross-Encoder reranking (recommended).
    """
    question: str = Field(
        ...,
        description="使用者問題",
        min_length=1,
        max_length=2000,
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="限定查詢的文件 ID 列表（留空則查詢全部文件）",
    )
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="對話歷史（最多 10 條，用於上下文感知對話）",
    )
    enable_hyde: bool = Field(
        default=False,
        description="啟用 HyDE 假設性文件增強檢索",
    )
    enable_multi_query: bool = Field(
        default=False,
        description="啟用多重查詢融合檢索",
    )
    enable_reranking: bool = Field(
        default=True,
        description="啟用 Cross-Encoder 重排序（建議開啟）",
    )


class AskResponse(BaseModel):
    """
    Response model for /ask endpoint.

    Attributes:
        question: Echo of the original question.
        answer: The generated answer.
        sources: List of source document IDs used in the response.
    """
    question: str
    answer: str
    sources: List[str] = []
