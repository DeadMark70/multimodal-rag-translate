"""
Chat Agentic Benchmark Schemas

Request/response contracts for chat-facing agentic benchmark streaming.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from data_base.schemas_deep_research import ExecutePlanResponse

MAX_AGENTIC_HISTORY_MESSAGES = 10
"""Maximum chat messages admitted for agentic query resolution."""

MAX_AGENTIC_HISTORY_TOKENS = 1024
"""Conservative token ceiling for the bounded query-resolution history."""


class AgenticHistoryMessage(BaseModel):
    """One untrusted conversational turn used only to resolve a query."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class AgenticBenchmarkStreamRequest(BaseModel):
    """Request model for POST /rag/agentic/stream."""

    question: str = Field(..., min_length=1, max_length=2000)
    doc_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    history: List[AgenticHistoryMessage] = Field(
        default_factory=list,
        max_length=MAX_AGENTIC_HISTORY_MESSAGES,
    )
    agentic_execution_version: Literal["v8", "v9"] = "v8"
    enable_reranking: bool = True
    enable_deep_image_analysis: bool = True


class AgenticBenchmarkCompletePayload(BaseModel):
    """Final SSE payload emitted by the chat agentic benchmark stream."""

    result: ExecutePlanResponse
    agent_trace: dict[str, Any]
