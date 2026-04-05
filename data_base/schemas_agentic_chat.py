"""
Chat Agentic Benchmark Schemas

Request/response contracts for chat-facing agentic benchmark streaming.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from data_base.schemas_deep_research import ExecutePlanResponse


class AgenticBenchmarkStreamRequest(BaseModel):
    """Request model for POST /rag/agentic/stream."""

    question: str = Field(..., min_length=1, max_length=2000)
    doc_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    enable_reranking: bool = True
    enable_deep_image_analysis: bool = True


class AgenticBenchmarkCompletePayload(BaseModel):
    """Final SSE payload emitted by the chat agentic benchmark stream."""

    result: ExecutePlanResponse
    agent_trace: dict[str, Any]
