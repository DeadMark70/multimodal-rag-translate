"""Schemas for dashboard statistics."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentStat(BaseModel):
    """Document usage statistics."""

    doc_id: str
    filename: str | None = None
    query_count: int


class DashboardStats(BaseModel):
    """Dashboard statistics response model."""

    total_queries: int
    accuracy_rate: float
    grounded_count: int
    hallucinated_count: int
    uncertain_count: int
    avg_confidence: float
    queries_last_7_days: list[int] = Field(default_factory=list)
    top_documents: list[DocumentStat] = Field(default_factory=list)
