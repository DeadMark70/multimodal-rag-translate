"""Schemas for PDF upload responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadPdfResponse(BaseModel):
    """Response model for PDF upload and processing kickoff."""

    doc_id: str = Field(..., description="Document UUID")
    status: str = Field(..., description="Document status in database")
    message: str = Field(..., description="Human-readable upload result")
    pdf_available: bool = Field(
        ..., description="Whether translated PDF is already available"
    )
    pdf_download_url: str | None = Field(
        default=None, description="Download URL when PDF is available"
    )
    pdf_error: str | None = Field(
        default=None, description="PDF generation error if generation failed"
    )
    rag_status: str = Field(
        default="processing_background",
        description="Background RAG post-processing status",
    )
