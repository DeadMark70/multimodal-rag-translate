"""Schemas for PDF upload responses."""

from __future__ import annotations

from datetime import datetime

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


class DocumentListItem(BaseModel):
    """Summary row in document list endpoint."""

    id: str
    file_name: str
    created_at: datetime
    status: str
    processing_step: str | None = None


class DocumentListResponse(BaseModel):
    """Response model for listing user documents."""

    documents: list[DocumentListItem] = Field(default_factory=list)
    total: int


class ProcessingStatusResponse(BaseModel):
    """Response model for document processing progress."""

    step: str
    step_label: str
    is_pdf_ready: bool
    is_fully_complete: bool


class DeleteDocumentResponse(BaseModel):
    """Response model for delete endpoint."""

    status: str
    message: str


class DocumentSummaryResponse(BaseModel):
    """Response model for summary retrieval."""

    summary: str | None = None
    status: str


class RegenerateSummaryResponse(BaseModel):
    """Response model for summary regeneration kickoff."""

    status: str
    message: str
