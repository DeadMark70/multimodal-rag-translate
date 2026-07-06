"""
PDF OCR & Translation Router

Provides API endpoints for PDF upload, OCR processing, translation, and file management.
"""

# Standard library
import asyncio
import logging
from typing import Literal
from uuid import UUID

# Third-party
from fastapi import APIRouter, UploadFile, File, Depends, Query, BackgroundTasks
from fastapi.responses import FileResponse

# Local application
from core.auth import get_current_user_id
from core import uploads as upload_paths
from pdfserviceMD.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentSummaryResponse,
    ProcessingStatusResponse,
    RegenerateSummaryResponse,
    RetryIndexResponse,
    TranslatePdfResponse,
    UploadPdfResponse,
)
from pdfserviceMD.service import (
    delete_user_document,
    get_document_file_info,
    get_document_processing_status,
    get_user_document_summary,
    list_user_documents,
    prepare_retry_index_context,
    regenerate_document_summary,
    run_upload_pipeline,
    translate_user_document,
)
from pdfserviceMD.indexing_tasks import (
    retry_document_index_task as _retry_document_index_task,
    run_post_processing_tasks,
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
upload_paths.ensure_upload_root()


# --- Document List Endpoint ---


@router.get("/list", response_model=DocumentListResponse)
async def list_documents_endpoint(
    user_id: str = Depends(get_current_user_id),
) -> DocumentListResponse:
    """
    Lists all documents uploaded by the user.

    Returns documents ordered by upload time (newest first).
    Limited to 50 documents maximum.

    Args:
        user_id: Authenticated user ID (injected).

    Returns:
        DocumentListResponse containing rows and total count.
    """
    response = await list_user_documents(user_id=user_id)
    logger.info("User %s listed %s documents", user_id, response.total)
    return response


# --- PDF Upload Endpoint ---
# Supports both /ocr (legacy) and /upload_pdf_md (new spec) paths


@router.post("/ocr")
@router.post("/upload_pdf_md")
async def upload_pdf_md(
    file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)
) -> UploadPdfResponse:
    """
    Uploads a PDF, runs OCR/translation/PDF generation, and starts background indexing.

    Pipeline:
    1. Save uploaded PDF
    2. OCR extraction (PaddleOCR - CPU bound, runs in threadpool)
    3. Add to RAG knowledge base
    4. Translate to Traditional Chinese (Gemini API)
    5. Generate translated PDF
    6. Return JSON status with doc_id and PDF availability

    Args:
        file: The uploaded PDF file.
        user_id: Authenticated user ID.

    Returns:
        UploadPdfResponse with processing status and download availability.

    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors.
    """
    # Input validation
    upload_paths.validate_pdf_upload(file)

    try:
        context = await run_upload_pipeline(
            file=file,
            user_id=user_id,
            base_upload_folder=upload_paths.ensure_upload_root(),
        )

        # Schedule background tasks (RAG + images + graph + summary).
        asyncio.create_task(
            run_post_processing_tasks(
                doc_id=context.doc_id,
                book_title=context.book_title,
                user_id=user_id,
                user_folder=context.user_folder,
            )
        )
        return context.response

    finally:
        await file.close()


@router.post("/file/{doc_id}/retry-index", response_model=RetryIndexResponse)
async def retry_document_index(
    doc_id: UUID,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> RetryIndexResponse:
    """Retry OCR-post indexing for an `index_failed` document without GraphRAG."""
    context = await prepare_retry_index_context(doc_id=str(doc_id), user_id=user_id)
    background_tasks.add_task(
        _retry_document_index_task,
        doc_id=context.doc_id,
        book_title=context.book_title,
        current_status=context.current_status,
        user_id=context.user_id,
        user_folder=context.user_folder,
    )
    return RetryIndexResponse(
        status="started",
        message="重新嵌入已開始",
    )


@router.get("/file/{doc_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> ProcessingStatusResponse:
    """
    Returns the current processing status of a document.

    Allows frontend to poll for progress updates during upload.

    Args:
        doc_id: Document UUID.
        user_id: Authenticated user ID.

    Returns:
        ProcessingStatusResponse with progress and readiness flags.
    """
    return await get_document_processing_status(doc_id=str(doc_id), user_id=user_id)


@router.get("/file/{doc_id}")
async def get_pdf_file(
    doc_id: UUID,
    type: Literal["original", "translated"] | None = Query(default=None),
    user_id: str = Depends(get_current_user_id),
) -> FileResponse:
    """
    Retrieves a processed PDF file by document ID.

    Args:
        doc_id: Document UUID.
        user_id: Authenticated user ID.

    Returns:
        FileResponse with the PDF file.

    Raises:
        AppError: 404 if document or file is unavailable.
    """
    file_path, filename = await get_document_file_info(
        doc_id=str(doc_id),
        user_id=user_id,
        file_type=type,
    )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf",
    )


@router.post("/file/{doc_id}/translate", response_model=TranslatePdfResponse)
async def translate_pdf_file(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> TranslatePdfResponse:
    """
    Translates an OCR-complete document and generates a translated PDF.

    Args:
        doc_id: Document UUID.
        user_id: Authenticated user ID.

    Returns:
        TranslatePdfResponse with translated PDF availability details.
    """
    return await translate_user_document(doc_id=str(doc_id), user_id=user_id)


@router.delete("/file/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_pdf_file(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> DeleteDocumentResponse:
    """
    Deletes a document and all associated files.

    Cleanup steps:
    1. Remove from FAISS vector index
    2. Delete physical files
    3. Remove database record

    Args:
        doc_id: Document UUID.
        user_id: Authenticated user ID.

    Returns:
        DeleteDocumentResponse.
    """
    doc_id_str = str(doc_id)
    logger.info(f"Delete request for doc {doc_id_str} by user {user_id}")
    return await delete_user_document(
        doc_id=doc_id_str,
        user_id=user_id,
        base_upload_folder=upload_paths.ensure_upload_root(),
    )


# --- Summary Endpoints ---


@router.get("/file/{doc_id}/summary", response_model=DocumentSummaryResponse)
async def get_document_summary_endpoint(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> DocumentSummaryResponse:
    """
    Retrieves the executive summary for a document.

    Args:
        doc_id: Document UUID.
        user_id: Authenticated user ID.

    Returns:
        DocumentSummaryResponse.
    """
    return await get_user_document_summary(doc_id=str(doc_id), user_id=user_id)


@router.post(
    "/file/{doc_id}/summary/regenerate",
    response_model=RegenerateSummaryResponse,
)
async def regenerate_summary_endpoint(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> RegenerateSummaryResponse:
    """
    Triggers regeneration of the executive summary.

    Retrieves the document's original text content and schedules
    a new background summary generation task.

    Args:
        doc_id: Document UUID.
        user_id: Authenticated user ID.

    Returns:
        Status indicating regeneration has started.
    """
    doc_id_str = str(doc_id)
    logger.info("Summary regeneration requested for doc %s", doc_id_str)
    return await regenerate_document_summary(doc_id=doc_id_str, user_id=user_id)
