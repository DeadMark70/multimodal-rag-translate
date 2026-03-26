"""
PDF OCR & Translation Router

Provides API endpoints for PDF upload, OCR processing, translation, and file management.
"""

# Standard library
import asyncio
import logging
import os
from typing import Literal
from uuid import UUID

# Third-party
from fastapi import APIRouter, UploadFile, File, Depends, Query, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

# Local application
from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from core import uploads as upload_paths
from data_base.indexing_service import index_markdown_document, index_visual_summaries
from data_base.vector_store_manager import delete_document_from_knowledge_base
from pdfserviceMD.image_processor import (
    extract_images_from_markdown,
    create_visual_elements,
)
from multimodal_rag.image_summarizer import summarizer as image_summarizer
from core.summary_service import schedule_summary_generation
from graph_rag.service import run_graph_extraction
from graph_rag.store import GraphStore
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
    finalize_indexing_status,
    get_document_file_info,
    get_document_processing_status,
    get_user_document_summary,
    load_ocr_artifacts,
    list_user_documents,
    record_background_processing_failure,
    regenerate_document_summary,
    run_upload_pipeline,
    safe_update_document_status,
    safe_update_processing_step,
    translate_user_document,
    update_indexing_processing_step,
)
from pdfserviceMD.repository import get_document

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
upload_paths.ensure_upload_root()


class DocumentImageProcessingError(RuntimeError):
    """Typed image-processing failure that preserves the failed stage."""

    def __init__(self, *, stage: str, detail: str) -> None:
        super().__init__(detail)
        self.stage = stage


def _format_image_processing_error(exc: DocumentImageProcessingError) -> str:
    """Maps internal image-processing stages to user-facing error text."""
    if exc.stage == "image_summary_api_failed":
        return f"Image summary generation failed: {exc}"
    if exc.stage == "visual_summary_index_failed":
        return f"Visual summary indexing failed: {exc}"
    return f"Image processing failed: {exc}"


async def _run_pre_graph_indexing_steps(
    *,
    doc_id: str,
    book_title: str,
    user_id: str,
    user_folder: str,
    markdown_text: str,
    clear_existing_vectors: bool,
) -> list[str]:
    """Run OCR-post indexing steps up to, but not including, GraphRAG."""
    error_messages: list[str] = []

    if clear_existing_vectors:
        try:
            await run_in_threadpool(delete_document_from_knowledge_base, user_id, doc_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[Background] Existing vector cleanup failed for doc %s: %s", doc_id, e)
            error_messages.append(f"Existing index cleanup failed: {e}")
            return error_messages

    try:
        await update_indexing_processing_step(
            doc_id=doc_id,
            user_id=user_id,
            step="indexing",
        )
        await index_markdown_document(
            user_id=user_id,
            markdown_text=markdown_text,
            pdf_title=book_title,
            doc_id=doc_id,
            k_retriever=3,
        )
        logger.info("[Background] RAG indexing complete for doc %s", doc_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("[Background] RAG indexing failed for doc %s: %s", doc_id, e)
        error_messages.append(f"RAG indexing failed: {e}")

    try:
        await update_indexing_processing_step(
            doc_id=doc_id,
            user_id=user_id,
            step="image_analysis",
        )
        await _process_document_images(
            user_id=user_id,
            doc_id=doc_id,
            markdown_text=markdown_text,
            user_folder=user_folder,
            book_title=book_title,
        )
    except DocumentImageProcessingError as e:
        logger.warning("[Background] %s for doc %s: %s", e.stage, doc_id, e)
        error_messages.append(_format_image_processing_error(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("[Background] Image analysis failed for doc %s: %s", doc_id, e)
        error_messages.append(f"Image analysis failed: {e}")

    return error_messages


async def run_post_processing_tasks(
    doc_id: str,
    book_title: str,
    user_id: str,
    user_folder: str,
) -> None:
    """
    Background task for post-PDF processing (RAG + Images + GraphRAG + Summary).

    Called after OCR artifact persistence completes.
    Runs RAG indexing, image summarization, GraphRAG extraction, and summary in background.

    Args:
        doc_id: Document UUID.
        book_title: Title for the document.
        user_id: User ID for knowledge base.
        user_folder: Path to document folder (for locating images).
    """
    logger.info(f"[Background] Starting post-processing for doc {doc_id}")

    try:
        markdown_text, _ = await run_in_threadpool(
            load_ocr_artifacts,
            user_folder=user_folder,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.warning(f"[Background] OCR artifacts unavailable for doc {doc_id}: {e}")
        await record_background_processing_failure(
            doc_id=doc_id,
            user_id=user_id,
            error_messages=[f"OCR artifacts unavailable: {e}"],
        )
        return

    error_messages = await _run_pre_graph_indexing_steps(
        doc_id=doc_id,
        book_title=book_title,
        user_id=user_id,
        user_folder=user_folder,
        markdown_text=markdown_text,
        clear_existing_vectors=False,
    )

    try:
        await update_indexing_processing_step(
            doc_id=doc_id,
            user_id=user_id,
            step="graph_indexing",
        )
        graph_result = await run_graph_extraction(
            user_id=user_id,
            doc_id=doc_id,
            markdown_text=markdown_text,
        )
        if graph_result.status in {"failed", "partial"} and graph_result.last_error:
            error_messages.append(f"Graph indexing failed: {graph_result.last_error}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Background] Graph indexing failed for doc {doc_id}: {e}")
        error_messages.append(f"Graph indexing failed: {e}")

    try:
        schedule_summary_generation(
            doc_id=doc_id,
            text_content=markdown_text,
            user_id=user_id,
        )
        logger.info(f"[Background] Summary scheduled for doc {doc_id}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Background] Summary scheduling failed for doc {doc_id}: {e}")
        error_messages.append(f"Summary scheduling failed: {e}")

    if error_messages:
        await record_background_processing_failure(
            doc_id=doc_id,
            user_id=user_id,
            error_messages=error_messages,
        )
        return

    await safe_update_processing_step(doc_id=doc_id, step="indexed")
    await finalize_indexing_status(doc_id=doc_id, user_id=user_id)


async def _retry_document_index_task(
    *,
    doc_id: str,
    book_title: str,
    current_status: str,
    user_id: str,
    user_folder: str,
) -> None:
    """Retry OCR-post indexing up to the GraphRAG boundary for one document."""
    logger.info("[Background] Starting retry-index for doc %s", doc_id)

    try:
        markdown_text, _ = await run_in_threadpool(
            load_ocr_artifacts,
            user_folder=user_folder,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.warning("[Background] OCR artifacts unavailable for retry-index doc %s: %s", doc_id, e)
        await record_background_processing_failure(
            doc_id=doc_id,
            user_id=user_id,
            error_messages=[f"OCR artifacts unavailable: {e}"],
        )
        return

    await safe_update_document_status(doc_id=doc_id, status=current_status, error_message=None)
    error_messages = await _run_pre_graph_indexing_steps(
        doc_id=doc_id,
        book_title=book_title,
        user_id=user_id,
        user_folder=user_folder,
        markdown_text=markdown_text,
        clear_existing_vectors=True,
    )

    if error_messages:
        await record_background_processing_failure(
            doc_id=doc_id,
            user_id=user_id,
            error_messages=error_messages,
        )
        return

    await safe_update_document_status(doc_id=doc_id, status=current_status, error_message=None)
    await safe_update_processing_step(doc_id=doc_id, step="indexed")
    await finalize_indexing_status(doc_id=doc_id, user_id=user_id)
    logger.info("[Background] Retry-index complete for doc %s", doc_id)


async def _process_document_images(
    user_id: str,
    doc_id: str,
    markdown_text: str,
    user_folder: str,
    book_title: str,
) -> int:
    """
    Process images in document: extract, summarize, and index.

    This is Phase 8 of the pipeline: image summarization for RAG.

    Args:
        user_id: User ID for knowledge base.
        doc_id: Document UUID.
        markdown_text: Markdown content to extract image paths from.
        user_folder: Path to document folder.
        book_title: Document title for context.

    Returns:
        Number of images processed.
    """
    try:
        # 1. Extract image paths from markdown
        image_data = extract_images_from_markdown(markdown_text, user_folder)

        if not image_data:
            logger.info(f"[Background] No images found in doc {doc_id}")
            return 0

        logger.info(f"[Background] Found {len(image_data)} images in doc {doc_id}")

        # 2. Create VisualElement objects
        elements = create_visual_elements(image_data, doc_title=book_title)
        figure_count = sum(
            1
            for element in elements
            if getattr(getattr(element, "type", None), "value", getattr(element, "type", None))
            == "figure"
        )

        # 3. Generate summaries using ImageSummarizer (async)
        logger.info(f"[Background] Summarizing {len(elements)} images...")
        summarized_elements = await image_summarizer.summarize_elements(
            elements=elements,
            doc_title=book_title,
        )

        # Count successful summaries
        success_count = sum(
            1 for e in summarized_elements if e.summary and "Error" not in e.summary
        )
        logger.info(f"[Background] Generated {success_count} image summaries")
        if figure_count > 0 and success_count == 0:
            raise DocumentImageProcessingError(
                stage="image_summary_api_failed",
                detail=f"No figure summaries were generated for {figure_count} extracted images",
            )

        # 4. Index summaries to vector store
        try:
            indexed_count = index_visual_summaries(
                user_id=user_id,
                doc_id=doc_id,
                elements=summarized_elements,
            )
        except Exception as exc:  # noqa: BLE001
            raise DocumentImageProcessingError(
                stage="visual_summary_index_failed",
                detail=(
                    f"Generated {success_count} image summaries, but vector indexing failed: {exc}"
                ),
            ) from exc

        if success_count > 0 and indexed_count == 0:
            raise DocumentImageProcessingError(
                stage="visual_summary_index_failed",
                detail=f"Generated {success_count} image summaries, but indexed 0 entries",
            )

        logger.info(
            f"[Background] Indexed {indexed_count} image summaries for doc {doc_id}"
        )
        return indexed_count

    except DocumentImageProcessingError:
        raise
    except FileNotFoundError as e:
        logger.warning(f"[Background] Image file not found: {e}")
        return 0
    except (RuntimeError, ValueError) as e:
        logger.warning(f"[Background] Image processing failed (non-fatal): {e}")
        return 0


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
    doc_id_str = str(doc_id)
    row = await get_document(
        doc_id=doc_id_str,
        user_id=user_id,
        columns="id, file_name, original_path, processing_step, status",
    )
    if not row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Document not found",
            status_code=404,
        )

    current_step = row.get("processing_step")
    if current_step != "index_failed":
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="Document is not eligible for retry-index",
            status_code=409,
        )

    if GraphStore(user_id).active_job_state:
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="Graph maintenance is currently running",
            status_code=409,
        )

    original_path = row.get("original_path")
    if not original_path:
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="Document has no original path",
            status_code=409,
        )

    user_folder = os.path.dirname(os.path.normpath(original_path))
    if not os.path.exists(os.path.join(user_folder, "extracted.md")):
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="Document has no OCR artifacts for retry-index",
            status_code=409,
        )

    book_title = os.path.splitext(row.get("file_name") or "document.pdf")[0]
    background_tasks.add_task(
        _retry_document_index_task,
        doc_id=doc_id_str,
        book_title=book_title,
        current_status=row.get("status") or "ready",
        user_id=user_id,
        user_folder=user_folder,
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
