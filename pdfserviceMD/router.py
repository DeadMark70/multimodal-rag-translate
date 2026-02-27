"""
PDF OCR & Translation Router

Provides API endpoints for PDF upload, OCR processing, translation, and file management.
"""

# Standard library
import asyncio
import logging
import os
from uuid import UUID

# Third-party
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import FileResponse

# Local application
from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from data_base.vector_store_manager import (
    add_markdown_to_knowledge_base,
    add_visual_summaries_to_knowledge_base,
)
from pdfserviceMD.image_processor import (
    extract_images_from_markdown,
    create_visual_elements,
)
from multimodal_rag.image_summarizer import summarizer as image_summarizer
from core.summary_service import schedule_summary_generation
from graph_rag.store import GraphStore
from graph_rag.extractor import extract_and_add_to_graph
from pdfserviceMD.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentSummaryResponse,
    ProcessingStatusResponse,
    RegenerateSummaryResponse,
    UploadPdfResponse,
)
from pdfserviceMD.service import (
    delete_user_document,
    get_document_file_info,
    get_document_processing_status,
    get_user_document_summary,
    list_user_documents,
    regenerate_document_summary,
    run_upload_pipeline,
    safe_update_processing_step,
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
BASE_UPLOAD_FOLDER = "uploads"
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)


def _validate_pdf_upload(file: UploadFile) -> None:
    """
    Validates that the uploaded file is a PDF.

    Args:
        file: The uploaded file object.

    Raises:
        AppError: 400 if file is not a valid PDF.
    """
    if file.content_type != "application/pdf":
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="File must be a PDF (invalid content-type)",
            status_code=400,
        )

    if file.filename:
        _, ext = os.path.splitext(file.filename)
        if ext.lower() != ".pdf":
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="File must be a PDF (invalid extension)",
                status_code=400,
            )


async def run_post_processing_tasks(
    doc_id: str,
    markdown_text: str,
    book_title: str,
    user_id: str,
    user_folder: str,
) -> None:
    """
    Background task for post-PDF processing (RAG + Images + GraphRAG + Summary).

    Called after the translated PDF is returned to user.
    Runs RAG indexing, image summarization, GraphRAG extraction, and summary in background.

    Args:
        doc_id: Document UUID.
        markdown_text: Extracted markdown content for indexing.
        book_title: Title for the document.
        user_id: User ID for knowledge base.
        user_folder: Path to document folder (for locating images).
    """
    logger.info(f"[Background] Starting post-processing for doc {doc_id}")
    await safe_update_processing_step(doc_id=doc_id, step="indexing")

    try:
        # 1. RAG indexing
        await add_markdown_to_knowledge_base(
            user_id=user_id,
            markdown_text=markdown_text,
            pdf_title=book_title,
            doc_id=doc_id,
            k_retriever=3,
        )
        logger.info(f"[Background] RAG indexing complete for doc {doc_id}")

        # 2. Image summarization and indexing (Phase 8)
        await safe_update_processing_step(doc_id=doc_id, step="image_analysis")
        await _process_document_images(
            user_id=user_id,
            doc_id=doc_id,
            markdown_text=markdown_text,
            user_folder=user_folder,
            book_title=book_title,
        )

        # 3. GraphRAG entity extraction
        await safe_update_processing_step(doc_id=doc_id, step="graph_indexing")
        await _run_graph_extraction(user_id, doc_id, markdown_text)

        # 3. Summary generation
        schedule_summary_generation(
            doc_id=doc_id,
            text_content=markdown_text,
            user_id=user_id,
        )
        logger.info(f"[Background] Summary scheduled for doc {doc_id}")

        await safe_update_processing_step(doc_id=doc_id, step="indexed")

    except (RuntimeError, ValueError) as e:
        logger.warning(f"[Background] Post-processing failed for doc {doc_id}: {e}")
        # Non-fatal: PDF was already delivered


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

        # 4. Index summaries to vector store
        indexed_count = add_visual_summaries_to_knowledge_base(
            user_id=user_id,
            doc_id=doc_id,
            elements=summarized_elements,
        )

        logger.info(
            f"[Background] Indexed {indexed_count} image summaries for doc {doc_id}"
        )
        return indexed_count

    except FileNotFoundError as e:
        logger.warning(f"[Background] Image file not found: {e}")
        return 0
    except (RuntimeError, ValueError) as e:
        logger.warning(f"[Background] Image processing failed (non-fatal): {e}")
        return 0


async def _run_graph_extraction(
    user_id: str,
    doc_id: str,
    markdown_text: str,
    batch_size: int = 3,
) -> None:
    """
    Run GraphRAG entity extraction on document content.

    Extracts entities and relations from the document and adds them to
    the user's knowledge graph. Uses batch parallel processing for efficiency.

    Args:
        user_id: User ID for graph store.
        doc_id: Document UUID.
        markdown_text: Text content to extract entities from.
        batch_size: Number of chunks to process in parallel (default 3).
    """
    try:
        # Split text into chunks for extraction (max ~8000 chars each)
        chunk_size = 8000
        all_chunks = [
            markdown_text[i : i + chunk_size]
            for i in range(0, len(markdown_text), chunk_size)
        ]

        # Pre-filter chunks that are too short
        chunks = [
            (idx, chunk)
            for idx, chunk in enumerate(all_chunks)
            if len(chunk.strip()) >= 100
        ]

        if not chunks:
            logger.info(f"[GraphRAG] No valid chunks to process for doc {doc_id}")
            return

        store = GraphStore(user_id)
        total_nodes = 0
        total_edges = 0

        # Process chunks in batches
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info(
            f"[GraphRAG] Processing {len(chunks)} chunks in {num_batches} batches "
            f"(batch_size={batch_size}) for doc {doc_id}"
        )

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            logger.info(f"[GraphRAG] Processing batch {batch_idx + 1}/{num_batches}...")

            # Create async tasks for parallel execution
            tasks = [
                extract_and_add_to_graph(
                    text=chunk,
                    doc_id=doc_id,
                    store=store,
                    chunk_index=idx,
                )
                for idx, chunk in batch
            ]

            # Execute batch in parallel, capturing exceptions
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                chunk_idx = batch[i][0]
                if isinstance(result, Exception):
                    logger.warning(
                        f"[GraphRAG] Chunk {chunk_idx} extraction failed: {result}"
                    )
                else:
                    nodes, edges = result
                    total_nodes += nodes
                    total_edges += edges

        # Save graph if any entities were extracted
        if total_nodes > 0:
            store.save()
            logger.info(
                f"[Background] GraphRAG complete for doc {doc_id}: "
                f"{total_nodes} nodes, {total_edges} edges"
            )
        else:
            logger.info(
                f"[Background] GraphRAG: No entities extracted from doc {doc_id}"
            )

    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"[GraphRAG] Store access failed for user {user_id}: {e}")
    except Exception as e:
        # Catch-all to prevent graph errors from breaking the pipeline
        logger.error(f"[GraphRAG] Unexpected error: {e}", exc_info=True)


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
    _validate_pdf_upload(file)

    try:
        context = await run_upload_pipeline(
            file=file,
            user_id=user_id,
            base_upload_folder=BASE_UPLOAD_FOLDER,
        )

        # Schedule background tasks (RAG + images + graph + summary).
        asyncio.create_task(
            run_post_processing_tasks(
                doc_id=context.doc_id,
                markdown_text=context.markdown_text,
                book_title=context.book_title,
                user_id=user_id,
                user_folder=context.user_folder,
            )
        )
        return context.response

    finally:
        await file.close()


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
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
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
    )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf",
    )


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
        base_upload_folder=BASE_UPLOAD_FOLDER,
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
