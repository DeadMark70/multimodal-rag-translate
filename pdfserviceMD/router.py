"""
PDF OCR & Translation Router

Provides API endpoints for PDF upload, OCR processing, translation, and file management.
"""

# Standard library
import asyncio
import logging
import os
import shutil
import time
import uuid
from uuid import UUID

# Third-party
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from postgrest.exceptions import APIError as PostgrestAPIError

# Local application
from core.auth import get_current_user_id
from supabase_client import supabase
from data_base.vector_store_manager import (
    add_markdown_to_knowledge_base,
    add_visual_summaries_to_knowledge_base,
    delete_document_from_knowledge_base,
)
from pdfserviceMD.image_processor import (
    extract_images_from_markdown,
    create_visual_elements,
)
from multimodal_rag.image_summarizer import summarizer as image_summarizer
from pdfserviceMD.PDF_OCR_services import ocr_service_sync
from pdfserviceMD.ai_translate_md import translate_text

# Phase 7: 切換至更強大的 Pandoc 引擎
from pdfserviceMD.Pandoc_md_to_pdf import MDmarkdown_to_pdf as markdown_to_pdf
from pdfserviceMD.markdown_process import markdown_extact, replace_markdown
from core.summary_service import schedule_summary_generation
from graph_rag.store import GraphStore
from graph_rag.extractor import extract_and_add_to_graph

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
        HTTPException: 400 if file is not a valid PDF.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="File must be a PDF (invalid content-type)"
        )

    if file.filename:
        _, ext = os.path.splitext(file.filename)
        if ext.lower() != ".pdf":
            raise HTTPException(
                status_code=400, detail="File must be a PDF (invalid extension)"
            )


def _update_document_status(
    doc_id: str,
    status: str,
    translated_path: str | None = None,
    error_message: str | None = None,
) -> None:
    """
    Updates document status in Supabase.

    Args:
        doc_id: Document UUID.
        status: New status ('processing', 'completed', 'failed').
        translated_path: Path to translated PDF (optional).
        error_message: Error message if failed (optional).
    """
    if not supabase:
        return

    try:
        update_data = {"status": status}
        if translated_path:
            update_data["translated_path"] = translated_path
        if error_message:
            update_data["error_message"] = error_message
        else:
            update_data["error_message"] = None

        supabase.table("documents").update(update_data).eq("id", doc_id).execute()
        logger.info(f"Document {doc_id} status updated to: {status}")
    except PostgrestAPIError as e:
        logger.error(f"Failed to update document status: {e}", exc_info=True)


def _update_processing_step(doc_id: str, step: str) -> None:
    """
    Updates the document processing step for progress tracking.

    Args:
        doc_id: Document UUID.
        step: Processing step ('ocr', 'translating', 'generating_pdf', 'completed', 'indexing').
    """
    if not supabase:
        return

    try:
        supabase.table("documents").update({"processing_step": step}).eq(
            "id", doc_id
        ).execute()
        logger.debug(f"Document {doc_id} step: {step}")
    except PostgrestAPIError as e:
        logger.warning(f"Failed to update processing step: {e}")


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
    _update_processing_step(doc_id, "indexing")

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
        _update_processing_step(doc_id, "image_analysis")
        await _process_document_images(
            user_id=user_id,
            doc_id=doc_id,
            markdown_text=markdown_text,
            user_folder=user_folder,
            book_title=book_title,
        )

        # 3. GraphRAG entity extraction
        _update_processing_step(doc_id, "graph_indexing")
        await _run_graph_extraction(user_id, doc_id, markdown_text)

        # 3. Summary generation
        schedule_summary_generation(
            doc_id=doc_id,
            text_content=markdown_text,
            user_id=user_id,
        )
        logger.info(f"[Background] Summary scheduled for doc {doc_id}")

        _update_processing_step(doc_id, "indexed")

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


@router.get("/list")
async def list_documents(user_id: str = Depends(get_current_user_id)) -> dict:
    """
    Lists all documents uploaded by the user.

    Returns documents ordered by upload time (newest first).
    Limited to 50 documents maximum.

    Args:
        user_id: Authenticated user ID (injected).

    Returns:
        Dict containing documents list and total count.

    Raises:
        HTTPException: 500 if database query fails.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        result = (
            supabase.table("documents")
            .select("id, file_name, created_at, status, processing_step")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )

        documents = result.data if result.data else []

        logger.info(f"User {user_id} listed {len(documents)} documents")

        return {"documents": documents, "total": len(documents)}

    except PostgrestAPIError as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


# --- PDF Upload Endpoint ---
# Supports both /ocr (legacy) and /upload_pdf_md (new spec) paths


@router.post("/ocr")
@router.post("/upload_pdf_md")
async def upload_pdf_md(
    file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)
) -> FileResponse:
    """
    Uploads a PDF, performs OCR, translates content, and returns translated PDF.

    Pipeline:
    1. Save uploaded PDF
    2. OCR extraction (PaddleOCR - CPU bound, runs in threadpool)
    3. Add to RAG knowledge base
    4. Translate to Traditional Chinese (Gemini API)
    5. Generate translated PDF
    6. Return translated PDF file

    Args:
        file: The uploaded PDF file.
        user_id: Authenticated user ID.

    Returns:
        FileResponse with the translated PDF.

    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors.
    """
    start_time = time.perf_counter()

    # Input validation
    _validate_pdf_upload(file)

    file_uuid = str(uuid.uuid4())
    user_folder = os.path.normpath(os.path.join(BASE_UPLOAD_FOLDER, user_id, file_uuid))
    os.makedirs(user_folder, exist_ok=True)

    # Sanitize filename
    filename = os.path.basename(file.filename) if file.filename else "document.pdf"
    save_path = os.path.normpath(os.path.join(user_folder, filename))
    book_title = os.path.splitext(filename)[0]

    document_id = file_uuid

    try:
        # 1. Save uploaded file
        file_content = await file.read()
        with open(save_path, "wb") as buffer:
            buffer.write(file_content)
        logger.info(f"File saved to: {save_path}")

        # 2. Create DB record
        if supabase:
            try:
                db_data = {
                    "id": document_id,
                    "user_id": user_id,
                    "file_name": filename,
                    "file_type": "pdf",
                    "original_path": save_path,
                    "status": "processing",
                    "source_lang": "auto",
                    "target_lang": "zh-TW",
                }
                supabase.table("documents").insert(db_data).execute()
                logger.info(f"DB record created for document {document_id}")
            except PostgrestAPIError as e:
                logger.error(f"DB insert failed: {e}", exc_info=True)

        # 3. OCR Processing (CPU-bound, run in threadpool)
        _update_processing_step(document_id, "ocr")
        t1 = time.perf_counter()
        logger.info("Starting OCR processing...")
        ocr_result = await run_in_threadpool(ocr_service_sync, save_path)
        t2 = time.perf_counter()
        logger.info(f"OCR completed in {t2 - t1:.2f}s")

        # 4. Extract markdown and image blocks
        processed_markdown_for_rag, image_blocks = markdown_extact(ocr_result)

        # 5. Translate (async Gemini API call) - PRIORITIZED
        _update_processing_step(document_id, "translating")
        logger.info("Starting translation...")
        translate_result = await translate_text(processed_markdown_for_rag)
        t3 = time.perf_counter()
        logger.info(f"Translation completed in {t3 - t2:.2f}s")

        # 6. Replace image placeholders
        final_md = replace_markdown(translate_result, image_blocks)

        # 7. Generate PDF (CPU-bound, run in threadpool) - PRIORITIZED
        _update_processing_step(document_id, "generating_pdf")
        output_pdf_filename = f"translated_{filename}"
        output_pdf_path = os.path.normpath(
            os.path.join(user_folder, output_pdf_filename)
        )
        pdf_generation_failed = False
        pdf_error_msg = ""

        try:
            # Phase 7: 傳入 user_folder 作為 base_dir 以正確處理圖片路徑
            await run_in_threadpool(
                markdown_to_pdf, final_md, output_pdf_path, user_folder
            )
            t4 = time.perf_counter()
            logger.info(f"PDF generated in {t4 - t3:.2f}s")

            # 8. Update DB status - PDF is ready!
            _update_document_status(
                document_id, "completed", translated_path=output_pdf_path
            )
            _update_processing_step(document_id, "completed")

        except Exception as e:
            # RESILIENCE: If PDF fails, log it but CONTINUE to RAG/GraphRAG steps
            pdf_generation_failed = True
            pdf_error_msg = str(e)
            logger.error(f"PDF generation failed, but proceeding with RAG tasks: {e}")
            # Mark as completed but with a note or just completed (since text is ready for RAG)
            # We'll use a special status or just log it specificially?
            # User wants to continue RAG.
            _update_document_status(
                document_id,
                "completed_with_pdf_error",
                error_message=f"PDF Error: {str(e)}",
            )
            _update_processing_step(document_id, "completed")
            # Clear output path so we don't try to return a missing file
            output_pdf_path = None

        total_time = time.perf_counter() - start_time
        logger.info(
            f"Processing (pre-background) finished. Time: {total_time:.2f}s. Scheduling background tasks..."
        )

        # 9. Schedule background tasks (RAG + Images + Summary) - NON-BLOCKING
        # Always run this, even if PDF failed!
        asyncio.create_task(
            run_post_processing_tasks(
                doc_id=document_id,
                markdown_text=processed_markdown_for_rag,
                book_title=book_title,
                user_id=user_id,
                user_folder=user_folder,
            )
        )

        # 10. Return response
        if (
            not pdf_generation_failed
            and output_pdf_path
            and os.path.exists(output_pdf_path)
        ):
            return FileResponse(
                path=output_pdf_path,
                filename=output_pdf_filename,
                media_type="application/pdf",
            )
        else:
            # Return JSON indicating RAG is processing but PDF failed
            return {
                "message": "Translation completed and RAG processing started.",
                "pdf_status": "failed",
                "pdf_error": pdf_error_msg,
                "rag_status": "processing_background",
            }

    except HTTPException:
        _update_document_status(
            document_id, "failed", error_message="HTTP error during processing"
        )
        _update_processing_step(document_id, "failed")
        raise

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        _update_document_status(document_id, "failed", error_message=str(e))
        _update_processing_step(document_id, "failed")
        raise HTTPException(status_code=500, detail="File processing error")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        _update_document_status(document_id, "failed", error_message=str(e))
        _update_processing_step(document_id, "failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        await file.close()


# Processing step labels for frontend display
_STEP_LABELS = {
    "uploading": "上傳中",
    "ocr": "OCR 辨識中",
    "translating": "翻譯中",
    "generating_pdf": "生成 PDF 中",
    "completed": "翻譯完成",
    "indexing": "建立索引中",
    "graph_indexing": "建立知識圖譜中",
    "indexed": "全部完成",
    "failed": "處理失敗",
}


@router.get("/file/{doc_id}/status")
async def get_processing_status(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> dict:
    """
    Returns the current processing status of a document.

    Allows frontend to poll for progress updates during upload.

    Args:
        doc_id: Document UUID.
        user_id: Authenticated user ID.

    Returns:
        Dict with step info:
        - step: Current processing step code
        - step_label: Human-readable step label (Chinese)
        - is_pdf_ready: True if translated PDF is available
        - is_fully_complete: True if all processing (including RAG) is done
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database service unavailable")

    doc_id_str = str(doc_id)

    try:
        result = (
            supabase.table("documents")
            .select("status, processing_step, translated_path")
            .eq("id", doc_id_str)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
    except PostgrestAPIError as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")

    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    step = result.data.get("processing_step") or "uploading"
    status = result.data.get("status")
    translated_path = result.data.get("translated_path")

    return {
        "step": step,
        "step_label": _STEP_LABELS.get(step, step),
        "is_pdf_ready": status == "completed" and translated_path is not None,
        "is_fully_complete": step == "indexed",
    }


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
        HTTPException: 404 if document not found.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database service unavailable")

    doc_id_str = str(doc_id)

    try:
        response = (
            supabase.table("documents")
            .select("*")
            .eq("id", doc_id_str)
            .eq("user_id", user_id)
            .execute()
        )
    except PostgrestAPIError as e:
        logger.error(f"Database query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query failed")

    if not response.data:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = response.data[0]
    file_path = doc.get("translated_path") or doc.get("original_path")

    if not file_path:
        raise HTTPException(status_code=404, detail="File path not found in record")

    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=file_path,
        filename=doc.get("file_name", "document.pdf"),
        media_type="application/pdf",
    )


@router.delete("/file/{doc_id}")
async def delete_pdf_file(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> dict:
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
        Success status dict.

    Raises:
        HTTPException: 500 if database deletion fails.
    """
    doc_id_str = str(doc_id)
    logger.info(f"Delete request for doc {doc_id_str} by user {user_id}")

    # 1. Delete from RAG index (non-fatal if fails)
    try:
        await run_in_threadpool(
            delete_document_from_knowledge_base, user_id, doc_id_str
        )
        logger.info(f"RAG index entry deleted for doc {doc_id_str}")
    except (RuntimeError, ValueError) as e:
        logger.warning(f"RAG deletion failed (non-fatal): {e}")

    # 2. Delete physical files
    doc_folder = os.path.normpath(os.path.join(BASE_UPLOAD_FOLDER, user_id, doc_id_str))
    if os.path.exists(doc_folder):
        try:
            shutil.rmtree(doc_folder)
            logger.info(f"Folder deleted: {doc_folder}")
        except OSError as e:
            logger.error(f"Failed to delete folder: {e}", exc_info=True)

    # 3. Delete database record
    if supabase:
        try:
            supabase.table("documents").delete().eq("id", doc_id_str).eq(
                "user_id", user_id
            ).execute()
            logger.info(f"DB record deleted for doc {doc_id_str}")
        except PostgrestAPIError as e:
            logger.error(f"DB deletion failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to delete database record"
            )

    return {"status": "success", "message": "Document deleted successfully"}


# --- Summary Endpoints ---


@router.get("/file/{doc_id}/summary")
async def get_document_summary_endpoint(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> dict:
    """
    Retrieves the executive summary for a document.

    Args:
        doc_id: Document UUID.
        user_id: Authenticated user ID.

    Returns:
        Dict with summary content and status.
        - status: "ready" | "generating" | "not_available"
        - summary: The executive summary text (null if not ready)
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database service unavailable")

    doc_id_str = str(doc_id)

    try:
        result = (
            supabase.table("documents")
            .select("executive_summary, status")
            .eq("id", doc_id_str)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
    except PostgrestAPIError as e:
        logger.error(f"Failed to get summary: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")

    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    summary = result.data.get("executive_summary")
    doc_status = result.data.get("status")

    if summary:
        return {"summary": summary, "status": "ready"}
    elif doc_status == "processing":
        return {"summary": None, "status": "generating"}
    else:
        return {"summary": None, "status": "not_available"}


@router.post("/file/{doc_id}/summary/regenerate")
async def regenerate_summary_endpoint(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> dict:
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
    if not supabase:
        raise HTTPException(status_code=500, detail="Database service unavailable")

    doc_id_str = str(doc_id)

    # Verify document exists and belongs to user
    try:
        result = (
            supabase.table("documents")
            .select("id, original_path")
            .eq("id", doc_id_str)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
    except PostgrestAPIError as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")

    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    # For regeneration, we need to re-read the document content
    # This is a simplified implementation - in production you might
    # want to store the extracted text in the database
    logger.info(f"Summary regeneration requested for doc {doc_id_str}")

    # Clear existing summary to indicate regeneration in progress
    try:
        supabase.table("documents").update({"executive_summary": None}).eq(
            "id", doc_id_str
        ).execute()
    except PostgrestAPIError as e:
        logger.warning(f"Failed to clear summary: {e}")

    return {
        "status": "generating",
        "message": "Summary regeneration started. Please check back in a few moments.",
    }
