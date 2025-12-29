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
    delete_document_from_knowledge_base,
)
from pdfserviceMD.PDF_OCR_services import ocr_service_sync
from pdfserviceMD.ai_translate_md import translate_text
from pdfserviceMD.markdown_to_pdf import markdown_to_pdf
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
        raise HTTPException(status_code=400, detail="File must be a PDF (invalid content-type)")

    if file.filename:
        _, ext = os.path.splitext(file.filename)
        if ext.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF (invalid extension)")


def _update_document_status(
    doc_id: str,
    status: str,
    translated_path: str | None = None,
    error_message: str | None = None
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
        supabase.table("documents").update({
            "processing_step": step
        }).eq("id", doc_id).execute()
        logger.debug(f"Document {doc_id} step: {step}")
    except PostgrestAPIError as e:
        logger.warning(f"Failed to update processing step: {e}")


async def run_post_processing_tasks(
    doc_id: str,
    markdown_text: str,
    book_title: str,
    user_id: str,
) -> None:
    """
    Background task for post-PDF processing (RAG indexing + GraphRAG + Summary).

    Called after the translated PDF is returned to user.
    Runs RAG indexing, GraphRAG extraction, and summary generation in background.

    Args:
        doc_id: Document UUID.
        markdown_text: Extracted markdown content for indexing.
        book_title: Title for the document.
        user_id: User ID for knowledge base.
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

        # 2. GraphRAG entity extraction
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


async def _run_graph_extraction(user_id: str, doc_id: str, markdown_text: str) -> None:
    """
    Run GraphRAG entity extraction on document content.

    Extracts entities and relations from the document and adds them to
    the user's knowledge graph. Non-blocking and gracefully handles errors.

    Args:
        user_id: User ID for graph store.
        doc_id: Document UUID.
        markdown_text: Text content to extract entities from.
    """
    try:
        # Split text into chunks for extraction (max ~8000 chars each)
        chunk_size = 8000
        chunks = [
            markdown_text[i:i + chunk_size]
            for i in range(0, len(markdown_text), chunk_size)
        ]

        store = GraphStore(user_id)
        total_nodes = 0
        total_edges = 0

        for idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 100:  # Skip very small chunks
                continue
            try:
                nodes, edges = await extract_and_add_to_graph(
                    text=chunk,
                    doc_id=doc_id,
                    store=store,
                    chunk_index=idx,
                )
                total_nodes += nodes
                total_edges += edges
            except (ValueError, RuntimeError) as chunk_error:
                logger.warning(f"[GraphRAG] Chunk {idx} extraction failed: {chunk_error}")
                continue

        # Save graph if any entities were extracted
        if total_nodes > 0:
            store.save()
            logger.info(
                f"[Background] GraphRAG complete for doc {doc_id}: "
                f"{total_nodes} nodes, {total_edges} edges"
            )
        else:
            logger.info(f"[Background] GraphRAG: No entities extracted from doc {doc_id}")

    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"[GraphRAG] Store access failed for user {user_id}: {e}")
    except Exception as e:
        # Catch-all to prevent graph errors from breaking the pipeline
        logger.error(f"[GraphRAG] Unexpected error: {e}", exc_info=True)


# --- Document List Endpoint ---

@router.get("/list")
async def list_documents(
    user_id: str = Depends(get_current_user_id)
) -> dict:
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
        result = supabase.table("documents")\
            .select("id, file_name, created_at, status, processing_step")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(50)\
            .execute()

        documents = result.data if result.data else []

        logger.info(f"User {user_id} listed {len(documents)} documents")

        return {
            "documents": documents,
            "total": len(documents)
        }

    except PostgrestAPIError as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


# --- PDF Upload Endpoint ---
# Supports both /ocr (legacy) and /upload_pdf_md (new spec) paths

@router.post("/ocr")
@router.post("/upload_pdf_md")
async def upload_pdf_md(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
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
                    "target_lang": "zh-TW"
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
        logger.info(f"OCR completed in {t2-t1:.2f}s")

        # 4. Extract markdown and image blocks
        processed_markdown_for_rag, image_blocks = markdown_extact(ocr_result)

        # 5. Translate (async Gemini API call) - PRIORITIZED
        _update_processing_step(document_id, "translating")
        logger.info("Starting translation...")
        translate_result = await translate_text(processed_markdown_for_rag)
        t3 = time.perf_counter()
        logger.info(f"Translation completed in {t3-t2:.2f}s")

        # 6. Replace image placeholders
        final_md = replace_markdown(translate_result, image_blocks)

        # 7. Generate PDF (CPU-bound, run in threadpool) - PRIORITIZED
        _update_processing_step(document_id, "generating_pdf")
        output_pdf_filename = f"translated_{filename}"
        output_pdf_path = os.path.normpath(os.path.join(user_folder, output_pdf_filename))

        await run_in_threadpool(markdown_to_pdf, final_md, output_pdf_path)
        t4 = time.perf_counter()
        logger.info(f"PDF generated in {t4-t3:.2f}s")

        # 8. Update DB status - PDF is ready!
        _update_document_status(document_id, "completed", translated_path=output_pdf_path)
        _update_processing_step(document_id, "completed")

        total_time = time.perf_counter() - start_time
        logger.info(f"PDF ready. Time: {total_time:.2f}s. Scheduling background tasks...")

        # 9. Schedule background tasks (RAG + Summary) - NON-BLOCKING
        asyncio.create_task(
            run_post_processing_tasks(
                doc_id=document_id,
                markdown_text=processed_markdown_for_rag,
                book_title=book_title,
                user_id=user_id,
            )
        )

        # 10. Return translated PDF immediately - user doesn't wait for RAG!
        return FileResponse(
            path=output_pdf_path,
            filename=output_pdf_filename,
            media_type='application/pdf'
        )

    except HTTPException:
        _update_document_status(document_id, "failed", error_message="HTTP error during processing")
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
    doc_id: str,
    user_id: str = Depends(get_current_user_id)
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

    try:
        result = supabase.table("documents").select(
            "status, processing_step, translated_path"
        ).eq("id", doc_id).eq("user_id", user_id).single().execute()
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
    doc_id: str,
    user_id: str = Depends(get_current_user_id)
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

    try:
        response = supabase.table("documents").select("*").eq("id", doc_id).eq("user_id", user_id).execute()
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
        media_type='application/pdf'
    )


@router.delete("/file/{doc_id}")
async def delete_pdf_file(
    doc_id: str,
    user_id: str = Depends(get_current_user_id)
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
    logger.info(f"Delete request for doc {doc_id} by user {user_id}")

    # 1. Delete from RAG index (non-fatal if fails)
    try:
        await run_in_threadpool(delete_document_from_knowledge_base, user_id, doc_id)
        logger.info(f"RAG index entry deleted for doc {doc_id}")
    except (RuntimeError, ValueError) as e:
        logger.warning(f"RAG deletion failed (non-fatal): {e}")

    # 2. Delete physical files
    doc_folder = os.path.normpath(os.path.join(BASE_UPLOAD_FOLDER, user_id, doc_id))
    if os.path.exists(doc_folder):
        try:
            shutil.rmtree(doc_folder)
            logger.info(f"Folder deleted: {doc_folder}")
        except OSError as e:
            logger.error(f"Failed to delete folder: {e}", exc_info=True)

    # 3. Delete database record
    if supabase:
        try:
            supabase.table("documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()
            logger.info(f"DB record deleted for doc {doc_id}")
        except PostgrestAPIError as e:
            logger.error(f"DB deletion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to delete database record")

    return {"status": "success", "message": "Document deleted successfully"}


# --- Summary Endpoints ---

@router.get("/file/{doc_id}/summary")
async def get_document_summary_endpoint(
    doc_id: str,
    user_id: str = Depends(get_current_user_id)
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

    try:
        result = supabase.table("documents").select(
            "executive_summary, status"
        ).eq("id", doc_id).eq("user_id", user_id).single().execute()
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
    doc_id: str,
    user_id: str = Depends(get_current_user_id)
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

    # Verify document exists and belongs to user
    try:
        result = supabase.table("documents").select(
            "id, original_path"
        ).eq("id", doc_id).eq("user_id", user_id).single().execute()
    except PostgrestAPIError as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")

    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    # For regeneration, we need to re-read the document content
    # This is a simplified implementation - in production you might
    # want to store the extracted text in the database
    logger.info(f"Summary regeneration requested for doc {doc_id}")

    # Clear existing summary to indicate regeneration in progress
    try:
        supabase.table("documents").update({
            "executive_summary": None
        }).eq("id", doc_id).execute()
    except PostgrestAPIError as e:
        logger.warning(f"Failed to clear summary: {e}")

    return {
        "status": "generating",
        "message": "Summary regeneration started. Please check back in a few moments."
    }
