"""
PDF OCR & Translation Router

Provides API endpoints for PDF upload, OCR processing, translation, and file management.
"""

# Standard library
import logging
import os
import shutil
import time
import uuid

# Third-party
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

# Local application
from core.auth import get_current_user_id
from supabase_client import supabase
from data_base.vector_store_manager import add_markdown_to_knowledge_base, delete_document_from_knowledge_base
from pdfserviceMD.PDF_OCR_services import ocr_service_sync
from pdfserviceMD.ai_translate_md import translate_text
from pdfserviceMD.markdown_to_pdf import markdown_to_pdf
from pdfserviceMD.markdown_process import markdown_extact, replace_markdown

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
    except Exception as e:
        logger.error(f"Failed to update document status: {e}", exc_info=True)


@router.post("/ocr")
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
            except Exception as e:
                logger.error(f"DB insert failed: {e}", exc_info=True)

        # 3. OCR Processing (CPU-bound, run in threadpool)
        t1 = time.perf_counter()
        logger.info("Starting OCR processing...")
        ocr_result = await run_in_threadpool(ocr_service_sync, save_path)
        t2 = time.perf_counter()
        logger.info(f"OCR completed in {t2-t1:.2f}s")

        # 4. Extract markdown and image blocks
        processed_markdown_for_rag, image_blocks = markdown_extact(ocr_result)

        # 5. Add to RAG Knowledge Base (isolated error handling)
        try:
            await add_markdown_to_knowledge_base(
                user_id=user_id,
                markdown_text=processed_markdown_for_rag,
                pdf_title=book_title,
                doc_id=document_id,
                k_retriever=3,
            )
            logger.info(f"Document added to user {user_id} knowledge base")
        except Exception as e:
            logger.warning(f"RAG indexing failed (non-fatal): {e}")

        # 6. Translate (async Gemini API call)
        logger.info("Starting translation...")
        translate_result = await translate_text(processed_markdown_for_rag)
        t3 = time.perf_counter()
        logger.info(f"Translation completed in {t3-t2:.2f}s")

        # 7. Replace image placeholders
        final_md = replace_markdown(translate_result, image_blocks)

        # 8. Generate PDF (CPU-bound, run in threadpool)
        output_pdf_filename = f"translated_{filename}"
        output_pdf_path = os.path.normpath(os.path.join(user_folder, output_pdf_filename))

        await run_in_threadpool(markdown_to_pdf, final_md, output_pdf_path)
        t4 = time.perf_counter()
        logger.info(f"PDF generated in {t4-t3:.2f}s")

        # 9. Update DB status
        _update_document_status(document_id, "completed", translated_path=output_pdf_path)

        total_time = time.perf_counter() - start_time
        logger.info(f"Processing complete. Total time: {total_time:.2f}s")

        return FileResponse(
            path=output_pdf_path,
            filename=output_pdf_filename,
            media_type='application/pdf'
        )

    except HTTPException:
        _update_document_status(document_id, "failed", error_message="HTTP error during processing")
        raise

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        _update_document_status(document_id, "failed", error_message=str(e))
        raise HTTPException(status_code=500, detail="File processing error")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        _update_document_status(document_id, "failed", error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        await file.close()


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
    except Exception as e:
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
    except Exception as e:
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
        except Exception as e:
            logger.error(f"DB deletion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to delete database record")

    return {"status": "success", "message": "Document deleted successfully"}