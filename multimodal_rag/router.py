"""
Multimodal RAG Router

Provides API endpoints for extracting, summarizing, and indexing
visual elements and text from PDF documents.
"""

# Standard library
import logging
import os
import shutil
import uuid
from uuid import UUID

# Third-party
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

# Local application
from core.auth import get_current_user_id
from data_base.vector_store_manager import (
    index_extracted_document,
    delete_document_from_knowledge_base,
)
from core.summary_service import schedule_summary_generation
from .structure_analyzer import analyzer
from .image_summarizer import summarizer
from .schemas import ExtractedDocument

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
BASE_UPLOAD_FOLDER = "uploads"


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


@router.post("/extract", response_model=ExtractedDocument)
async def extract_from_pdf_endpoint(
    file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)
) -> ExtractedDocument:
    """
    Extracts text and visual elements from a PDF, summarizes images, and indexes content.

    Pipeline:
    1. Phase 1: Extract text chunks and visual elements (PaddleOCR)
    2. Phase 2: Summarize visual elements (Gemini)
    3. Phase 3: Index all content to FAISS

    Args:
        file: The uploaded PDF file.
        user_id: Authenticated user ID (injected via dependency).

    Returns:
        ExtractedDocument containing text chunks and visual elements with summaries.

    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors.
    """
    _validate_pdf_upload(file)

    doc_uuid = str(uuid.uuid4())
    doc_dir = os.path.normpath(os.path.join(BASE_UPLOAD_FOLDER, user_id, doc_uuid))
    os.makedirs(doc_dir, exist_ok=True)

    input_pdf_path = os.path.normpath(os.path.join(doc_dir, "original.pdf"))

    try:
        with open(input_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Phase 1: Extraction (CPU-bound)
        logger.info(f"[Phase 1] Starting extraction for doc {doc_uuid}")
        extracted_doc = await run_in_threadpool(
            analyzer.extract_from_pdf,
            pdf_path=input_pdf_path,
            user_id=user_id,
            doc_id=doc_uuid,
            output_base_dir=doc_dir,
        )

        # Phase 2: Summarization (async Gemini API)
        if extracted_doc.visual_elements:
            logger.info(
                f"[Phase 2] Summarizing {len(extracted_doc.visual_elements)} visual elements"
            )
            extracted_doc.visual_elements = await summarizer.summarize_elements(
                extracted_doc.visual_elements
            )

        # Phase 3: Indexing (isolated error handling)
        try:
            logger.info(f"[Phase 3] Indexing document for user {user_id}")
            await run_in_threadpool(
                index_extracted_document, user_id=user_id, doc=extracted_doc
            )

            # Trigger background summary generation (non-blocking)
            combined_text = "\n\n".join([c.content for c in extracted_doc.text_chunks])
            schedule_summary_generation(
                doc_id=doc_uuid,
                text_content=combined_text,
                user_id=user_id,
            )
            logger.info(
                f"[Phase 3] Background summary task scheduled for doc {doc_uuid}"
            )

        except ValueError as e:
            logger.warning(f"Indexing skipped (embedding model not ready): {e}")
        except Exception as e:
            logger.error(f"Indexing failed (non-fatal): {e}", exc_info=True)

        logger.info(f"Extraction complete for doc {doc_uuid}")
        return extracted_doc

    except HTTPException:
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir)
        raise

    except FileNotFoundError as e:
        logger.error(f"File operation failed: {e}")
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir)
        raise HTTPException(status_code=500, detail="File processing error")

    except Exception as e:
        logger.error(f"Extraction failed for doc {doc_uuid}: {e}", exc_info=True)
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    finally:
        file.file.close()


@router.delete("/file/{doc_id}")
async def delete_multimodal_document(
    doc_id: UUID, user_id: str = Depends(get_current_user_id)
) -> dict:
    """
    Deletes a multimodal document and all associated files.

    Cleanup steps:
    1. Remove from FAISS vector index (text chunks + visual summaries)
    2. Delete physical files (original PDF + cropped images)

    Args:
        doc_id: Document UUID.
        user_id: Authenticated user ID.

    Returns:
        Success status dict.
    """
    doc_id_str = str(doc_id)
    logger.info(f"Delete multimodal doc {doc_id_str} for user {user_id}")

    # 1. Delete from RAG index (non-fatal if fails)
    try:
        await run_in_threadpool(
            delete_document_from_knowledge_base, user_id, doc_id_str
        )
        logger.info(f"RAG index entries deleted for doc {doc_id_str}")
    except Exception as e:
        logger.warning(f"RAG deletion failed (non-fatal): {e}")

    # 2. Delete physical files
    doc_folder = os.path.normpath(os.path.join(BASE_UPLOAD_FOLDER, user_id, doc_id_str))
    if os.path.exists(doc_folder):
        try:
            shutil.rmtree(doc_folder)
            logger.info(f"Folder deleted: {doc_folder}")
        except OSError as e:
            logger.error(f"Failed to delete folder: {e}", exc_info=True)

    return {"status": "success", "message": "Multimodal document deleted successfully"}
