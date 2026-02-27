"""Service layer for PDF upload pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

from core.errors import AppError, ErrorCode
from data_base.vector_store_manager import delete_document_from_knowledge_base
from pdfserviceMD.PDF_OCR_services import ocr_service_sync
from pdfserviceMD.Pandoc_md_to_pdf import MDmarkdown_to_pdf as markdown_to_pdf
from pdfserviceMD.ai_translate_md import translate_text
from pdfserviceMD.markdown_process import markdown_extact, replace_markdown
from pdfserviceMD.repository import (
    clear_document_summary,
    create_document_record,
    delete_document,
    get_document,
    list_documents,
    update_document_status,
    update_processing_step,
)
from pdfserviceMD.schemas import (
    DeleteDocumentResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentSummaryResponse,
    ProcessingStatusResponse,
    RegenerateSummaryResponse,
    UploadPdfResponse,
)

logger = logging.getLogger(__name__)

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


@dataclass
class UploadPipelineContext:
    """Internal context used to continue background tasks in router."""

    doc_id: str
    markdown_text: str
    book_title: str
    user_folder: str
    response: UploadPdfResponse


async def run_upload_pipeline(
    *,
    file: UploadFile,
    user_id: str,
    base_upload_folder: str,
) -> UploadPipelineContext:
    """Runs upload + OCR + translate + PDF generation before background indexing."""
    start_time = time.perf_counter()
    file_uuid = str(uuid.uuid4())
    user_folder = os.path.normpath(os.path.join(base_upload_folder, user_id, file_uuid))
    os.makedirs(user_folder, exist_ok=True)

    filename = os.path.basename(file.filename) if file.filename else "document.pdf"
    save_path = os.path.normpath(os.path.join(user_folder, filename))
    book_title = os.path.splitext(filename)[0]
    document_id = file_uuid

    try:
        file_content = await file.read()
        with open(save_path, "wb") as buffer:
            buffer.write(file_content)
        logger.info("File saved to: %s", save_path)

        await create_document_record(
            doc_id=document_id,
            user_id=user_id,
            file_name=filename,
            original_path=save_path,
        )

        await update_processing_step(doc_id=document_id, step="ocr")
        t1 = time.perf_counter()
        ocr_result = await run_in_threadpool(ocr_service_sync, save_path)
        t2 = time.perf_counter()
        logger.info("OCR completed in %.2fs", t2 - t1)

        processed_markdown_for_rag, image_blocks = markdown_extact(ocr_result)

        await update_processing_step(doc_id=document_id, step="translating")
        translate_result = await translate_text(processed_markdown_for_rag)
        t3 = time.perf_counter()
        logger.info("Translation completed in %.2fs", t3 - t2)

        final_md = replace_markdown(translate_result, image_blocks)
        await update_processing_step(doc_id=document_id, step="generating_pdf")

        output_pdf_filename = f"translated_{filename}"
        output_pdf_path = os.path.normpath(os.path.join(user_folder, output_pdf_filename))

        pdf_available = False
        pdf_error_message: str | None = None
        pdf_download_url: str | None = None
        final_status = "completed"

        try:
            await run_in_threadpool(markdown_to_pdf, final_md, output_pdf_path, user_folder)
            pdf_available = os.path.exists(output_pdf_path)
            if pdf_available:
                pdf_download_url = f"/pdfmd/file/{document_id}"
                await update_document_status(
                    doc_id=document_id,
                    status="completed",
                    translated_path=output_pdf_path,
                )
            else:
                final_status = "completed_with_pdf_error"
                pdf_error_message = "Translated PDF not found on disk after generation"
                await update_document_status(
                    doc_id=document_id,
                    status=final_status,
                    error_message=pdf_error_message,
                )

            await update_processing_step(doc_id=document_id, step="completed")
        except Exception as exc:  # noqa: BLE001
            logger.error("PDF generation failed: %s", exc, exc_info=True)
            final_status = "completed_with_pdf_error"
            pdf_error_message = "PDF generation failed"
            await update_document_status(
                doc_id=document_id,
                status=final_status,
                error_message=str(exc),
            )
            await update_processing_step(doc_id=document_id, step="completed")

        total_time = time.perf_counter() - start_time
        logger.info("Upload pipeline finished in %.2fs for doc %s", total_time, document_id)

        response = UploadPdfResponse(
            doc_id=document_id,
            status=final_status,
            message="Upload accepted. Background indexing started.",
            pdf_available=pdf_available,
            pdf_download_url=pdf_download_url,
            pdf_error=pdf_error_message,
            rag_status="processing_background",
        )
        return UploadPipelineContext(
            doc_id=document_id,
            markdown_text=processed_markdown_for_rag,
            book_title=book_title,
            user_folder=user_folder,
            response=response,
        )
    except FileNotFoundError as exc:
        await _mark_failed(document_id, "File processing error")
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="File processing error",
            status_code=500,
        ) from exc
    except AppError:
        await _mark_failed(document_id, "Document processing failed")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Processing failed: %s", exc, exc_info=True)
        await _mark_failed(document_id, str(exc))
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Document processing failed",
            status_code=500,
        ) from exc


async def _mark_failed(doc_id: str, error_message: str) -> None:
    """Marks document as failed if persistence is available."""
    if not doc_id:
        return

    try:
        await update_document_status(
            doc_id=doc_id,
            status="failed",
            error_message=error_message,
        )
        await update_processing_step(doc_id=doc_id, step="failed")
    except AppError:
        logger.warning("Failed to persist failure state for doc %s", doc_id)


async def list_user_documents(*, user_id: str) -> DocumentListResponse:
    """Lists all documents for one user."""
    rows = await list_documents(user_id=user_id, limit=50)
    return DocumentListResponse(
        documents=[DocumentListItem(**row) for row in rows],
        total=len(rows),
    )


async def get_document_processing_status(
    *, doc_id: str, user_id: str
) -> ProcessingStatusResponse:
    """Returns processing step and readiness flags for a document."""
    row = await get_document(
        doc_id=doc_id,
        user_id=user_id,
        columns="status, processing_step, translated_path",
    )
    if not row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Document not found",
            status_code=404,
        )

    step = row.get("processing_step") or "uploading"
    status = row.get("status")
    translated_path = row.get("translated_path")

    return ProcessingStatusResponse(
        step=step,
        step_label=_STEP_LABELS.get(step, step),
        is_pdf_ready=status == "completed" and translated_path is not None,
        is_fully_complete=step == "indexed",
    )


async def get_document_file_info(*, doc_id: str, user_id: str) -> tuple[str, str]:
    """Returns file path and download filename for a document."""
    row = await get_document(
        doc_id=doc_id,
        user_id=user_id,
        columns="file_name, translated_path, original_path",
    )
    if not row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Document not found",
            status_code=404,
        )

    file_path = row.get("translated_path") or row.get("original_path")
    if not file_path:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="File path not found in record",
            status_code=404,
        )

    normalized_path = os.path.normpath(file_path)
    if not os.path.exists(normalized_path):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="File not found on disk",
            status_code=404,
        )

    return normalized_path, row.get("file_name", "document.pdf")


async def delete_user_document(
    *, doc_id: str, user_id: str, base_upload_folder: str
) -> DeleteDocumentResponse:
    """Deletes document index data, files, and persistence record."""
    row = await get_document(doc_id=doc_id, user_id=user_id, columns="id")
    if not row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Document not found",
            status_code=404,
        )

    try:
        await run_in_threadpool(delete_document_from_knowledge_base, user_id, doc_id)
    except (RuntimeError, ValueError) as exc:
        logger.warning("RAG deletion failed (non-fatal): %s", exc)

    doc_folder = os.path.normpath(os.path.join(base_upload_folder, user_id, doc_id))
    if os.path.exists(doc_folder):
        try:
            await run_in_threadpool(shutil.rmtree, doc_folder)
        except OSError as exc:
            logger.error("Failed to delete folder %s: %s", doc_folder, exc, exc_info=True)

    await delete_document(doc_id=doc_id, user_id=user_id)
    return DeleteDocumentResponse(status="success", message="Document deleted successfully")


async def get_user_document_summary(
    *, doc_id: str, user_id: str
) -> DocumentSummaryResponse:
    """Gets summary status + content for a document."""
    row = await get_document(
        doc_id=doc_id,
        user_id=user_id,
        columns="executive_summary, status",
    )
    if not row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Document not found",
            status_code=404,
        )

    summary = row.get("executive_summary")
    doc_status = row.get("status")

    if summary:
        return DocumentSummaryResponse(summary=summary, status="ready")
    if doc_status == "processing":
        return DocumentSummaryResponse(summary=None, status="generating")
    return DocumentSummaryResponse(summary=None, status="not_available")


async def regenerate_document_summary(
    *, doc_id: str, user_id: str
) -> RegenerateSummaryResponse:
    """Clears current summary and signals frontend to poll regeneration status."""
    row = await get_document(doc_id=doc_id, user_id=user_id, columns="id")
    if not row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Document not found",
            status_code=404,
        )

    try:
        await clear_document_summary(doc_id=doc_id, user_id=user_id)
    except AppError as exc:
        logger.warning("Failed to clear summary for doc %s: %s", doc_id, exc)

    return RegenerateSummaryResponse(
        status="generating",
        message="Summary regeneration started. Please check back in a few moments.",
    )


async def safe_update_document_status(
    *,
    doc_id: str,
    status: str,
    translated_path: str | None = None,
    error_message: str | None = None,
) -> None:
    """Best-effort status update for background tasks."""
    try:
        await update_document_status(
            doc_id=doc_id,
            status=status,
            translated_path=translated_path,
            error_message=error_message,
        )
    except AppError as exc:
        logger.warning("Failed to update document status for %s: %s", doc_id, exc)


async def safe_update_processing_step(*, doc_id: str, step: str) -> None:
    """Best-effort processing step update for background tasks."""
    try:
        await update_processing_step(doc_id=doc_id, step=step)
    except AppError as exc:
        logger.warning("Failed to update processing step for %s: %s", doc_id, exc)
