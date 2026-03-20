"""Service layer for PDF upload pipeline."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Literal

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
    TranslatePdfResponse,
    UploadPdfResponse,
)

logger = logging.getLogger(__name__)

_STEP_LABELS = {
    "uploading": "上傳中",
    "ocr": "OCR 辨識中",
    "ocr_completed": "OCR 已完成",
    "translating": "翻譯中",
    "generating_pdf": "生成 PDF 中",
    "completed": "翻譯完成",
    "indexing": "建立索引中",
    "image_analysis": "分析圖片中",
    "graph_indexing": "建立知識圖譜中",
    "indexed": "全部完成",
    "failed": "處理失敗",
}

_EXTRACTED_MARKDOWN_FILENAME = "extracted.md"
_IMAGE_BLOCKS_FILENAME = "image_blocks.json"
_TRANSLATION_LOCKED_STEPS = {"translating", "generating_pdf"}
_TRANSLATABLE_STATUSES = {"ready", "indexing", "indexed"}


@dataclass
class UploadPipelineContext:
    """Internal context used to continue background tasks in router."""

    doc_id: str
    book_title: str
    user_folder: str
    response: UploadPdfResponse


def get_ocr_artifact_paths(*, user_folder: str) -> tuple[str, str]:
    """Returns the stored OCR artifact paths for one document folder."""
    return (
        os.path.normpath(os.path.join(user_folder, _EXTRACTED_MARKDOWN_FILENAME)),
        os.path.normpath(os.path.join(user_folder, _IMAGE_BLOCKS_FILENAME)),
    )


def save_ocr_artifacts(
    *,
    user_folder: str,
    markdown_text: str,
    image_blocks: list[str],
) -> tuple[str, str]:
    """Persists OCR markdown and image blocks for later manual translation."""
    extracted_path, image_blocks_path = get_ocr_artifact_paths(user_folder=user_folder)

    with open(extracted_path, "w", encoding="utf-8") as extracted_file:
        extracted_file.write(markdown_text)

    with open(image_blocks_path, "w", encoding="utf-8") as image_blocks_file:
        json.dump(image_blocks, image_blocks_file, ensure_ascii=False, indent=2)

    return extracted_path, image_blocks_path


def load_ocr_artifacts(*, user_folder: str) -> tuple[str, list[str]]:
    """Loads stored OCR markdown and image blocks from disk."""
    extracted_path, image_blocks_path = get_ocr_artifact_paths(user_folder=user_folder)
    if not os.path.exists(extracted_path) or not os.path.exists(image_blocks_path):
        raise FileNotFoundError("OCR artifacts not found")

    with open(extracted_path, "r", encoding="utf-8") as extracted_file:
        markdown_text = extracted_file.read()

    with open(image_blocks_path, "r", encoding="utf-8") as image_blocks_file:
        image_blocks = json.load(image_blocks_file)

    if not isinstance(image_blocks, list) or not all(
        isinstance(item, str) for item in image_blocks
    ):
        raise ValueError("image_blocks.json must contain a JSON array of strings")

    return markdown_text, image_blocks


def _has_file(path: str | None) -> bool:
    """Returns True when a stored file path exists on disk."""
    return bool(path) and os.path.exists(os.path.normpath(path))


def _build_download_url(doc_id: str, file_type: Literal["original", "translated"]) -> str:
    """Builds a typed PDF file URL."""
    return f"/pdfmd/file/{doc_id}?type={file_type}"


def _can_translate_document(row: dict) -> bool:
    """Derives whether manual translation should be enabled for the document."""
    status = row.get("status")
    if status not in _TRANSLATABLE_STATUSES:
        return False
    if _has_file(row.get("translated_path")):
        return False
    if not _has_file(row.get("original_path")):
        return False

    original_path = row.get("original_path")
    if not original_path:
        return False

    user_folder = os.path.dirname(os.path.normpath(original_path))
    extracted_path, image_blocks_path = get_ocr_artifact_paths(user_folder=user_folder)
    return os.path.exists(extracted_path) and os.path.exists(image_blocks_path)


async def run_upload_pipeline(
    *,
    file: UploadFile,
    user_id: str,
    base_upload_folder: str,
) -> UploadPipelineContext:
    """Runs upload + OCR artifact persistence before background indexing."""
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
        await run_in_threadpool(
            save_ocr_artifacts,
            user_folder=user_folder,
            markdown_text=processed_markdown_for_rag,
            image_blocks=image_blocks,
        )
        await update_document_status(doc_id=document_id, status="ready")
        await update_processing_step(doc_id=document_id, step="ocr_completed")

        total_time = time.perf_counter() - start_time
        logger.info("Upload pipeline finished in %.2fs for doc %s", total_time, document_id)

        response = UploadPdfResponse(
            doc_id=document_id,
            status="ready",
            message="OCR 已完成，背景索引進行中；如需翻譯請從歷史列表手動觸發。",
            pdf_available=True,
            pdf_download_url=_build_download_url(document_id, "original"),
            pdf_error=None,
            rag_status="processing_background",
        )
        return UploadPipelineContext(
            doc_id=document_id,
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
    documents = []
    for row in rows:
        documents.append(
            DocumentListItem(
                id=row["id"],
                file_name=row["file_name"],
                created_at=row["created_at"],
                status=row["status"],
                processing_step=row.get("processing_step"),
                has_original_pdf=_has_file(row.get("original_path")),
                has_translated_pdf=_has_file(row.get("translated_path")),
                can_translate=_can_translate_document(row),
            )
        )
    return DocumentListResponse(
        documents=documents,
        total=len(rows),
    )


async def get_document_processing_status(
    *, doc_id: str, user_id: str
) -> ProcessingStatusResponse:
    """Returns processing step and readiness flags for a document."""
    row = await get_document(
        doc_id=doc_id,
        user_id=user_id,
        columns="status, processing_step, original_path, translated_path",
    )
    if not row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Document not found",
            status_code=404,
        )

    step = row.get("processing_step") or "uploading"
    status = row.get("status")
    has_original_pdf = _has_file(row.get("original_path"))
    has_translated_pdf = _has_file(row.get("translated_path"))

    return ProcessingStatusResponse(
        step=step,
        step_label=_STEP_LABELS.get(step, step),
        is_pdf_ready=has_original_pdf or has_translated_pdf,
        is_fully_complete=step == "indexed",
    )


async def get_document_file_info(
    *,
    doc_id: str,
    user_id: str,
    file_type: Literal["original", "translated"] | None = None,
) -> tuple[str, str]:
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

    original_path = row.get("original_path")
    translated_path = row.get("translated_path")

    if file_type == "original":
        file_path = original_path
    elif file_type == "translated":
        file_path = translated_path
    else:
        file_path = translated_path or original_path

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

    if file_type == "translated" or (
        file_type is None and translated_path and normalized_path == os.path.normpath(translated_path)
    ):
        download_name = os.path.basename(normalized_path)
    else:
        download_name = row.get("file_name", "document.pdf")

    return normalized_path, download_name


async def update_indexing_processing_step(
    *,
    doc_id: str,
    user_id: str,
    step: str,
) -> None:
    """Updates indexing progress unless translation is actively running."""
    row = await get_document(
        doc_id=doc_id,
        user_id=user_id,
        columns="processing_step",
    )
    if not row:
        return

    current_step = row.get("processing_step")
    if current_step in _TRANSLATION_LOCKED_STEPS:
        logger.info(
            "Skip indexing step update for %s because translation step %s is active",
            doc_id,
            current_step,
        )
        return

    await update_processing_step(doc_id=doc_id, step=step)


async def finalize_indexing_status(*, doc_id: str, user_id: str) -> None:
    """Sets final document status after indexing without overwriting translation completion."""
    row = await get_document(
        doc_id=doc_id,
        user_id=user_id,
        columns="status, translated_path",
    )
    if not row:
        return

    if _has_file(row.get("translated_path")):
        logger.info("Preserving translated status for %s after indexing", doc_id)
        return

    await update_document_status(doc_id=doc_id, status="indexed")


async def translate_user_document(*, doc_id: str, user_id: str) -> TranslatePdfResponse:
    """Translates one OCR-complete document into a PDF on demand."""
    row = await get_document(
        doc_id=doc_id,
        user_id=user_id,
        columns="file_name, original_path, translated_path, status, processing_step",
    )
    if not row:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Document not found",
            status_code=404,
        )

    if _has_file(row.get("translated_path")):
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="Translated PDF already exists",
            status_code=409,
        )

    status = row.get("status")
    previous_step = row.get("processing_step")
    if status not in _TRANSLATABLE_STATUSES:
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="Document is not ready for translation",
            status_code=409,
        )

    original_path = row.get("original_path")
    if not original_path or not os.path.exists(os.path.normpath(original_path)):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Original PDF not found on disk",
            status_code=404,
        )

    user_folder = os.path.dirname(os.path.normpath(original_path))
    output_pdf_filename = f"translated_{row.get('file_name', 'document.pdf')}"
    output_pdf_path = os.path.normpath(os.path.join(user_folder, output_pdf_filename))

    try:
        markdown_text, image_blocks = await run_in_threadpool(
            load_ocr_artifacts,
            user_folder=user_folder,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="OCR artifacts are unavailable for translation",
            status_code=409,
        ) from exc

    await update_processing_step(doc_id=doc_id, step="translating")
    translate_result = await translate_text(markdown_text)
    final_md = replace_markdown(translate_result, image_blocks)

    await update_processing_step(doc_id=doc_id, step="generating_pdf")

    try:
        await run_in_threadpool(markdown_to_pdf, final_md, output_pdf_path, user_folder)
        if not os.path.exists(output_pdf_path):
            raise FileNotFoundError("Translated PDF not found on disk after generation")

        await update_document_status(
            doc_id=doc_id,
            status="completed",
            translated_path=output_pdf_path,
            error_message=None,
        )
        next_step = "completed" if status == "indexed" else (previous_step or "ocr_completed")
        await update_processing_step(doc_id=doc_id, step=next_step)
        return TranslatePdfResponse(
            doc_id=doc_id,
            status="completed",
            message="Translation completed successfully.",
            pdf_available=True,
            pdf_download_url=_build_download_url(doc_id, "translated"),
            pdf_error=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Manual translation failed for %s: %s", doc_id, exc, exc_info=True)
        await update_document_status(
            doc_id=doc_id,
            status="completed_with_pdf_error",
            error_message=str(exc),
        )
        await update_processing_step(
            doc_id=doc_id,
            step=previous_step or "ocr_completed",
        )
        return TranslatePdfResponse(
            doc_id=doc_id,
            status="completed_with_pdf_error",
            message="Translation finished but PDF generation failed.",
            pdf_available=False,
            pdf_download_url=None,
            pdf_error=str(exc),
        )


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
