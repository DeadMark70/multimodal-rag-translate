"""Background indexing tasks for PDF OCR post-processing."""

from __future__ import annotations

import logging

from fastapi.concurrency import run_in_threadpool

from core.summary_service import schedule_summary_generation
from data_base.indexing_service import (
    DEFAULT_PRODUCTION_INDEXING_PROFILE,
    index_markdown_document,
    index_visual_summaries,
)
from data_base.vector_store_manager import delete_document_from_knowledge_base_async
from graph_rag.service import run_graph_extraction
from multimodal_rag.image_summarizer import summarizer as image_summarizer
from pdfserviceMD.image_processor import (
    create_visual_elements,
    extract_images_from_markdown,
)
from pdfserviceMD.service import (
    finalize_indexing_status,
    load_ocr_artifacts,
    record_background_processing_failure,
    safe_update_document_status,
    safe_update_processing_step,
    update_indexing_processing_step,
)

logger = logging.getLogger(__name__)


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
            await delete_document_from_knowledge_base_async(user_id, doc_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[Background] Existing vector cleanup failed for doc %s: %s", doc_id, e
            )
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
            indexing_profile=DEFAULT_PRODUCTION_INDEXING_PROFILE,
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
        await process_document_images(
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


async def retry_document_index_task(
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
        logger.warning(
            "[Background] OCR artifacts unavailable for retry-index doc %s: %s",
            doc_id,
            e,
        )
        await record_background_processing_failure(
            doc_id=doc_id,
            user_id=user_id,
            error_messages=[f"OCR artifacts unavailable: {e}"],
        )
        return

    await safe_update_document_status(
        doc_id=doc_id, status=current_status, error_message=None
    )
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

    await safe_update_document_status(
        doc_id=doc_id, status=current_status, error_message=None
    )
    await safe_update_processing_step(doc_id=doc_id, step="indexed")
    await finalize_indexing_status(doc_id=doc_id, user_id=user_id)
    logger.info("[Background] Retry-index complete for doc %s", doc_id)


async def process_document_images(
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
            if getattr(
                getattr(element, "type", None), "value", getattr(element, "type", None)
            )
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
            indexed_count = await index_visual_summaries(
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
