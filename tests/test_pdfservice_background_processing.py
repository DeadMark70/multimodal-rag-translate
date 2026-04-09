from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from data_base.indexing_service import DEFAULT_PRODUCTION_INDEXING_PROFILE
from main import app
from pdfserviceMD.router import (
    DocumentImageProcessingError,
    _process_document_images,
    _retry_document_index_task,
    run_post_processing_tasks,
)
from pdfserviceMD.schemas import (
    DocumentListItem,
    DocumentListResponse,
    ProcessingStatusResponse,
)
from pdfserviceMD.service import record_background_processing_failure

TEST_USER_ID = "test-user-123"


@contextmanager
def _build_client():
    from core.auth import get_current_user_id

    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
    ):
        app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
        with TestClient(app) as client:
            yield client
    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_record_background_processing_failure_preserves_status() -> None:
    with (
        patch(
            "pdfserviceMD.service.get_document",
            new=AsyncMock(return_value={"status": "completed"}),
        ),
        patch(
            "pdfserviceMD.service.safe_update_document_status",
            new=AsyncMock(),
        ) as update_status,
        patch(
            "pdfserviceMD.service.safe_update_processing_step",
            new=AsyncMock(),
        ) as update_step,
    ):
        await record_background_processing_failure(
            doc_id="doc-1",
            user_id=TEST_USER_ID,
            error_messages=["RAG indexing failed: boom"],
        )

    update_status.assert_awaited_once_with(
        doc_id="doc-1",
        status="completed",
        error_message="RAG indexing failed: boom",
    )
    update_step.assert_awaited_once_with(doc_id="doc-1", step="index_failed")


@pytest.mark.asyncio
async def test_run_post_processing_tasks_records_background_failure_and_continues() -> None:
    with (
        patch(
            "pdfserviceMD.router.run_in_threadpool",
            new=AsyncMock(return_value=("markdown", [])),
        ),
        patch(
            "pdfserviceMD.router.update_indexing_processing_step",
            new=AsyncMock(),
        ) as update_step,
        patch(
            "pdfserviceMD.router.index_markdown_document",
            new=AsyncMock(side_effect=RuntimeError("vector store exploded")),
        ),
        patch(
            "pdfserviceMD.router._process_document_images",
            new=AsyncMock(return_value=0),
        ) as process_images,
        patch(
            "pdfserviceMD.router.run_graph_extraction",
            new=AsyncMock(),
        ) as run_graph,
        patch(
            "pdfserviceMD.router.schedule_summary_generation",
            new=Mock(),
        ) as schedule_summary,
        patch(
            "pdfserviceMD.router.record_background_processing_failure",
            new=AsyncMock(),
        ) as record_failure,
        patch(
            "pdfserviceMD.router.safe_update_processing_step",
            new=AsyncMock(),
        ) as safe_update_step,
        patch(
            "pdfserviceMD.router.finalize_indexing_status",
            new=AsyncMock(),
        ) as finalize_status,
    ):
        await run_post_processing_tasks(
            doc_id="doc-1",
            book_title="Demo",
            user_id=TEST_USER_ID,
            user_folder="uploads/test-user-123/doc-1",
        )

    assert update_step.await_count == 3
    process_images.assert_awaited_once()
    run_graph.assert_awaited_once()
    schedule_summary.assert_called_once()
    record_failure.assert_awaited_once()
    error_messages = record_failure.await_args.kwargs["error_messages"]
    assert any("RAG indexing failed: vector store exploded" in msg for msg in error_messages)
    safe_update_step.assert_not_called()
    finalize_status.assert_not_called()


@pytest.mark.asyncio
async def test_process_document_images_classifies_summary_stage_failures() -> None:
    figure_element = SimpleNamespace(type="figure", summary=None)
    failed_summary = SimpleNamespace(type="figure", summary="Error: timeout")

    with (
        patch(
            "pdfserviceMD.router.extract_images_from_markdown",
            return_value=["page1.png"],
        ),
        patch(
            "pdfserviceMD.router.create_visual_elements",
            return_value=[figure_element],
        ),
        patch(
            "pdfserviceMD.router.image_summarizer.summarize_elements",
            new=AsyncMock(return_value=[failed_summary]),
        ),
    ):
        with pytest.raises(DocumentImageProcessingError) as exc_info:
            await _process_document_images(
                user_id=TEST_USER_ID,
                doc_id="doc-1",
                markdown_text="markdown",
                user_folder="uploads/test-user-123/doc-1",
                book_title="Demo",
            )

    assert exc_info.value.stage == "image_summary_api_failed"
    assert "No figure summaries were generated" in str(exc_info.value)


@pytest.mark.asyncio
async def test_process_document_images_classifies_visual_index_failures() -> None:
    figure_element = SimpleNamespace(type="figure", summary=None)
    successful_summary = SimpleNamespace(type="figure", summary="Figure summary")

    with (
        patch(
            "pdfserviceMD.router.extract_images_from_markdown",
            return_value=["page1.png"],
        ),
        patch(
            "pdfserviceMD.router.create_visual_elements",
            return_value=[figure_element],
        ),
        patch(
            "pdfserviceMD.router.image_summarizer.summarize_elements",
            new=AsyncMock(return_value=[successful_summary]),
        ),
        patch(
            "pdfserviceMD.router.index_visual_summaries",
            new=AsyncMock(side_effect=RuntimeError("Embedding transport error after 4 attempts")),
        ),
    ):
        with pytest.raises(DocumentImageProcessingError) as exc_info:
            await _process_document_images(
                user_id=TEST_USER_ID,
                doc_id="doc-1",
                markdown_text="markdown",
                user_folder="uploads/test-user-123/doc-1",
                book_title="Demo",
            )

    assert exc_info.value.stage == "visual_summary_index_failed"
    assert "vector indexing failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_post_processing_tasks_records_visual_index_failure_stage() -> None:
    with (
        patch(
            "pdfserviceMD.router.run_in_threadpool",
            new=AsyncMock(return_value=("markdown", [])),
        ),
        patch(
            "pdfserviceMD.router.update_indexing_processing_step",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.router.index_markdown_document",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.router._process_document_images",
            new=AsyncMock(
                side_effect=DocumentImageProcessingError(
                    stage="visual_summary_index_failed",
                    detail="Generated 1 image summaries, but vector indexing failed: boom",
                )
            ),
        ),
        patch(
            "pdfserviceMD.router.run_graph_extraction",
            new=AsyncMock(return_value=SimpleNamespace(status="indexed", last_error=None)),
        ),
        patch(
            "pdfserviceMD.router.schedule_summary_generation",
            new=Mock(),
        ),
        patch(
            "pdfserviceMD.router.record_background_processing_failure",
            new=AsyncMock(),
        ) as record_failure,
        patch(
            "pdfserviceMD.router.safe_update_processing_step",
            new=AsyncMock(),
        ) as safe_update_step,
        patch(
            "pdfserviceMD.router.finalize_indexing_status",
            new=AsyncMock(),
        ) as finalize_status,
    ):
        await run_post_processing_tasks(
            doc_id="doc-1",
            book_title="Demo",
            user_id=TEST_USER_ID,
            user_folder="uploads/test-user-123/doc-1",
        )

    error_messages = record_failure.await_args.kwargs["error_messages"]
    assert "Visual summary indexing failed:" in error_messages[0]
    assert not any("Image analysis failed" in message for message in error_messages)
    safe_update_step.assert_not_called()
    finalize_status.assert_not_called()


@pytest.mark.asyncio
async def test_run_post_processing_tasks_uses_profiled_production_indexing() -> None:
    with (
        patch(
            "pdfserviceMD.router.run_in_threadpool",
            new=AsyncMock(return_value=("markdown", [])),
        ),
        patch(
            "pdfserviceMD.router.update_indexing_processing_step",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.router.index_markdown_document",
            new=AsyncMock(),
        ) as index_markdown,
        patch(
            "pdfserviceMD.router._process_document_images",
            new=AsyncMock(return_value=0),
        ),
        patch(
            "pdfserviceMD.router.run_graph_extraction",
            new=AsyncMock(return_value=SimpleNamespace(status="indexed", last_error=None)),
        ),
        patch(
            "pdfserviceMD.router.schedule_summary_generation",
            new=Mock(),
        ),
        patch(
            "pdfserviceMD.router.record_background_processing_failure",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.router.safe_update_processing_step",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.router.finalize_indexing_status",
            new=AsyncMock(),
        ),
    ):
        await run_post_processing_tasks(
            doc_id="doc-1",
            book_title="Demo",
            user_id=TEST_USER_ID,
            user_folder="uploads/test-user-123/doc-1",
        )

    assert index_markdown.await_args.kwargs["indexing_profile"] == DEFAULT_PRODUCTION_INDEXING_PROFILE


def test_list_endpoint_includes_error_message_field() -> None:
    payload = DocumentListResponse(
        documents=[
            DocumentListItem(
                id="doc-1",
                file_name="demo.pdf",
                created_at="2026-03-21T00:00:00Z",
                status="ready",
                processing_step="index_failed",
                has_original_pdf=True,
                has_translated_pdf=False,
                can_translate=True,
                error_message="Graph indexing failed: quota exceeded",
            )
        ],
        total=1,
    )

    with _build_client() as client, patch(
        "pdfserviceMD.router.list_user_documents",
        new=AsyncMock(return_value=payload),
    ):
        response = client.get("/pdfmd/list")

    assert response.status_code == 200
    assert response.json()["documents"][0]["error_message"] == "Graph indexing failed: quota exceeded"


def test_status_endpoint_includes_error_message_field() -> None:
    payload = ProcessingStatusResponse(
        step="index_failed",
        step_label="索引失敗",
        is_pdf_ready=True,
        is_fully_complete=False,
        error_message="RAG indexing failed: vector store exploded",
    )

    with _build_client() as client, patch(
        "pdfserviceMD.router.get_document_processing_status",
        new=AsyncMock(return_value=payload),
    ):
        response = client.get("/pdfmd/file/00000000-0000-0000-0000-000000000001/status")

    assert response.status_code == 200
    assert response.json()["step"] == "index_failed"
    assert response.json()["error_message"] == "RAG indexing failed: vector store exploded"


@pytest.mark.asyncio
async def test_retry_document_index_task_clears_vectors_and_skips_graph_pipeline() -> None:
    with (
        patch(
            "pdfserviceMD.router.run_in_threadpool",
            new=AsyncMock(return_value=("markdown", [])),
        ),
        patch("pdfserviceMD.router.update_indexing_processing_step", new=AsyncMock()),
        patch("pdfserviceMD.router.index_markdown_document", new=AsyncMock()),
        patch("pdfserviceMD.router._process_document_images", new=AsyncMock(return_value=1)),
        patch(
            "pdfserviceMD.router.delete_document_from_knowledge_base_async",
            new=AsyncMock(return_value=True),
        ) as delete_vectors,
        patch("pdfserviceMD.router.record_background_processing_failure", new=AsyncMock()) as record_failure,
        patch("pdfserviceMD.router.safe_update_document_status", new=AsyncMock()) as safe_update_status,
        patch("pdfserviceMD.router.safe_update_processing_step", new=AsyncMock()) as safe_update_step,
        patch("pdfserviceMD.router.finalize_indexing_status", new=AsyncMock()) as finalize_status,
    ):
        await _retry_document_index_task(
            doc_id="doc-1",
            book_title="Demo",
            current_status="completed",
            user_id=TEST_USER_ID,
            user_folder="uploads/test-user-123/doc-1",
        )

    delete_vectors.assert_awaited_once_with(TEST_USER_ID, "doc-1")
    record_failure.assert_not_awaited()
    safe_update_status.assert_any_await(doc_id="doc-1", status="completed", error_message=None)
    safe_update_step.assert_awaited_once_with(doc_id="doc-1", step="indexed")
    finalize_status.assert_awaited_once_with(doc_id="doc-1", user_id=TEST_USER_ID)


def test_retry_index_endpoint_starts_background_task_for_index_failed_doc() -> None:
    payload = {
        "id": "doc-1",
        "file_name": "demo.pdf",
        "original_path": "uploads/test-user-123/doc-1/demo.pdf",
        "processing_step": "index_failed",
    }

    with (
        _build_client() as client,
        patch("pdfserviceMD.router.get_document", new=AsyncMock(return_value=payload)),
        patch("pdfserviceMD.router.os.path.exists", return_value=True),
        patch("pdfserviceMD.router.GraphStore") as graph_store_cls,
        patch("pdfserviceMD.router._retry_document_index_task", new=AsyncMock()) as retry_task,
    ):
        graph_store_cls.return_value.active_job_state = None
        response = client.post("/pdfmd/file/00000000-0000-0000-0000-000000000001/retry-index")

    assert response.status_code == 200
    assert response.json()["status"] == "started"
    retry_task.assert_awaited_once()


def test_retry_index_endpoint_rejects_non_index_failed_doc() -> None:
    payload = {
        "id": "doc-1",
        "file_name": "demo.pdf",
        "original_path": "uploads/test-user-123/doc-1/demo.pdf",
        "processing_step": "indexed",
    }

    with (
        _build_client() as client,
        patch("pdfserviceMD.router.get_document", new=AsyncMock(return_value=payload)),
    ):
        response = client.post("/pdfmd/file/00000000-0000-0000-0000-000000000001/retry-index")

    assert response.status_code == 409
