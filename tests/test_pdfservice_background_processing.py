from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from pdfserviceMD.router import run_post_processing_tasks
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
            "pdfserviceMD.router.add_markdown_to_knowledge_base",
            new=AsyncMock(side_effect=RuntimeError("vector store exploded")),
        ),
        patch(
            "pdfserviceMD.router._process_document_images",
            new=AsyncMock(return_value=0),
        ) as process_images,
        patch(
            "pdfserviceMD.router._run_graph_extraction",
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
