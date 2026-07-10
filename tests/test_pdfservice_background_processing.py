from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from core.errors import AppError, ErrorCode
from data_base.indexing_service import DEFAULT_PRODUCTION_INDEXING_PROFILE
from graph_rag.schemas import GraphAssetLink
from main import app
from pdfserviceMD.indexing_tasks import (
    DocumentImageProcessingError,
    process_document_images,
    retry_document_index_task,
    run_post_processing_tasks,
)
from pdfserviceMD.schemas import (
    DocumentListItem,
    DocumentListResponse,
    ProcessingStatusResponse,
)
from pdfserviceMD.service import (
    RetryIndexContext,
    prepare_retry_index_context,
    record_background_processing_failure,
)

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
async def test_run_post_processing_tasks_records_background_failure_and_continues() -> (
    None
):
    with (
        patch(
            "pdfserviceMD.indexing_tasks.run_in_threadpool",
            new=AsyncMock(return_value=("markdown", [])),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.update_indexing_processing_step",
            new=AsyncMock(),
        ) as update_step,
        patch(
            "pdfserviceMD.indexing_tasks.index_markdown_document",
            new=AsyncMock(side_effect=RuntimeError("vector store exploded")),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.process_document_images",
            new=AsyncMock(return_value=0),
        ) as process_images,
        patch(
            "pdfserviceMD.indexing_tasks.run_graph_extraction",
            new=AsyncMock(),
        ) as run_graph,
        patch(
            "pdfserviceMD.indexing_tasks.schedule_summary_generation",
            new=Mock(),
        ) as schedule_summary,
        patch(
            "pdfserviceMD.indexing_tasks.record_background_processing_failure",
            new=AsyncMock(),
        ) as record_failure,
        patch(
            "pdfserviceMD.indexing_tasks.safe_update_processing_step",
            new=AsyncMock(),
        ) as safe_update_step,
        patch(
            "pdfserviceMD.indexing_tasks.finalize_indexing_status",
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
    assert any(
        "RAG indexing failed: vector store exploded" in msg for msg in error_messages
    )
    safe_update_step.assert_not_called()
    finalize_status.assert_not_called()


@pytest.mark.asyncio
async def test_process_document_images_classifies_summary_stage_failures() -> None:
    figure_element = SimpleNamespace(type="figure", summary=None)
    failed_summary = SimpleNamespace(type="figure", summary="Error: timeout")

    with (
        patch(
            "pdfserviceMD.indexing_tasks.extract_images_from_markdown",
            return_value=["page1.png"],
        ),
        patch(
            "pdfserviceMD.indexing_tasks.create_visual_elements",
            return_value=[figure_element],
        ),
        patch(
            "pdfserviceMD.indexing_tasks.image_summarizer.summarize_elements",
            new=AsyncMock(return_value=[failed_summary]),
        ),
    ):
        with pytest.raises(DocumentImageProcessingError) as exc_info:
            await process_document_images(
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
            "pdfserviceMD.indexing_tasks.extract_images_from_markdown",
            return_value=["page1.png"],
        ),
        patch(
            "pdfserviceMD.indexing_tasks.create_visual_elements",
            return_value=[figure_element],
        ),
        patch(
            "pdfserviceMD.indexing_tasks.image_summarizer.summarize_elements",
            new=AsyncMock(return_value=[successful_summary]),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.index_visual_summaries",
            new=AsyncMock(
                side_effect=RuntimeError("Embedding transport error after 4 attempts")
            ),
        ),
    ):
        with pytest.raises(DocumentImageProcessingError) as exc_info:
            await process_document_images(
                user_id=TEST_USER_ID,
                doc_id="doc-1",
                markdown_text="markdown",
                user_folder="uploads/test-user-123/doc-1",
                book_title="Demo",
            )

    assert exc_info.value.stage == "visual_summary_index_failed"
    assert "vector indexing failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_process_document_images_records_only_successfully_indexed_assets() -> None:
    figure_element = SimpleNamespace(
        id="visual-1",
        type="figure",
        summary=None,
        page_number=3,
        bbox=[0, 0, 10, 10],
        context_text="Figure 2 shows the architecture.",
        figure_reference="Figure 2",
    )
    successful_summary = SimpleNamespace(**figure_element.__dict__)
    successful_summary.summary = "A source image summary."
    asset_link = GraphAssetLink(
        asset_id="figure-1",
        doc_id="doc-1",
        page=3,
        asset_type="figure",
        text_or_markdown="A source image summary.",
        asset_text_hash="hash",
        asset_parse_status="parsed",
        source_chunk_id="graph:asset:figure-1",
    )
    graph_store = Mock()

    with (
        patch(
            "pdfserviceMD.indexing_tasks.extract_images_from_markdown",
            return_value=["page1.png"],
        ),
        patch(
            "pdfserviceMD.indexing_tasks.create_visual_elements",
            return_value=[figure_element],
        ),
        patch(
            "pdfserviceMD.indexing_tasks.image_summarizer.summarize_elements",
            new=AsyncMock(return_value=[successful_summary]),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.build_visual_asset_links",
            return_value=[asset_link],
        ) as build_links,
        patch(
            "pdfserviceMD.indexing_tasks.index_visual_summaries",
            new=AsyncMock(return_value=1),
        ),
        patch("pdfserviceMD.indexing_tasks.GraphStore", return_value=graph_store),
    ):
        indexed_count = await process_document_images(
            user_id=TEST_USER_ID,
            doc_id="doc-1",
            markdown_text="markdown",
            user_folder="uploads/test-user-123/doc-1",
            book_title="Demo",
        )

    assert indexed_count == 1
    build_links.assert_called_once_with(doc_id="doc-1", elements=[successful_summary])
    graph_store.record_asset_link.assert_called_once_with(asset_link)
    graph_store.save_sidecars.assert_called_once()


@pytest.mark.asyncio
async def test_run_post_processing_tasks_records_visual_index_failure_stage() -> None:
    with (
        patch(
            "pdfserviceMD.indexing_tasks.run_in_threadpool",
            new=AsyncMock(return_value=("markdown", [])),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.update_indexing_processing_step",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.index_markdown_document",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.process_document_images",
            new=AsyncMock(
                side_effect=DocumentImageProcessingError(
                    stage="visual_summary_index_failed",
                    detail="Generated 1 image summaries, but vector indexing failed: boom",
                )
            ),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.run_graph_extraction",
            new=AsyncMock(
                return_value=SimpleNamespace(status="indexed", last_error=None)
            ),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.schedule_summary_generation",
            new=Mock(),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.record_background_processing_failure",
            new=AsyncMock(),
        ) as record_failure,
        patch(
            "pdfserviceMD.indexing_tasks.safe_update_processing_step",
            new=AsyncMock(),
        ) as safe_update_step,
        patch(
            "pdfserviceMD.indexing_tasks.finalize_indexing_status",
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
            "pdfserviceMD.indexing_tasks.run_in_threadpool",
            new=AsyncMock(return_value=("markdown", [])),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.update_indexing_processing_step",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.index_markdown_document",
            new=AsyncMock(),
        ) as index_markdown,
        patch(
            "pdfserviceMD.indexing_tasks.process_document_images",
            new=AsyncMock(return_value=0),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.run_graph_extraction",
            new=AsyncMock(
                return_value=SimpleNamespace(status="indexed", last_error=None)
            ),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.schedule_summary_generation",
            new=Mock(),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.record_background_processing_failure",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.safe_update_processing_step",
            new=AsyncMock(),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.finalize_indexing_status",
            new=AsyncMock(),
        ),
    ):
        await run_post_processing_tasks(
            doc_id="doc-1",
            book_title="Demo",
            user_id=TEST_USER_ID,
            user_folder="uploads/test-user-123/doc-1",
        )

    assert (
        index_markdown.await_args.kwargs["indexing_profile"]
        == DEFAULT_PRODUCTION_INDEXING_PROFILE
    )


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

    with (
        _build_client() as client,
        patch(
            "pdfserviceMD.router.list_user_documents",
            new=AsyncMock(return_value=payload),
        ),
    ):
        response = client.get("/pdfmd/list")

    assert response.status_code == 200
    assert (
        response.json()["documents"][0]["error_message"]
        == "Graph indexing failed: quota exceeded"
    )


def test_status_endpoint_includes_error_message_field() -> None:
    payload = ProcessingStatusResponse(
        step="index_failed",
        step_label="索引失敗",
        is_pdf_ready=True,
        is_fully_complete=False,
        error_message="RAG indexing failed: vector store exploded",
    )

    with (
        _build_client() as client,
        patch(
            "pdfserviceMD.router.get_document_processing_status",
            new=AsyncMock(return_value=payload),
        ),
    ):
        response = client.get("/pdfmd/file/00000000-0000-0000-0000-000000000001/status")

    assert response.status_code == 200
    assert response.json()["step"] == "index_failed"
    assert (
        response.json()["error_message"] == "RAG indexing failed: vector store exploded"
    )


@pytest.mark.asyncio
async def test_retry_document_index_task_clears_vectors_and_skips_graph_pipeline() -> (
    None
):
    with (
        patch(
            "pdfserviceMD.indexing_tasks.run_in_threadpool",
            new=AsyncMock(return_value=("markdown", [])),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.update_indexing_processing_step",
            new=AsyncMock(),
        ),
        patch("pdfserviceMD.indexing_tasks.index_markdown_document", new=AsyncMock()),
        patch(
            "pdfserviceMD.indexing_tasks.process_document_images",
            new=AsyncMock(return_value=1),
        ),
        patch(
            "pdfserviceMD.indexing_tasks.delete_document_from_knowledge_base_async",
            new=AsyncMock(return_value=True),
        ) as delete_vectors,
        patch(
            "pdfserviceMD.indexing_tasks.record_background_processing_failure",
            new=AsyncMock(),
        ) as record_failure,
        patch(
            "pdfserviceMD.indexing_tasks.safe_update_document_status", new=AsyncMock()
        ) as safe_update_status,
        patch(
            "pdfserviceMD.indexing_tasks.safe_update_processing_step", new=AsyncMock()
        ) as safe_update_step,
        patch(
            "pdfserviceMD.indexing_tasks.finalize_indexing_status", new=AsyncMock()
        ) as finalize_status,
    ):
        await retry_document_index_task(
            doc_id="doc-1",
            book_title="Demo",
            current_status="completed",
            user_id=TEST_USER_ID,
            user_folder="uploads/test-user-123/doc-1",
        )

    delete_vectors.assert_awaited_once_with(TEST_USER_ID, "doc-1")
    record_failure.assert_not_awaited()
    safe_update_status.assert_any_await(
        doc_id="doc-1", status="completed", error_message=None
    )
    safe_update_step.assert_awaited_once_with(doc_id="doc-1", step="indexed")
    finalize_status.assert_awaited_once_with(doc_id="doc-1", user_id=TEST_USER_ID)


def test_retry_index_endpoint_starts_background_task_for_index_failed_doc() -> None:
    context = RetryIndexContext(
        doc_id="00000000-0000-0000-0000-000000000001",
        book_title="demo",
        current_status="completed",
        user_id=TEST_USER_ID,
        user_folder="uploads/test-user-123/doc-1",
    )

    with (
        _build_client() as client,
        patch(
            "pdfserviceMD.router.prepare_retry_index_context",
            new=AsyncMock(return_value=context),
        ) as prepare_context,
        patch(
            "pdfserviceMD.router._retry_document_index_task", new=AsyncMock()
        ) as retry_task,
    ):
        response = client.post(
            "/pdfmd/file/00000000-0000-0000-0000-000000000001/retry-index"
        )

    assert response.status_code == 200
    assert response.json()["status"] == "started"
    prepare_context.assert_awaited_once_with(
        doc_id="00000000-0000-0000-0000-000000000001",
        user_id=TEST_USER_ID,
    )
    retry_task.assert_awaited_once_with(
        doc_id="00000000-0000-0000-0000-000000000001",
        book_title="demo",
        current_status="completed",
        user_id=TEST_USER_ID,
        user_folder="uploads/test-user-123/doc-1",
    )


def test_retry_index_endpoint_rejects_non_index_failed_doc() -> None:
    with (
        _build_client() as client,
        patch(
            "pdfserviceMD.router.prepare_retry_index_context",
            new=AsyncMock(
                side_effect=AppError(
                    code=ErrorCode.BAD_REQUEST,
                    message="Document is not eligible for retry-index",
                    status_code=409,
                )
            ),
        ),
    ):
        response = client.post(
            "/pdfmd/file/00000000-0000-0000-0000-000000000001/retry-index"
        )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_prepare_retry_index_context_validates_and_derives_payload(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "doc-1"
    artifact_dir.mkdir()
    (artifact_dir / "extracted.md").write_text("markdown", encoding="utf-8")

    with (
        patch(
            "pdfserviceMD.service.get_document",
            new=AsyncMock(
                return_value={
                    "id": "doc-1",
                    "file_name": "demo.pdf",
                    "original_path": str(artifact_dir / "demo.pdf"),
                    "processing_step": "index_failed",
                    "status": "completed",
                }
            ),
        ),
        patch("pdfserviceMD.service.GraphStore") as graph_store_cls,
    ):
        graph_store_cls.return_value.active_job_state = None
        context = await prepare_retry_index_context(
            doc_id="doc-1", user_id=TEST_USER_ID
        )

    assert context == RetryIndexContext(
        doc_id="doc-1",
        book_title="demo",
        current_status="completed",
        user_id=TEST_USER_ID,
        user_folder=str(artifact_dir),
    )
