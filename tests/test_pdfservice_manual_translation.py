from contextlib import contextmanager
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from core.errors import AppError
from main import app
from pdfserviceMD.service import (
    delete_user_document,
    finalize_indexing_status,
    get_document_file_info,
    translate_user_document,
)
from pdfserviceMD.schemas import TranslatePdfResponse

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
async def test_get_document_file_info_supports_type_selection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "uploads" / TEST_USER_ID / "doc-1"
    folder.mkdir(parents=True)
    original_file = folder / "demo.pdf"
    translated_file = folder / "translated_demo.pdf"
    original_file.write_bytes(b"%PDF")
    translated_file.write_bytes(b"%PDF")
    row = {
        "file_name": "demo.pdf",
        "original_path": rf"uploads\{TEST_USER_ID}\doc-1\demo.pdf",
        "translated_path": f"uploads/{TEST_USER_ID}/doc-1/translated_demo.pdf",
    }

    with patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)):
        default_path, default_name = await get_document_file_info(
            doc_id="doc-1",
            user_id=TEST_USER_ID,
        )
        original_only_path, original_only_name = await get_document_file_info(
            doc_id="doc-1",
            user_id=TEST_USER_ID,
            file_type="original",
        )
        translated_only_path, translated_only_name = await get_document_file_info(
            doc_id="doc-1",
            user_id=TEST_USER_ID,
            file_type="translated",
        )

    assert default_path == str(translated_file.resolve())
    assert default_name == "translated_demo.pdf"
    assert original_only_path == str(original_file.resolve())
    assert original_only_name == "demo.pdf"
    assert translated_only_path == str(translated_file.resolve())
    assert translated_only_name == "translated_demo.pdf"


@pytest.mark.asyncio
async def test_translate_user_document_returns_409_when_artifacts_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    original_file = tmp_path / "uploads" / TEST_USER_ID / "doc-1" / "demo.pdf"
    original_file.parent.mkdir(parents=True)
    original_file.write_bytes(b"%PDF")
    row = {
        "file_name": "demo.pdf",
        "original_path": rf"uploads\{TEST_USER_ID}\doc-1\demo.pdf",
        "translated_path": None,
        "status": "ready",
        "processing_step": "ocr_completed",
    }

    with patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)):
        with pytest.raises(AppError) as exc_info:
            await translate_user_document(doc_id="doc-1", user_id=TEST_USER_ID)

    assert exc_info.value.status_code == 409
    assert exc_info.value.message == "OCR artifacts are unavailable for translation"


@pytest.mark.asyncio
async def test_finalize_indexing_status_preserves_completed_translation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    translated_file = (
        tmp_path / "uploads" / TEST_USER_ID / "doc-1" / "translated_demo.pdf"
    )
    translated_file.parent.mkdir(parents=True)
    translated_file.write_bytes(b"%PDF")
    row = {
        "status": "completed",
        "translated_path": f"uploads/{TEST_USER_ID}/doc-1/translated_demo.pdf",
    }

    with (
        patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)),
        patch("pdfserviceMD.service.update_document_status", new=AsyncMock()) as update_status,
    ):
        await finalize_indexing_status(doc_id="doc-1", user_id=TEST_USER_ID)

    update_status.assert_not_called()


@pytest.mark.asyncio
async def test_translate_user_document_persists_portable_path_and_generates_locally(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "uploads" / TEST_USER_ID / "doc-1"
    folder.mkdir(parents=True)
    original_file = folder / "demo.pdf"
    original_file.write_bytes(b"%PDF")
    (folder / "extracted.md").write_text("markdown", encoding="utf-8")
    (folder / "image_blocks.json").write_text("[]", encoding="utf-8")
    row = {
        "file_name": "demo.pdf",
        "original_path": f"uploads/{TEST_USER_ID}/doc-1/demo.pdf",
        "translated_path": None,
        "status": "ready",
        "processing_step": "ocr_completed",
    }
    output_pdf = folder / "translated_demo.pdf"

    def write_pdf(_markdown: str, output_path: str, _user_folder: str) -> None:
        assert Path(output_path) == output_pdf.resolve()
        output_pdf.write_bytes(b"%PDF")

    with (
        patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)),
        patch("pdfserviceMD.service.update_processing_step", new=AsyncMock()),
        patch("pdfserviceMD.service.update_document_status", new=AsyncMock()) as update_status,
        patch("pdfserviceMD.service.translate_text", new=AsyncMock(return_value="translated")),
        patch("pdfserviceMD.service.replace_markdown", return_value="translated"),
        patch("pdfserviceMD.service.markdown_to_pdf", side_effect=write_pdf),
    ):
        await translate_user_document(doc_id="doc-1", user_id=TEST_USER_ID)

    assert update_status.await_args.kwargs["translated_path"] == (
        f"uploads/{TEST_USER_ID}/doc-1/translated_demo.pdf"
    )


@pytest.mark.asyncio
async def test_pdf_generation_failure_keeps_exception_path_out_of_outputs(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "uploads" / TEST_USER_ID / "doc-1"
    folder.mkdir(parents=True)
    (folder / "demo.pdf").write_bytes(b"%PDF")
    (folder / "extracted.md").write_text("markdown", encoding="utf-8")
    (folder / "image_blocks.json").write_text("[]", encoding="utf-8")
    sentinel = str(tmp_path / "private-sentinel.pdf")
    row = {
        "file_name": "demo.pdf",
        "original_path": f"uploads/{TEST_USER_ID}/doc-1/demo.pdf",
        "translated_path": None,
        "status": "ready",
        "processing_step": "ocr_completed",
    }
    update_status = AsyncMock()

    with (
        patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)),
        patch("pdfserviceMD.service.update_processing_step", new=AsyncMock()),
        patch(
            "pdfserviceMD.service.update_document_status", new=update_status
        ),
        patch(
            "pdfserviceMD.service.translate_text",
            new=AsyncMock(return_value="translated"),
        ),
        patch("pdfserviceMD.service.replace_markdown", return_value="translated"),
        patch(
            "pdfserviceMD.service.markdown_to_pdf",
            side_effect=RuntimeError(f"renderer failed at {sentinel}"),
        ),
        caplog.at_level(logging.INFO, logger="pdfserviceMD.service"),
    ):
        response = await translate_user_document(
            doc_id="doc-1", user_id=TEST_USER_ID
        )

    assert update_status.await_args.kwargs["error_message"] == "PDF generation failed"
    assert response.pdf_error == "PDF generation failed"
    assert sentinel not in caplog.text
    assert sentinel not in response.model_dump_json()


def test_translate_endpoint_returns_service_response() -> None:
    doc_id = str(uuid4())
    payload = TranslatePdfResponse(
        doc_id=doc_id,
        status="completed",
        message="Translation completed successfully.",
        pdf_available=True,
        pdf_download_url=f"/pdfmd/file/{doc_id}?type=translated",
        pdf_error=None,
    )

    with _build_client() as client, patch(
        "pdfserviceMD.router.translate_user_document",
        new=AsyncMock(return_value=payload),
    ):
        response = client.post(f"/pdfmd/file/{doc_id}/translate")

    assert response.status_code == 200
    assert response.json()["pdf_download_url"] == f"/pdfmd/file/{doc_id}?type=translated"


def test_get_pdf_file_forwards_type_query_param() -> None:
    doc_id = str(uuid4())
    with _build_client() as client, patch(
        "pdfserviceMD.router.get_document_file_info",
        new=AsyncMock(return_value=(__file__, "demo.pdf")),
    ) as get_document_file_info_mock:
        response = client.get(f"/pdfmd/file/{doc_id}?type=original")

    assert response.status_code == 200
    get_document_file_info_mock.assert_awaited_once_with(
        doc_id=doc_id,
        user_id=TEST_USER_ID,
        file_type="original",
    )


@pytest.mark.asyncio
async def test_delete_user_document_triggers_best_effort_graph_purge() -> None:
    with (
        patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value={"id": "doc-1"})),
        patch("pdfserviceMD.service.run_in_threadpool", new=AsyncMock()),
        patch("pdfserviceMD.service.os.path.exists", return_value=False),
        patch("pdfserviceMD.service.delete_document", new=AsyncMock()) as delete_record,
        patch("graph_rag.maintenance.purge_graph_document_task", new=AsyncMock()) as purge_task,
        patch("graph_rag.store.GraphStore") as graph_store_cls,
    ):
        graph_store = graph_store_cls.return_value
        graph_store.active_job_state = None
        graph_store.get_document_status.return_value = object()
        graph_store.get_documents.return_value = set()

        response = await delete_user_document(
            doc_id="doc-1",
            user_id=TEST_USER_ID,
            base_upload_folder="uploads",
        )

    assert response.status == "success"
    delete_record.assert_awaited_once_with(doc_id="doc-1", user_id=TEST_USER_ID)
    purge_task.assert_awaited_once_with(TEST_USER_ID, "doc-1")
