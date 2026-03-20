from contextlib import contextmanager
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from core.errors import AppError
from main import app
from pdfserviceMD.service import (
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
async def test_get_document_file_info_supports_type_selection() -> None:
    original_path = r"D:\flutterserver\pdftopng\uploads\demo.pdf"
    translated_path = r"D:\flutterserver\pdftopng\uploads\translated_demo.pdf"
    row = {
        "file_name": "demo.pdf",
        "original_path": original_path,
        "translated_path": translated_path,
    }

    with (
        patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)),
        patch("pdfserviceMD.service.os.path.exists", return_value=True),
    ):
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

    assert default_path == str(translated_path)
    assert default_name == "translated_demo.pdf"
    assert original_only_path == str(original_path)
    assert original_only_name == "demo.pdf"
    assert translated_only_path == str(translated_path)
    assert translated_only_name == "translated_demo.pdf"


@pytest.mark.asyncio
async def test_translate_user_document_returns_409_when_artifacts_missing() -> None:
    original_path = r"D:\flutterserver\pdftopng\uploads\demo.pdf"
    row = {
        "file_name": "demo.pdf",
        "original_path": original_path,
        "translated_path": None,
        "status": "ready",
        "processing_step": "ocr_completed",
    }

    def fake_exists(path: str) -> bool:
        normalized = path.replace("/", "\\")
        return normalized == original_path

    with (
        patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)),
        patch("pdfserviceMD.service.os.path.exists", side_effect=fake_exists),
    ):
        with pytest.raises(AppError) as exc_info:
            await translate_user_document(doc_id="doc-1", user_id=TEST_USER_ID)

    assert exc_info.value.status_code == 409
    assert exc_info.value.message == "OCR artifacts are unavailable for translation"


@pytest.mark.asyncio
async def test_finalize_indexing_status_preserves_completed_translation() -> None:
    row = {
        "status": "completed",
        "translated_path": __file__,
    }

    with (
        patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)),
        patch("pdfserviceMD.service.update_document_status", new=AsyncMock()) as update_status,
    ):
        await finalize_indexing_status(doc_id="doc-1", user_id=TEST_USER_ID)

    update_status.assert_not_called()


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
