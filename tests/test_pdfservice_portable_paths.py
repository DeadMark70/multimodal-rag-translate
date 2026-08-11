import logging
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from core.errors import AppError
from pdfserviceMD.service import (
    get_document_file_info,
    list_user_documents,
    run_upload_pipeline,
)


@pytest.mark.asyncio
async def test_upload_persists_portable_original_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    create_record = AsyncMock()
    upload = UploadFile(
        filename="paper.pdf",
        file=BytesIO(b"%PDF-1.7"),
        headers=Headers({"content-type": "application/pdf"}),
    )
    with (
        patch("pdfserviceMD.service.create_document_record", create_record),
        patch("pdfserviceMD.service.update_processing_step", new=AsyncMock()),
        patch("pdfserviceMD.service.update_document_status", new=AsyncMock()),
        patch("pdfserviceMD.service.ocr_service_sync", return_value="markdown"),
        patch("pdfserviceMD.service.markdown_extact", return_value=("markdown", [])),
        patch("pdfserviceMD.service.save_ocr_artifacts"),
    ):
        result = await run_upload_pipeline(
            file=upload, user_id="user-1", base_upload_folder="uploads"
        )
    stored = create_record.await_args.kwargs["original_path"]
    assert stored == f"uploads/user-1/{result.doc_id}/paper.pdf"
    assert "\\" not in stored


@pytest.mark.asyncio
async def test_download_resolves_legacy_windows_relative_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "uploads" / "user-1" / "doc-1" / "paper.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    row = {
        "file_name": "paper.pdf",
        "original_path": r"uploads\user-1\doc-1\paper.pdf",
        "translated_path": None,
    }
    with patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)):
        path, name = await get_document_file_info(
            doc_id="doc-1", user_id="user-1", file_type="original"
        )
    assert Path(path) == pdf.resolve()
    assert name == "paper.pdf"


@pytest.mark.asyncio
async def test_download_rejects_cross_document_path() -> None:
    row = {
        "file_name": "paper.pdf",
        "original_path": "uploads/user-1/doc-2/paper.pdf",
        "translated_path": None,
    }
    with patch("pdfserviceMD.service.get_document", new=AsyncMock(return_value=row)):
        with pytest.raises(AppError) as exc_info:
            await get_document_file_info(
                doc_id="doc-1", user_id="user-1", file_type="original"
            )
    assert exc_info.value.status_code == 404
    assert "uploads/" not in exc_info.value.message


@pytest.mark.asyncio
async def test_upload_failure_keeps_filesystem_path_out_of_logs_and_persistence(
    tmp_path, caplog
) -> None:
    private_root = tmp_path / "private-sentinel"
    sentinel = str(private_root)
    update_status = AsyncMock()
    upload = UploadFile(
        filename="paper.pdf",
        file=BytesIO(b"%PDF-1.7"),
        headers=Headers({"content-type": "application/pdf"}),
    )
    with (
        patch("pdfserviceMD.service.create_document_record", new=AsyncMock()),
        patch("pdfserviceMD.service.update_processing_step", new=AsyncMock()),
        patch(
            "pdfserviceMD.service.update_document_status", new=update_status
        ),
        patch(
            "pdfserviceMD.service.ocr_service_sync",
            side_effect=RuntimeError(f"OCR failed at {sentinel}"),
        ),
        caplog.at_level(logging.INFO, logger="pdfserviceMD.service"),
        pytest.raises(AppError) as exc_info,
    ):
        await run_upload_pipeline(
            file=upload,
            user_id="user-1",
            base_upload_folder=str(private_root),
        )

    assert exc_info.value.message == "Document processing failed"
    assert update_status.await_args.kwargs["error_message"] == (
        "Document processing failed"
    )
    assert sentinel not in caplog.text
    assert sentinel not in str(exc_info.value)


@pytest.mark.parametrize(
    "translated_path",
    [
        "uploads/user-1/doc-1/missing.pdf",
        "uploads/other-user/doc-1/translated.pdf",
    ],
)
@pytest.mark.asyncio
async def test_nonempty_translated_reference_disables_translation_when_unavailable(
    tmp_path, monkeypatch, translated_path: str
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "uploads" / "user-1" / "doc-1"
    folder.mkdir(parents=True)
    (folder / "paper.pdf").write_bytes(b"%PDF")
    (folder / "extracted.md").write_text("markdown", encoding="utf-8")
    (folder / "image_blocks.json").write_text("[]", encoding="utf-8")
    row = {
        "id": "doc-1",
        "file_name": "paper.pdf",
        "created_at": "2026-08-12T00:00:00Z",
        "status": "ready",
        "processing_step": "ocr_completed",
        "original_path": "uploads/user-1/doc-1/paper.pdf",
        "translated_path": translated_path,
        "error_message": None,
    }
    with patch(
        "pdfserviceMD.service.list_documents",
        new=AsyncMock(return_value=[row]),
    ):
        response = await list_user_documents(user_id="user-1")

    assert response.documents[0].has_translated_pdf is False
    assert response.documents[0].can_translate is False
