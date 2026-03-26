from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.errors import AppError
from core import uploads as upload_paths


def test_validate_pdf_upload_accepts_pdf() -> None:
    upload_paths.validate_pdf_upload(
        SimpleNamespace(content_type="application/pdf", filename="demo.pdf")
    )


def test_validate_pdf_upload_rejects_non_pdf_content_type() -> None:
    with pytest.raises(AppError) as exc_info:
        upload_paths.validate_pdf_upload(
            SimpleNamespace(content_type="image/png", filename="demo.pdf")
        )

    assert exc_info.value.status_code == 400
    assert "invalid content-type" in exc_info.value.message


def test_validate_pdf_upload_rejects_non_pdf_extension() -> None:
    with pytest.raises(AppError) as exc_info:
        upload_paths.validate_pdf_upload(
            SimpleNamespace(content_type="application/pdf", filename="demo.txt")
        )

    assert exc_info.value.status_code == 400
    assert "invalid extension" in exc_info.value.message


def test_resolve_document_user_folder_prefers_original_path() -> None:
    resolved = upload_paths.resolve_document_user_folder(
        user_id="user-1",
        doc_id="doc-1",
        original_path="uploads/user-1/doc-1/demo.pdf",
    )

    assert resolved.name == "doc-1"


def test_resolve_document_user_folder_uses_fallback_layout() -> None:
    resolved = upload_paths.resolve_document_user_folder(
        user_id="user-1",
        doc_id="doc-1",
        original_path=None,
    )

    assert str(resolved).endswith("uploads\\user-1\\doc-1")
