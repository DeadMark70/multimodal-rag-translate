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


def test_resolve_document_user_folder_prefers_original_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = upload_paths.resolve_document_user_folder(
        user_id="user-1",
        doc_id="doc-1",
        original_path=r"uploads\user-1\doc-1\demo.pdf",
    )

    assert resolved == (tmp_path / "uploads" / "user-1" / "doc-1").resolve()


def test_resolve_document_user_folder_uses_fallback_layout() -> None:
    resolved = upload_paths.resolve_document_user_folder(
        user_id="user-1",
        doc_id="doc-1",
        original_path=None,
    )

    assert resolved.parts[-3:] == ("uploads", "user-1", "doc-1")


def test_build_document_storage_path_is_posix() -> None:
    assert upload_paths.build_document_storage_path(
        user_id="user-1", doc_id="doc-1", filename="paper.pdf"
    ) == "uploads/user-1/doc-1/paper.pdf"


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("uploads/user-1/doc-1/paper.pdf", "uploads/user-1/doc-1/paper.pdf"),
        (r"uploads\user-1\doc-1\paper.pdf", "uploads/user-1/doc-1/paper.pdf"),
    ],
)
def test_normalize_accepts_portable_and_legacy_relative(
    stored: str, expected: str
) -> None:
    assert upload_paths.normalize_document_storage_path(
        user_id="user-1", doc_id="doc-1", storage_path=stored
    ) == expected


@pytest.mark.parametrize(
    "stored",
    [
        "./uploads/user-1/doc-1/paper.pdf",
        "uploads//user-1/doc-1/paper.pdf",
        "uploads/user-1/doc-1/paper.pdf/",
        "/app/uploads/user-1/doc-1/paper.pdf",
        r"D:\uploads\user-1\doc-1\paper.pdf",
        r"\\server\share\paper.pdf",
        "uploads/user-1/doc-1/../doc-2/paper.pdf",
        "uploads/user-2/doc-1/paper.pdf",
        "uploads/user-1/doc-2/paper.pdf",
        r"uploads/user-1\doc-1/paper.pdf",
    ],
)
def test_normalize_rejects_unsafe_values(stored: str) -> None:
    with pytest.raises(ValueError):
        upload_paths.normalize_document_storage_path(
            user_id="user-1", doc_id="doc-1", storage_path=stored
        )


def test_resolve_legacy_path_inside_exact_document_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "uploads" / "user-1" / "doc-1" / "paper.pdf"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"%PDF")
    resolved = upload_paths.resolve_document_storage_path(
        user_id="user-1",
        doc_id="doc-1",
        storage_path=r"uploads\user-1\doc-1\paper.pdf",
    )
    assert resolved == expected.resolve()


def test_resolve_document_rejects_symlinked_document_root_escape(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    upload_root = tmp_path / "uploads"
    document_link = upload_root / "user-1" / "doc-1"
    outside = tmp_path / "outside"
    document_link.parent.mkdir(parents=True)
    outside.mkdir()
    try:
        document_link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError):
        upload_paths.resolve_document_storage_path(
            user_id="user-1",
            doc_id="doc-1",
            storage_path="uploads/user-1/doc-1/paper.pdf",
        )


def test_resolve_manifest_reference_rejects_symlinked_document_root_escape(
    tmp_path, monkeypatch
) -> None:
    upload_root = tmp_path / "uploads"
    document_link = upload_root / "user-1" / "doc-1"
    outside = tmp_path / "outside"
    document_link.parent.mkdir(parents=True)
    outside.mkdir()
    try:
        document_link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    monkeypatch.setattr(upload_paths, "BASE_UPLOAD_FOLDER", str(upload_root))

    with pytest.raises(ValueError):
        upload_paths.resolve_upload_storage_reference(
            user_id="user-1",
            doc_id="doc-1",
            storage_reference="user-1/doc-1/paper.png",
        )
