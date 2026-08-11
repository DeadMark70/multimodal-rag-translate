"""Shared upload-path and PDF validation helpers."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import UploadFile

from core.errors import AppError, ErrorCode

BASE_UPLOAD_FOLDER = "uploads"


def ensure_upload_root() -> str:
    """Ensure the shared upload root exists and return it."""
    os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
    return BASE_UPLOAD_FOLDER


def get_user_upload_dir(user_id: str) -> str:
    """Return the normalized upload directory for one user."""
    return os.path.normpath(os.path.join(ensure_upload_root(), user_id))


def get_document_upload_dir(user_id: str, doc_id: str) -> str:
    """Return the normalized upload directory for one document."""
    return os.path.normpath(os.path.join(get_user_upload_dir(user_id), doc_id))


def get_rag_index_dir(user_id: str) -> str:
    """Return the normalized RAG index directory for one user."""
    return os.path.normpath(os.path.join(get_user_upload_dir(user_id), "rag_index"))


def get_rag_index_dir_path(user_id: str) -> Path:
    """Return the user RAG index directory as a Path."""
    return Path(get_rag_index_dir(user_id))


def get_evaluation_dir(user_id: str) -> Path:
    """Return the evaluation-data directory for one user."""
    return Path(get_user_upload_dir(user_id)) / "evaluation"


def _validate_path_component(value: str, *, label: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single path component")


def build_document_storage_path(
    *, user_id: str, doc_id: str, filename: str
) -> str:
    """Build the canonical portable storage reference for one document."""
    for label, value in (
        ("user_id", user_id), ("doc_id", doc_id), ("filename", filename)
    ):
        _validate_path_component(value, label=label)
    return PurePosixPath(BASE_UPLOAD_FOLDER, user_id, doc_id, filename).as_posix()


def normalize_document_storage_path(
    *, user_id: str, doc_id: str, storage_path: str
) -> str:
    """Normalize a portable or legacy storage reference for its exact document."""
    _validate_path_component(user_id, label="user_id")
    _validate_path_component(doc_id, label="doc_id")
    if not isinstance(storage_path, str) or not storage_path:
        raise ValueError("storage path must be a nonempty string")
    if "/" in storage_path and "\\" in storage_path:
        raise ValueError("storage path must use one separator style")
    separator = "\\" if "\\" in storage_path else "/"
    raw_parts = storage_path.split(separator)
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("storage path contains an unsafe component")
    parsed = (
        PureWindowsPath(storage_path)
        if "\\" in storage_path
        else PurePosixPath(storage_path)
    )
    if parsed.is_absolute() or parsed.drive or parsed.root:
        raise ValueError("storage path must be relative")
    parts = parsed.parts
    if len(parts) != 4 or parts[:3] != (BASE_UPLOAD_FOLDER, user_id, doc_id):
        raise ValueError("storage path is outside the authorized document")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("storage path contains an unsafe component")
    _validate_path_component(parts[3], label="filename")
    return PurePosixPath(*parts).as_posix()


def resolve_document_storage_path(
    *, user_id: str, doc_id: str, storage_path: str
) -> Path:
    """Resolve one authorized portable storage reference to the local filesystem."""
    canonical = normalize_document_storage_path(
        user_id=user_id, doc_id=doc_id, storage_path=storage_path
    )
    upload_root = Path(ensure_upload_root()).resolve()
    document_root = (upload_root / user_id / doc_id).resolve()
    if not document_root.is_relative_to(upload_root):
        raise ValueError("document root escapes the upload root")
    canonical_parts = PurePosixPath(canonical).parts
    candidate = upload_root.joinpath(*canonical_parts[1:]).resolve()
    if not candidate.is_relative_to(upload_root) or not candidate.is_relative_to(
        document_root
    ):
        raise ValueError("storage path escapes the authorized document")
    return candidate


def resolve_upload_storage_reference(
    *,
    user_id: str,
    doc_id: str,
    storage_reference: str,
) -> Path:
    """Resolve one manifest reference beneath its exact user/document folder."""
    if not storage_reference or Path(storage_reference).is_absolute():
        raise ValueError("storage reference must be upload-root-relative")
    if any(
        value in {"", ".", ".."} or "/" in value or "\\" in value
        for value in (user_id, doc_id)
    ):
        raise ValueError("user and document IDs must be single path components")
    upload_root = Path(ensure_upload_root()).resolve()
    document_root = (upload_root / user_id / doc_id).resolve()
    if not document_root.is_relative_to(upload_root):
        raise ValueError("document root escapes the upload root")
    candidate = (upload_root / Path(storage_reference)).resolve()
    if not candidate.is_relative_to(upload_root) or not candidate.is_relative_to(
        document_root
    ):
        raise ValueError("storage reference escapes the authorized document")
    return candidate


def resolve_document_user_folder(
    *,
    user_id: str,
    doc_id: str,
    original_path: str | None,
) -> Path:
    """Resolve the document folder from stored file metadata or fallback upload layout."""
    if original_path:
        return resolve_document_storage_path(
            user_id=user_id,
            doc_id=doc_id,
            storage_path=original_path,
        ).parent
    return Path(get_document_upload_dir(user_id, doc_id))


def validate_pdf_upload(file: UploadFile) -> None:
    """Validate that the uploaded file is a PDF."""
    if file.content_type != "application/pdf":
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="File must be a PDF (invalid content-type)",
            status_code=400,
        )

    if file.filename:
        _, ext = os.path.splitext(file.filename)
        if ext.lower() != ".pdf":
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="File must be a PDF (invalid extension)",
                status_code=400,
            )
