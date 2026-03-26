"""Shared upload-path and PDF validation helpers."""

from __future__ import annotations

import os
from pathlib import Path

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


def resolve_document_user_folder(
    *,
    user_id: str,
    doc_id: str,
    original_path: str | None,
) -> Path:
    """Resolve the document folder from stored file metadata or fallback upload layout."""
    if original_path:
        return Path(original_path).resolve().parent
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
