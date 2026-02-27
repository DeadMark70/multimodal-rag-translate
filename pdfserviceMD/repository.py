"""Repository helpers for PDF document persistence."""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool
from postgrest.exceptions import APIError as PostgrestAPIError

from core.errors import AppError, ErrorCode
from supabase_client import get_supabase


async def create_document_record(
    *,
    doc_id: str,
    user_id: str,
    file_name: str,
    original_path: str,
    source_lang: str = "auto",
    target_lang: str = "zh-TW",
) -> None:
    """Inserts document metadata row for processing pipeline."""
    client = get_supabase()
    if not client:
        return

    payload = {
        "id": doc_id,
        "user_id": user_id,
        "file_name": file_name,
        "file_type": "pdf",
        "original_path": original_path,
        "status": "processing",
        "source_lang": source_lang,
        "target_lang": target_lang,
    }

    try:
        await run_in_threadpool(
            lambda: client.table("documents").insert(payload).execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to create document record",
            status_code=500,
            details={"operation": "create_document_record", "error": str(exc)},
        ) from exc


async def update_document_status(
    *,
    doc_id: str,
    status: str,
    translated_path: str | None = None,
    error_message: str | None = None,
) -> None:
    """Updates processing status for a document."""
    client = get_supabase()
    if not client:
        return

    payload: dict[str, str | None] = {"status": status, "error_message": error_message}
    if translated_path:
        payload["translated_path"] = translated_path

    try:
        await run_in_threadpool(
            lambda: client.table("documents")
            .update(payload)
            .eq("id", doc_id)
            .execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to update document status",
            status_code=500,
            details={"operation": "update_document_status", "error": str(exc)},
        ) from exc


async def update_processing_step(*, doc_id: str, step: str) -> None:
    """Updates processing step field."""
    client = get_supabase()
    if not client:
        return

    try:
        await run_in_threadpool(
            lambda: client.table("documents")
            .update({"processing_step": step})
            .eq("id", doc_id)
            .execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to update processing step",
            status_code=500,
            details={"operation": "update_processing_step", "error": str(exc)},
        ) from exc


async def list_documents(*, user_id: str, limit: int = 50) -> list[dict]:
    """Lists document metadata rows for one user."""
    client = get_supabase()
    if not client:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Database service unavailable",
            status_code=500,
        )

    try:
        response = await run_in_threadpool(
            lambda: client.table("documents")
            .select("id, file_name, created_at, status, processing_step")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to retrieve documents",
            status_code=500,
            details={"operation": "list_documents"},
        ) from exc


async def get_document(
    *,
    doc_id: str,
    user_id: str,
    columns: str = "*",
) -> dict | None:
    """Gets one document row by id + user."""
    client = get_supabase()
    if not client:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Database service unavailable",
            status_code=500,
        )

    try:
        response = await run_in_threadpool(
            lambda: client.table("documents")
            .select(columns)
            .eq("id", doc_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return response.data[0]
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to query document",
            status_code=500,
            details={"operation": "get_document"},
        ) from exc


async def delete_document(*, doc_id: str, user_id: str) -> None:
    """Deletes one document row by id + user."""
    client = get_supabase()
    if not client:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Database service unavailable",
            status_code=500,
        )

    try:
        await run_in_threadpool(
            lambda: client.table("documents")
            .delete()
            .eq("id", doc_id)
            .eq("user_id", user_id)
            .execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to delete document record",
            status_code=500,
            details={"operation": "delete_document"},
        ) from exc


async def clear_document_summary(*, doc_id: str, user_id: str) -> None:
    """Clears executive summary field for regeneration."""
    client = get_supabase()
    if not client:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Database service unavailable",
            status_code=500,
        )

    try:
        await run_in_threadpool(
            lambda: client.table("documents")
            .update({"executive_summary": None})
            .eq("id", doc_id)
            .eq("user_id", user_id)
            .execute()
        )
    except PostgrestAPIError as exc:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Failed to reset executive summary",
            status_code=500,
            details={"operation": "clear_document_summary"},
        ) from exc
