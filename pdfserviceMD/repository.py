"""Repository helpers for PDF document persistence."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import httpx
from fastapi.concurrency import run_in_threadpool
from postgrest.exceptions import APIError as PostgrestAPIError

from core.errors import AppError, ErrorCode
from supabase_client import get_supabase, init_supabase

logger = logging.getLogger(__name__)

_SUPABASE_MAX_ATTEMPTS = 3
_SUPABASE_BASE_DELAY_SECONDS = 0.25


def _database_unavailable_error(*, operation: str, error: str) -> AppError:
    return AppError(
        code=ErrorCode.DATABASE_ERROR,
        message="Database service temporarily unavailable",
        status_code=503,
        details={"operation": operation, "error": error},
    )


async def _execute_supabase_operation(
    *,
    operation: str,
    failure_message: str,
    handler: Callable[[Any], Any],
) -> Any:
    """Runs one Supabase request with retry on transient transport errors."""
    client = get_supabase()
    if not client:
        raise AppError(
            code=ErrorCode.DATABASE_ERROR,
            message="Database service unavailable",
            status_code=500,
        )

    for attempt in range(1, _SUPABASE_MAX_ATTEMPTS + 1):
        client = get_supabase()
        if not client:
            raise AppError(
                code=ErrorCode.DATABASE_ERROR,
                message="Database service unavailable",
                status_code=500,
            )

        try:
            return await run_in_threadpool(lambda: handler(client))
        except PostgrestAPIError as exc:
            raise AppError(
                code=ErrorCode.DATABASE_ERROR,
                message=failure_message,
                status_code=500,
                details={"operation": operation, "error": str(exc)},
            ) from exc
        except httpx.TransportError as exc:
            if attempt >= _SUPABASE_MAX_ATTEMPTS:
                raise _database_unavailable_error(
                    operation=operation,
                    error=str(exc),
                ) from exc

            delay = _SUPABASE_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Transient Supabase transport error during %s (attempt %s/%s): %s",
                operation,
                attempt,
                _SUPABASE_MAX_ATTEMPTS,
                exc,
            )
            init_supabase(force=True)
            await asyncio.sleep(delay)

    raise _database_unavailable_error(
        operation=operation,
        error="retry loop exhausted without a response",
    )


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

    await _execute_supabase_operation(
        operation="create_document_record",
        failure_message="Failed to create document record",
        handler=lambda client: client.table("documents").insert(payload).execute(),
    )


async def update_document_status(
    *,
    doc_id: str,
    status: str,
    translated_path: str | None = None,
    error_message: str | None = None,
) -> None:
    """Updates processing status for a document."""
    payload: dict[str, str | None] = {"status": status, "error_message": error_message}
    if translated_path:
        payload["translated_path"] = translated_path

    await _execute_supabase_operation(
        operation="update_document_status",
        failure_message="Failed to update document status",
        handler=lambda client: client.table("documents")
        .update(payload)
        .eq("id", doc_id)
        .execute(),
    )


async def update_processing_step(*, doc_id: str, step: str) -> None:
    """Updates processing step field."""
    await _execute_supabase_operation(
        operation="update_processing_step",
        failure_message="Failed to update processing step",
        handler=lambda client: client.table("documents")
        .update({"processing_step": step})
        .eq("id", doc_id)
        .execute(),
    )


async def list_documents(*, user_id: str, limit: int = 50) -> list[dict]:
    """Lists document metadata rows for one user."""
    response = await _execute_supabase_operation(
        operation="list_documents",
        failure_message="Failed to retrieve documents",
        handler=lambda client: client.table("documents")
            .select(
                "id, file_name, created_at, status, processing_step, original_path, translated_path, error_message"
            )
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute(),
    )
    return response.data or []


async def get_document(
    *,
    doc_id: str,
    user_id: str,
    columns: str = "*",
) -> dict | None:
    """Gets one document row by id + user."""
    response = await _execute_supabase_operation(
        operation="get_document",
        failure_message="Failed to query document",
        handler=lambda client: client.table("documents")
            .select(columns)
            .eq("id", doc_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute(),
    )
    if not response.data:
        return None
    return response.data[0]


async def delete_document(*, doc_id: str, user_id: str) -> None:
    """Deletes one document row by id + user."""
    await _execute_supabase_operation(
        operation="delete_document",
        failure_message="Failed to delete document record",
        handler=lambda client: client.table("documents")
        .delete()
        .eq("id", doc_id)
        .eq("user_id", user_id)
        .execute(),
    )


async def clear_document_summary(*, doc_id: str, user_id: str) -> None:
    """Clears executive summary field for regeneration."""
    await _execute_supabase_operation(
        operation="clear_document_summary",
        failure_message="Failed to reset executive summary",
        handler=lambda client: client.table("documents")
        .update({"executive_summary": None})
        .eq("id", doc_id)
        .eq("user_id", user_id)
        .execute(),
    )
