"""Shared Supabase repository helpers."""

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

SUPABASE_MAX_ATTEMPTS = 3
SUPABASE_BASE_DELAY_SECONDS = 0.25


def get_supabase_client_or_raise(
    *,
    error_code: ErrorCode = ErrorCode.DATABASE_ERROR,
    message: str = "Database service unavailable",
    status_code: int = 500,
) -> Any:
    """Return the cached Supabase client or raise a typed app error."""
    client = get_supabase()
    if client:
        return client
    raise AppError(code=error_code, message=message, status_code=status_code)


def _temporary_unavailable_error(
    *,
    operation: str,
    error_code: ErrorCode,
    message: str,
    status_code: int,
    error: str,
) -> AppError:
    return AppError(
        code=error_code,
        message=message,
        status_code=status_code,
        details={"operation": operation, "error": error},
    )


async def execute_supabase_operation(
    *,
    operation: str,
    handler: Callable[[Any], Any],
    failure_message: str,
    unavailable_error_code: ErrorCode = ErrorCode.DATABASE_ERROR,
    unavailable_message: str = "Database service unavailable",
    unavailable_status_code: int = 500,
    temporary_unavailable_message: str = "Database service temporarily unavailable",
    temporary_unavailable_status_code: int = 503,
    failure_status_code: int = 500,
    retry_transport_errors: bool = True,
) -> Any:
    """Run one Supabase operation with standardized error handling."""
    for attempt in range(1, SUPABASE_MAX_ATTEMPTS + 1):
        client = get_supabase_client_or_raise(
            error_code=unavailable_error_code,
            message=unavailable_message,
            status_code=unavailable_status_code,
        )

        try:
            return await run_in_threadpool(lambda: handler(client))
        except PostgrestAPIError as exc:
            raise AppError(
                code=unavailable_error_code,
                message=failure_message,
                status_code=failure_status_code,
                details={"operation": operation, "error": str(exc)},
            ) from exc
        except httpx.TransportError as exc:
            if not retry_transport_errors or attempt >= SUPABASE_MAX_ATTEMPTS:
                raise _temporary_unavailable_error(
                    operation=operation,
                    error_code=unavailable_error_code,
                    message=temporary_unavailable_message,
                    status_code=temporary_unavailable_status_code,
                    error=str(exc),
                ) from exc

            delay = SUPABASE_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Transient Supabase transport error during %s (attempt %s/%s): %s",
                operation,
                attempt,
                SUPABASE_MAX_ATTEMPTS,
                exc,
            )
            init_supabase(force=True)
            await asyncio.sleep(delay)

    raise _temporary_unavailable_error(
        operation=operation,
        error_code=unavailable_error_code,
        message=temporary_unavailable_message,
        status_code=temporary_unavailable_status_code,
        error="retry loop exhausted without a response",
    )
