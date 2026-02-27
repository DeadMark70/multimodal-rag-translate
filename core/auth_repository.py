"""Repository helpers for authentication-related Supabase access."""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool

from core.errors import AppError, ErrorCode
from supabase_client import get_supabase


async def fetch_user_id_from_token(token: str) -> str:
    """Validates token via Supabase and returns user id."""
    client = get_supabase()
    if not client:
        raise AppError(
            code=ErrorCode.AUTH_SERVICE_UNAVAILABLE,
            message="Authentication service unavailable",
            status_code=500,
        )

    try:
        user_response = await run_in_threadpool(lambda: client.auth.get_user(token))
    except Exception as exc:  # noqa: BLE001
        raise AppError(
            code=ErrorCode.UNAUTHORIZED,
            message="Authentication failed",
            status_code=401,
        ) from exc

    if not user_response or not user_response.user:
        raise AppError(
            code=ErrorCode.UNAUTHORIZED,
            message="Invalid token",
            status_code=401,
        )

    return user_response.user.id
