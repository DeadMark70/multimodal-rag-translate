from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from postgrest.exceptions import APIError as PostgrestAPIError

from core.errors import AppError, ErrorCode
from core.supabase_repository import execute_supabase_operation, get_supabase_client_or_raise


def test_get_supabase_client_or_raise_uses_typed_unavailable_error() -> None:
    with patch("core.supabase_repository.get_supabase", return_value=None):
        with pytest.raises(AppError) as exc_info:
            get_supabase_client_or_raise(
                error_code=ErrorCode.AUTH_SERVICE_UNAVAILABLE,
                message="Authentication service unavailable",
                status_code=500,
            )

    assert exc_info.value.code == ErrorCode.AUTH_SERVICE_UNAVAILABLE
    assert exc_info.value.message == "Authentication service unavailable"
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_execute_supabase_operation_retries_transport_errors() -> None:
    response = SimpleNamespace(data=[{"id": "doc-1"}])

    with (
        patch("core.supabase_repository.get_supabase", return_value=Mock()),
        patch(
            "core.supabase_repository.run_in_threadpool",
            new=AsyncMock(side_effect=[httpx.ReadError("boom"), response]),
        ) as run_in_threadpool_mock,
        patch("core.supabase_repository.init_supabase") as init_supabase_mock,
        patch("core.supabase_repository.asyncio.sleep", new=AsyncMock()) as sleep_mock,
    ):
        result = await execute_supabase_operation(
            operation="list_documents",
            failure_message="Failed to list documents",
            handler=lambda client: client.table("documents").select("*").execute(),
        )

    assert result is response
    assert run_in_threadpool_mock.await_count == 2
    init_supabase_mock.assert_called_once_with(force=True)
    sleep_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_supabase_operation_returns_503_after_retry_exhaustion() -> None:
    with (
        patch("core.supabase_repository.get_supabase", return_value=Mock()),
        patch(
            "core.supabase_repository.run_in_threadpool",
            new=AsyncMock(
                side_effect=[
                    httpx.ReadError("boom-1"),
                    httpx.ReadError("boom-2"),
                    httpx.ReadError("boom-3"),
                ]
            ),
        ),
        patch("core.supabase_repository.init_supabase") as init_supabase_mock,
        patch("core.supabase_repository.asyncio.sleep", new=AsyncMock()) as sleep_mock,
    ):
        with pytest.raises(AppError) as exc_info:
            await execute_supabase_operation(
                operation="get_document",
                failure_message="Failed to load document",
                handler=lambda client: client.table("documents").select("*").execute(),
            )

    assert exc_info.value.status_code == 503
    assert exc_info.value.message == "Database service temporarily unavailable"
    assert init_supabase_mock.call_count == 2
    assert sleep_mock.await_count == 2


@pytest.mark.asyncio
async def test_execute_supabase_operation_maps_postgrest_errors() -> None:
    with (
        patch("core.supabase_repository.get_supabase", return_value=Mock()),
        patch(
            "core.supabase_repository.run_in_threadpool",
            new=AsyncMock(side_effect=PostgrestAPIError({"message": "db-fail"})),
        ),
    ):
        with pytest.raises(AppError) as exc_info:
            await execute_supabase_operation(
                operation="create_message",
                failure_message="Failed to create message",
                handler=lambda client: client.table("messages").insert({}).execute(),
            )

    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "Failed to create message"
