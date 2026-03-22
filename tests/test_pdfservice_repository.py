from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from core.errors import AppError
from pdfserviceMD.repository import get_document


@pytest.mark.asyncio
async def test_get_document_retries_transient_transport_error() -> None:
    response = SimpleNamespace(data=[{"id": "doc-1", "status": "ready"}])

    with (
        patch("pdfserviceMD.repository.get_supabase", return_value=Mock()),
        patch(
            "pdfserviceMD.repository.run_in_threadpool",
            new=AsyncMock(side_effect=[httpx.ReadError("boom"), response]),
        ) as run_in_threadpool_mock,
        patch("pdfserviceMD.repository.init_supabase") as init_supabase_mock,
        patch("pdfserviceMD.repository.asyncio.sleep", new=AsyncMock()) as sleep_mock,
    ):
        row = await get_document(doc_id="doc-1", user_id="user-1")

    assert row == {"id": "doc-1", "status": "ready"}
    assert run_in_threadpool_mock.await_count == 2
    init_supabase_mock.assert_called_once_with(force=True)
    sleep_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_document_returns_503_after_exhausting_transport_retries() -> None:
    with (
        patch("pdfserviceMD.repository.get_supabase", return_value=Mock()),
        patch(
            "pdfserviceMD.repository.run_in_threadpool",
            new=AsyncMock(
                side_effect=[
                    httpx.ReadError("boom-1"),
                    httpx.ReadError("boom-2"),
                    httpx.ReadError("boom-3"),
                ]
            ),
        ),
        patch("pdfserviceMD.repository.init_supabase") as init_supabase_mock,
        patch("pdfserviceMD.repository.asyncio.sleep", new=AsyncMock()) as sleep_mock,
    ):
        with pytest.raises(AppError) as exc_info:
            await get_document(doc_id="doc-1", user_id="user-1")

    assert exc_info.value.status_code == 503
    assert exc_info.value.message == "Database service temporarily unavailable"
    assert init_supabase_mock.call_count == 2
    assert sleep_mock.await_count == 2
