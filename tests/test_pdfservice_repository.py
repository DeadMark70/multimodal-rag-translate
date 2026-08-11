from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from core.errors import AppError
from pdfserviceMD.repository import (
    get_document,
    get_owned_documents_by_ids,
    list_document_path_rows,
    update_owned_document_paths,
)


@pytest.mark.asyncio
async def test_get_document_retries_transient_transport_error() -> None:
    response = SimpleNamespace(data=[{"id": "doc-1", "status": "ready"}])

    with (
        patch("core.supabase_repository.get_supabase", return_value=Mock()),
        patch(
            "core.supabase_repository.run_in_threadpool",
            new=AsyncMock(side_effect=[httpx.ReadError("boom"), response]),
        ) as run_in_threadpool_mock,
        patch("core.supabase_repository.init_supabase") as init_supabase_mock,
        patch("core.supabase_repository.asyncio.sleep", new=AsyncMock()) as sleep_mock,
    ):
        row = await get_document(doc_id="doc-1", user_id="user-1")

    assert row == {"id": "doc-1", "status": "ready"}
    assert run_in_threadpool_mock.await_count == 2
    init_supabase_mock.assert_called_once_with(force=True)
    sleep_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_document_returns_503_after_exhausting_transport_retries() -> None:
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
            await get_document(doc_id="doc-1", user_id="user-1")

    assert exc_info.value.status_code == 503
    assert exc_info.value.message == "Database service temporarily unavailable"
    assert init_supabase_mock.call_count == 2
    assert sleep_mock.await_count == 2


@pytest.mark.asyncio
async def test_get_owned_documents_by_ids_batches_and_scopes_every_query() -> None:
    requested_ids = [f"doc-{index:03d}" for index in range(101)]
    seen_batches: list[list[str]] = []
    seen_users: list[str] = []

    class Query:
        def __init__(self) -> None:
            self.ids: list[str] = []

        def select(self, columns: str):
            assert columns == "id,file_name"
            return self

        def eq(self, field: str, value: str):
            assert field == "user_id"
            seen_users.append(value)
            return self

        def in_(self, field: str, values: list[str]):
            assert field == "id"
            self.ids = list(values)
            seen_batches.append(self.ids)
            return self

        def execute(self):
            return SimpleNamespace(
                data=[{"id": doc_id, "file_name": f"{doc_id}.pdf"} for doc_id in self.ids]
            )

    class Client:
        def table(self, name: str):
            assert name == "documents"
            return Query()

    async def fake_execute(*, handler, **_):
        return handler(Client())

    with patch(
        "pdfserviceMD.repository.execute_supabase_operation",
        new=AsyncMock(side_effect=fake_execute),
    ):
        rows = await get_owned_documents_by_ids(
            doc_ids=requested_ids,
            user_id="user-1",
            columns="id,file_name",
        )

    assert {row["id"] for row in rows} == set(requested_ids)
    assert len(seen_batches) == 2
    assert max(map(len, seen_batches)) <= 100
    assert seen_users == ["user-1", "user-1"]


@pytest.mark.asyncio
async def test_update_owned_document_paths_scopes_by_document_and_user() -> None:
    seen: list[tuple[str, object]] = []

    class Query:
        def update(self, payload: dict[str, str]):
            seen.append(("payload", payload))
            return self

        def eq(self, field: str, value: str):
            seen.append((field, value))
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    class Client:
        def table(self, name: str):
            assert name == "documents"
            return Query()

    async def fake_execute(*, handler, **_):
        return handler(Client())

    with patch(
        "pdfserviceMD.repository.execute_supabase_operation",
        new=AsyncMock(side_effect=fake_execute),
    ):
        await update_owned_document_paths(
            doc_id="doc-1",
            user_id="user-1",
            paths={"original_path": "uploads/user-1/doc-1/paper.pdf"},
        )

    assert seen == [
        ("payload", {"original_path": "uploads/user-1/doc-1/paper.pdf"}),
        ("id", "doc-1"),
        ("user_id", "user-1"),
    ]


@pytest.mark.asyncio
async def test_list_document_path_rows_uses_stable_id_pagination() -> None:
    seen: list[tuple[str, object]] = []

    class Query:
        def select(self, columns: str):
            seen.append(("select", columns))
            return self

        def order(self, field: str):
            seen.append(("order", field))
            return self

        def range(self, start: int, end: int):
            seen.append(("range", (start, end)))
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    class Client:
        def table(self, name: str):
            assert name == "documents"
            return Query()

    async def fake_execute(*, handler, **_):
        return handler(Client())

    with patch(
        "pdfserviceMD.repository.execute_supabase_operation",
        new=AsyncMock(side_effect=fake_execute),
    ):
        rows = await list_document_path_rows(offset=100, limit=100)

    assert rows == []
    assert seen == [
        ("select", "id,user_id,original_path,translated_path"),
        ("order", "id"),
        ("range", (100, 199)),
    ]
