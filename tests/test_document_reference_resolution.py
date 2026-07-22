"""Tests for resolving legacy evaluation source references safely."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from data_base import repository


class _DocumentsQuery:
    def __init__(self, rows_by_field: dict[str, list[dict[str, str]]]) -> None:
        self._rows_by_field = rows_by_field
        self._field: str | None = None
        self.user_ids: list[str] = []

    def select(self, _columns: str) -> _DocumentsQuery:
        return self

    def eq(self, field: str, value: str) -> _DocumentsQuery:
        assert field == "user_id"
        self.user_ids.append(value)
        return self

    def in_(self, field: str, references: list[str]) -> _DocumentsQuery:
        if field == "id":
            assert references == ["11111111-1111-4111-8111-111111111111"]
        self._field = field
        return self

    def execute(self) -> SimpleNamespace:
        assert self._field is not None
        return SimpleNamespace(data=self._rows_by_field[self._field])


@pytest.mark.asyncio
async def test_resolve_document_references_maps_filename_and_uuid_without_hiding_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _DocumentsQuery(
        {
            "id": [
                {
                    "id": "11111111-1111-4111-8111-111111111111",
                    "file_name": "paper.pdf",
                }
            ],
            "file_name": [
                {
                    "id": "11111111-1111-4111-8111-111111111111",
                    "file_name": "paper.pdf",
                },
                {
                    "id": "22222222-2222-4222-8222-222222222222",
                    "file_name": "paper.pdf",
                },
            ],
        }
    )

    async def fake_execute(*, handler, **_kwargs):
        return handler(SimpleNamespace(table=lambda name: _table(name, query)))

    def _table(name: str, result: _DocumentsQuery) -> _DocumentsQuery:
        assert name == "documents"
        return result

    monkeypatch.setattr(repository, "execute_supabase_operation", fake_execute)

    resolved = await repository.resolve_document_references(
        user_id="user-a",
        references=["11111111-1111-4111-8111-111111111111", "paper.pdf"],
    )

    assert resolved == {
        "11111111-1111-4111-8111-111111111111": [
            "11111111-1111-4111-8111-111111111111"
        ],
        "paper.pdf": [
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        ],
    }
    assert query.user_ids == ["user-a", "user-a"]
