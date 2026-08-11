from unittest.mock import AsyncMock, patch

import pytest

from scripts.migrate_document_paths import migrate_document_paths


@pytest.mark.asyncio
async def test_dry_run_classifies_without_writes(capsys) -> None:
    rows = [{
        "id": "doc-1",
        "user_id": "user-1",
        "original_path": r"uploads\user-1\doc-1\paper.pdf",
        "translated_path": None,
    }]
    with (
        patch(
            "scripts.migrate_document_paths.list_document_path_rows",
            new=AsyncMock(side_effect=[rows, []]),
        ),
        patch(
            "scripts.migrate_document_paths.update_owned_document_paths",
            new=AsyncMock(),
        ) as update,
    ):
        summary = await migrate_document_paths(apply=False, batch_size=100)

    assert summary.changed_fields == 1
    assert summary.applied_fields == 0
    update.assert_not_awaited()
    output = capsys.readouterr().out
    assert "doc-1 original_path convertible" in output
    assert "user-1" not in output
    assert "paper.pdf" not in output
    assert "uploads" not in output


@pytest.mark.asyncio
async def test_apply_updates_only_convertible_fields() -> None:
    legacy = {
        "id": "doc-1",
        "user_id": "user-1",
        "original_path": r"uploads\user-1\doc-1\paper.pdf",
        "translated_path": "uploads/user-1/doc-1/translated_paper.pdf",
    }
    update = AsyncMock()
    with (
        patch(
            "scripts.migrate_document_paths.list_document_path_rows",
            new=AsyncMock(side_effect=[[legacy], []]),
        ),
        patch(
            "scripts.migrate_document_paths.update_owned_document_paths",
            new=update,
        ),
    ):
        summary = await migrate_document_paths(apply=True, batch_size=100)

    update.assert_awaited_once_with(
        doc_id="doc-1",
        user_id="user-1",
        paths={"original_path": "uploads/user-1/doc-1/paper.pdf"},
    )
    assert summary.applied_fields == 1


@pytest.mark.asyncio
async def test_migration_counts_null_portable_and_rejected_paths() -> None:
    rows = [{
        "id": "doc-1",
        "user_id": "user-1",
        "original_path": None,
        "translated_path": "uploads/user-1/doc-1/paper.pdf",
    }, {
        "id": "doc-2",
        "user_id": "user-2",
        "original_path": r"C:\uploads\user-2\doc-2\paper.pdf",
        "translated_path": "uploads/user-2/other-doc/paper.pdf",
    }]
    with patch(
        "scripts.migrate_document_paths.list_document_path_rows",
        new=AsyncMock(side_effect=[rows, []]),
    ):
        summary = await migrate_document_paths(apply=False)

    assert summary.scanned_rows == 2
    assert summary.changed_fields == 0
    assert summary.unchanged_fields == 1
    assert summary.rejected_fields == 2


@pytest.mark.asyncio
async def test_migration_rejects_unc_and_traversal_paths() -> None:
    rows = [{
        "id": "doc-1",
        "user_id": "user-1",
        "original_path": r"\\server\share\paper.pdf",
        "translated_path": r"uploads\user-1\doc-1\..\other-doc\paper.pdf",
    }]
    with patch(
        "scripts.migrate_document_paths.list_document_path_rows",
        new=AsyncMock(side_effect=[rows, []]),
    ):
        summary = await migrate_document_paths(apply=False)

    assert summary.changed_fields == 0
    assert summary.rejected_fields == 2


@pytest.mark.asyncio
async def test_migration_uses_batch_size_and_second_portable_run_is_idempotent() -> None:
    portable = [{
        "id": "doc-1",
        "user_id": "user-1",
        "original_path": "uploads/user-1/doc-1/paper.pdf",
        "translated_path": None,
    }]
    list_rows = AsyncMock(side_effect=[portable, []])
    with patch(
        "scripts.migrate_document_paths.list_document_path_rows",
        new=list_rows,
    ):
        summary = await migrate_document_paths(apply=True, batch_size=17)

    assert list_rows.await_args_list[0].kwargs == {"offset": 0, "limit": 17}
    assert list_rows.await_args_list[1].kwargs == {"offset": 17, "limit": 17}
    assert summary.changed_fields == 0
    assert summary.applied_fields == 0
    assert summary.unchanged_fields == 1


@pytest.mark.asyncio
async def test_migration_rejects_out_of_range_batch_sizes() -> None:
    with pytest.raises(ValueError, match="batch_size must be between 1 and 1000"):
        await migrate_document_paths(apply=False, batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be between 1 and 1000"):
        await migrate_document_paths(apply=False, batch_size=1001)
