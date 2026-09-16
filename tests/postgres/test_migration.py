"""The offline importer preserves source bytes and rolls back partial imports."""

import hashlib
from pathlib import Path
import sqlite3

from psycopg.errors import ForeignKeyViolation
import pytest

from evaluation.postgres import connect_db
from scripts.migrate_evaluation import import_sqlite


def source_database(path: Path, *, orphan: bool = False) -> Path:
    schema = Path(__file__).resolve().parents[2] / "evaluation/migrations/001_initial.sql"
    with sqlite3.connect(path) as source:
        source.executescript(schema.read_text(encoding="utf-8"))
        for key in ("z", "A", "中文"):
            source.execute(
                "INSERT INTO campaigns(id,user_id,status,config_json,created_at,updated_at) "
                "VALUES (?, 'owner','completed','{}','2026-09-16','2026-09-16')", (key,),
            )
        if orphan:
            source.execute(
                "INSERT INTO evaluation_jobs VALUES ('job','owner','absent','initial','{}','{}','2026-09-16')"
            )
    return path


@pytest.mark.asyncio
async def test_verified_import_preserves_source_and_rejects_overwrite(tmp_path):
    path = source_database(tmp_path / "source.db")
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    report = await import_sqlite(path)
    assert report["verified"]
    assert len(report["tables"]) == 25
    assert report["tables"]["campaigns"]["rows"] == 3
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    with pytest.raises(ValueError, match="not empty"):
        await import_sqlite(path)
    async with connect_db() as conn:
        row = await (await conn.execute("SELECT COUNT(*) AS n FROM campaigns")).fetchone()
    assert row["n"] == 3


@pytest.mark.asyncio
async def test_invalid_child_rolls_back_parent_rows(tmp_path):
    path = source_database(tmp_path / "orphan.db", orphan=True)
    with pytest.raises(ForeignKeyViolation):
        await import_sqlite(path)
    async with connect_db() as conn:
        row = await (await conn.execute("SELECT COUNT(*) AS n FROM campaigns")).fetchone()
    assert row["n"] == 0
