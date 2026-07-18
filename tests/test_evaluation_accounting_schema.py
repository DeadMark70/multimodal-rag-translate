from datetime import UTC, datetime

import pytest

from evaluation import db as evaluation_db
from evaluation.accounting_schemas import AccountingScopeStart
from evaluation.accounting_store import EvaluationAccountingStore


def test_execution_scope_requires_one_target() -> None:
    scope = AccountingScopeStart(
        scope_id="scope-1",
        campaign_id="campaign-1",
        scope_type="execution_run",
        scope_key="run-1",
        run_id="run-1",
        metric_name=None,
        targets=[
            {
                "campaign_result_id": None,
                "job_id": "job-1",
                "work_item_id": "work-1",
                "attempt_id": "attempt-1",
                "metric_name": None,
                "mode": "naive",
            }
        ],
    )

    assert scope.accounting_schema_version == "2"
    assert len(scope.targets) == 1
    assert scope.targets[0].mode == "naive"


@pytest.mark.asyncio
async def test_init_db_creates_accounting_tables(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")

    await evaluation_db.force_init_db()

    async with evaluation_db.connect_db() as connection:
        cursor = await connection.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name IN (
                   'evaluation_accounting_scopes',
                   'evaluation_accounting_scope_targets',
                   'evaluation_usage_events'
               )"""
        )
        names = {row["name"] for row in await cursor.fetchall()}
        target_columns = await evaluation_db._table_columns(
            connection, "evaluation_accounting_scope_targets"
        )
        scope_columns = await evaluation_db._table_columns(
            connection, "evaluation_accounting_scopes"
        )

    assert names == {
        "evaluation_accounting_scopes",
        "evaluation_accounting_scope_targets",
        "evaluation_usage_events",
    }
    assert "mode" in target_columns
    assert "retry_count" in scope_columns


@pytest.mark.asyncio
async def test_existing_accounting_targets_gain_nullable_mode_additively(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "legacy.db")
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            CREATE TABLE evaluation_accounting_scope_targets (
                scope_id TEXT NOT NULL,
                campaign_result_id TEXT,
                job_id TEXT NOT NULL,
                work_item_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                metric_name TEXT,
                is_official INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY(scope_id, attempt_id)
            )
            """
        )
        await connection.commit()

    await evaluation_db.force_init_db()

    async with evaluation_db.connect_db() as connection:
        columns = await evaluation_db._table_columns(
            connection, "evaluation_accounting_scope_targets"
        )
        cursor = await connection.execute(
            "PRAGMA table_info(evaluation_accounting_scope_targets)"
        )
        mode_row = next(row for row in await cursor.fetchall() if row[1] == "mode")

    assert "mode" in columns
    assert mode_row[3] == 0


@pytest.mark.asyncio
async def test_legacy_accounting_scope_retry_count_remains_unknown_after_migration(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "legacy.db"
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", database_path)

    legacy_sql = evaluation_db._INIT_SQL.replace(
        "    retry_count INTEGER DEFAULT 0 CHECK (retry_count >= 0),\n", ""
    )
    now = datetime.now(UTC).isoformat()
    async with evaluation_db.connect_db() as connection:
        await connection.executescript(legacy_sql)
        await connection.execute(
            """INSERT INTO campaigns (id, user_id, name, status, config_json, created_at, updated_at)
               VALUES ('campaign-1', 'user-1', 'Legacy', 'completed', '{}', ?, ?)""",
            (now, now),
        )
        await connection.execute(
            """INSERT INTO evaluation_accounting_scopes (
                   scope_id, campaign_id, scope_type, scope_key, accounting_schema_version,
                   status, started_at, created_at, updated_at
               ) VALUES ('legacy-ragas', 'campaign-1', 'ragas_batch', 'legacy', '2',
                         'completed', ?, ?, ?)""",
            (now, now, now),
        )
        await connection.commit()

    await evaluation_db.force_init_db()

    scope = await EvaluationAccountingStore().get_scope("legacy-ragas")

    assert scope.retry_count is None
