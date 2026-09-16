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

    await evaluation_db.force_init_db()

    async with evaluation_db.connect_db() as connection:
        cursor = await connection.execute(
            """SELECT table_name AS name FROM information_schema.tables
               WHERE table_schema=current_schema() AND table_name IN (
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
async def test_accounting_target_mode_is_nullable(
    tmp_path, monkeypatch
) -> None:
    await evaluation_db.force_init_db()

    async with evaluation_db.connect_db() as connection:
        columns = await evaluation_db._table_columns(
            connection, "evaluation_accounting_scope_targets"
        )
        cursor = await connection.execute(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_schema=current_schema() "
            "AND table_name='evaluation_accounting_scope_targets' AND column_name='mode'"
        )
        mode_row = await cursor.fetchone()

    assert "mode" in columns
    assert mode_row["is_nullable"] == "YES"


@pytest.mark.asyncio
async def test_unknown_accounting_retry_count_survives_reinitialization(
    tmp_path, monkeypatch
) -> None:
    now = datetime.now(UTC).isoformat()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """INSERT INTO campaigns (id, user_id, name, status, config_json, created_at, updated_at)
               VALUES ('campaign-1', 'user-1', 'Legacy', 'completed', '{}', ?, ?)""",
            (now, now),
        )
        await connection.execute(
            """INSERT INTO evaluation_accounting_scopes (
                   scope_id, campaign_id, scope_type, scope_key, accounting_schema_version,
                   status, started_at, created_at, updated_at, retry_count
               ) VALUES ('legacy-ragas', 'campaign-1', 'ragas_batch', 'legacy', '2',
                         'completed', ?, ?, ?, NULL)""",
            (now, now, now),
        )
        await connection.commit()

    await evaluation_db.force_init_db()

    scope = await EvaluationAccountingStore().get_scope("legacy-ragas")

    assert scope.retry_count is None
