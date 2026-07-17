import pytest

from evaluation import db as evaluation_db
from evaluation.accounting_schemas import AccountingScopeStart


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
            }
        ],
    )

    assert scope.accounting_schema_version == "2"
    assert len(scope.targets) == 1


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

    assert names == {
        "evaluation_accounting_scopes",
        "evaluation_accounting_scope_targets",
        "evaluation_usage_events",
    }
