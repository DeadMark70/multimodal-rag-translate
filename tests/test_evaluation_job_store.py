from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from core.errors import AppError, ErrorCode
import evaluation.db as evaluation_db
from evaluation.error_policy import ErrorDecision
from evaluation.job_schemas import WorkItemSpec


@pytest_asyncio.fixture
async def store(monkeypatch):  # noqa: ANN001
    database_path = (
        Path(__file__).resolve().parent.parent / ".test-artifacts"
        / f"evaluation-ledger-{uuid4().hex}.db"
    )
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", database_path)
    try:
        await evaluation_db.force_init_db()
        async with evaluation_db.connect_db() as connection:
            await connection.execute(
                """
                INSERT INTO campaigns (
                    id, user_id, name, status, config_json, created_at, updated_at
                ) VALUES (?, ?, NULL, ?, '{}', ?, ?)
                """,
                (
                    "cmp-1",
                    "user-a",
                    "pending",
                    "2026-07-14T00:00:00+00:00",
                    "2026-07-14T00:00:00+00:00",
                ),
            )
            await connection.commit()

        from evaluation.job_store import EvaluationJobStore

        yield EvaluationJobStore()
    finally:
        for path in (
            database_path,
            database_path.with_suffix(".db-shm"),
            database_path.with_suffix(".db-wal"),
        ):
            path.unlink(missing_ok=True)
        try:
            database_path.parent.rmdir()
        except OSError:
            pass


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)


def _spec(*, logical_key: str = "execution:Q1:naive:1:none", max_attempts: int = 3) -> WorkItemSpec:
    return WorkItemSpec(
        work_type="dataset_execution",
        logical_key=logical_key,
        input_snapshot={"question_id": "Q1"},
        max_attempts=max_attempts,
    )


@pytest.mark.asyncio
async def test_create_job_reuses_stable_work_item_and_creates_new_job_item(store) -> None:  # noqa: ANN001
    spec = _spec()

    first = await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[spec],
    )
    second = await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={"scope": "all"},
        config_snapshot={},
        items=[spec],
    )

    assert first.id != second.id
    assert first.job_id == first.id
    assert await store.count_work_items(campaign_id="cmp-1") == 1
    assert await store.count_job_items(campaign_id="cmp-1") == 2


@pytest.mark.asyncio
async def test_claim_creates_running_attempt_atomically(store, fixed_now) -> None:  # noqa: ANN001
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec()],
    )

    claimed = await store.claim_ready_items(limit=1, now=fixed_now)

    assert len(claimed) == 1
    assert claimed[0].attempt_number == 1
    assert await store.get_job_item_status(claimed[0].job_item_id) == "running"
    assert await store.claim_ready_items(limit=1, now=fixed_now) == []


@pytest.mark.asyncio
async def test_retryable_failure_requeues_only_below_its_job_item_budget(store, fixed_now) -> None:  # noqa: ANN001
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec(max_attempts=2)],
    )
    first_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]

    first_attempt = await store.fail_attempt(
        first_claim,
        ErrorDecision("timeout", True, None, "The evaluation provider timed out."),
        next_retry_at=fixed_now + timedelta(minutes=1),
    )

    assert first_attempt.status == "failed"
    assert await store.get_job_item_status(first_claim.job_item_id) == "retry_wait"
    assert await store.claim_ready_items(limit=1, now=fixed_now) == []

    second_claim = (
        await store.claim_ready_items(limit=1, now=fixed_now + timedelta(minutes=1))
    )[0]
    second_attempt = await store.fail_attempt(
        second_claim,
        ErrorDecision("timeout", True, None, "The evaluation provider timed out."),
        next_retry_at=fixed_now + timedelta(minutes=2),
    )

    assert second_attempt.status == "failed"
    assert await store.get_job_item_status(second_claim.job_item_id) == "failed"


@pytest.mark.asyncio
async def test_rerun_receives_a_fresh_retry_budget_for_a_reused_work_item(
    store, fixed_now
) -> None:  # noqa: ANN001
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec(max_attempts=1)],
    )
    initial_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    await store.fail_attempt(
        initial_claim,
        ErrorDecision("invalid", False, None, "The evaluation input is invalid."),
        next_retry_at=None,
    )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={},
        config_snapshot={},
        items=[_spec(max_attempts=2)],
    )
    rerun_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]

    assert rerun_claim.attempt_number == 2
    await store.fail_attempt(
        rerun_claim,
        ErrorDecision("timeout", True, None, "The evaluation provider timed out."),
        next_retry_at=fixed_now + timedelta(minutes=1),
    )
    assert await store.get_job_item_status(rerun_claim.job_item_id) == "retry_wait"


@pytest.mark.asyncio
async def test_retryable_failure_without_schedule_uses_a_claimable_default_backoff(
    store, fixed_now
) -> None:  # noqa: ANN001
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec(max_attempts=2)],
    )
    claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    failed = await store.fail_attempt(
        claim,
        ErrorDecision("timeout", True, None, "The evaluation provider timed out."),
        next_retry_at=None,
    )

    assert await store.get_job_item_status(claim.job_item_id) == "retry_wait"
    assert failed.finished_at is not None
    retry_claim = await store.claim_ready_items(
        limit=1, now=failed.finished_at + timedelta(seconds=2)
    )
    assert [item.work_item_id for item in retry_claim] == [claim.work_item_id]


@pytest.mark.asyncio
async def test_claim_excludes_second_non_terminal_job_item_for_same_work_item(
    store, fixed_now
) -> None:  # noqa: ANN001
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec()],
    )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={},
        config_snapshot={},
        items=[_spec()],
    )

    claimed = await store.claim_ready_items(limit=2, now=fixed_now)

    assert len(claimed) == 1
    assert await store.claim_ready_items(limit=2, now=fixed_now) == []


@pytest.mark.asyncio
async def test_cancel_and_startup_recovery_preserve_attempt_history(store, fixed_now) -> None:  # noqa: ANN001
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec(logical_key="execution:Q1:naive:1:cancel")],
    )
    cancelled_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    cancelled_attempt = await store.cancel_attempt(
        cancelled_claim,
        safe_message="Evaluation was cancelled.",
    )

    assert cancelled_attempt.status == "cancelled"
    assert await store.get_job_item_status(cancelled_claim.job_item_id) == "cancelled"

    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec(logical_key="execution:Q1:naive:1:recover")],
    )
    running_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]

    assert await store.recover_interrupted_attempts(at=fixed_now + timedelta(minutes=5)) == 1
    attempts = await store.list_attempts(
        user_id="user-a", work_item_id=running_claim.work_item_id
    )
    assert [attempt.status for attempt in attempts] == ["interrupted"]
    assert await store.get_job_item_status(running_claim.job_item_id) == "pending"


@pytest.mark.asyncio
async def test_additive_migration_adds_result_provenance_columns(store) -> None:  # noqa: ANN001
    async with evaluation_db.connect_db() as connection:
        campaign_result_columns = await evaluation_db._table_columns(connection, "campaign_results")
        ragas_score_columns = await evaluation_db._table_columns(connection, "ragas_scores")

    assert "source_attempt_id" in campaign_result_columns
    assert {"source_attempt_id", "evaluation_signature"} <= ragas_score_columns


@pytest.mark.asyncio
async def test_heartbeat_and_job_read_apis_are_user_scoped(store, fixed_now) -> None:  # noqa: ANN001
    created = await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={"only": "Q1"},
        config_snapshot={"model": "test"},
        items=[_spec()],
    )
    claimed = (await store.claim_ready_items(limit=1, now=fixed_now))[0]

    await store.heartbeat_attempt(
        claimed.attempt_id, at=fixed_now + timedelta(seconds=15)
    )

    assert await store.get_job(user_id="user-a", job_id=created.id) == created
    assert await store.list_jobs(user_id="user-a", campaign_id="cmp-1") == [created]
    with pytest.raises(AppError) as exc_info:
        await store.get_job(user_id="user-b", job_id=created.id)
    assert exc_info.value.code is ErrorCode.NOT_FOUND
    attempts = await store.list_attempts(
        user_id="user-a", work_item_id=claimed.work_item_id
    )
    assert attempts[0].last_heartbeat_at == fixed_now + timedelta(seconds=15)
