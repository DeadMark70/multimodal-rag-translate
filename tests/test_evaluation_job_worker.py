"""Lifecycle coverage for the single-process durable evaluation worker."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

import evaluation.db as evaluation_db
from evaluation.job_schemas import WorkItemSpec
from evaluation.job_store import EvaluationJobStore
from evaluation.job_worker import EvaluationJobWorker


@pytest_asyncio.fixture
async def store(
    monkeypatch: pytest.MonkeyPatch,
) -> EvaluationJobStore:
    database_path = Path(mkdtemp(prefix="evaluation-worker-")) / "worker.db"
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", database_path)
    try:
        await evaluation_db.force_init_db()
        async with evaluation_db.connect_db() as connection:
            now = "2026-07-14T00:00:00+00:00"
            await connection.execute(
                """
                INSERT INTO campaigns (id, user_id, name, status, config_json, created_at, updated_at)
                VALUES ('cmp-1', 'user-a', NULL, 'pending', '{}', ?, ?)
                """,
                (now, now),
            )
            await connection.commit()
        yield EvaluationJobStore()
    finally:
        for path in (database_path, database_path.with_suffix(".db-shm"), database_path.with_suffix(".db-wal")):
            path.unlink(missing_ok=True)
        rmtree(database_path.parent, ignore_errors=True)


async def _seed_work(
    store: EvaluationJobStore,
    *,
    logical_key: str,
    max_attempts: int = 2,
) -> None:
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[
            WorkItemSpec(
                work_type="dataset_execution",
                logical_key=logical_key,
                input_snapshot={"question_id": logical_key.split(":")[1]},
                max_attempts=max_attempts,
            )
        ],
    )


async def _seed_running_attempt(store: EvaluationJobStore, *, logical_key: str) -> None:
    await _seed_work(store, logical_key=logical_key)
    claimed = await store.claim_ready_items(
        limit=1, now=datetime(2026, 7, 14, tzinfo=timezone.utc)
    )
    assert len(claimed) == 1


async def _seed_successful_work(store: EvaluationJobStore, *, logical_key: str) -> None:
    await _seed_work(store, logical_key=logical_key, max_attempts=1)
    claim = (
        await store.claim_ready_items(limit=1, now=datetime(2026, 7, 14, tzinfo=timezone.utc))
    )[0]
    await store.cancel_attempt(claim, safe_message="Fixture success substitute")


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


@pytest.mark.asyncio
async def test_start_recovers_running_attempt_and_executes_only_unresolved_work(
    store: EvaluationJobStore,
) -> None:
    await _seed_running_attempt(store, logical_key="execution:Q1:naive:1:none")
    await _seed_successful_work(store, logical_key="execution:Q2:naive:1:none")
    executor = AsyncMock()
    worker = EvaluationJobWorker(store=store, execution_handler=executor, ragas_handler=AsyncMock())

    await worker.start()
    worker.notify()
    await _wait_until(lambda: executor.await_count == 1)
    await worker.stop()

    assert executor.await_args.args[0].logical_key == "execution:Q1:naive:1:none"


@pytest.mark.asyncio
async def test_stop_does_not_claim_new_work(store: EvaluationJobStore) -> None:
    worker = EvaluationJobWorker(store=store, execution_handler=AsyncMock(), ragas_handler=AsyncMock())

    await worker.start()
    await worker.stop()
    await _seed_work(store, logical_key="execution:Q1:naive:1:none")
    worker.notify()

    assert await worker.run_once() == 0


@pytest.mark.asyncio
async def test_execution_dispatch_never_exceeds_its_limit(store: EvaluationJobStore) -> None:
    for number in range(5):
        await _seed_work(store, logical_key=f"execution:Q{number}:naive:1:none")
    release = asyncio.Event()
    active = 0
    peak = 0

    async def execute(_claim) -> None:  # noqa: ANN001
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await release.wait()
        active -= 1

    worker = EvaluationJobWorker(store=store, execution_handler=execute, ragas_handler=AsyncMock())
    await worker.start()
    worker.notify()
    await _wait_until(lambda: active == 4)
    release.set()
    await worker.stop()

    assert peak == 4


@pytest.mark.asyncio
async def test_stop_cancels_active_handler_and_recovers_its_attempt(store: EvaluationJobStore) -> None:
    await _seed_work(store, logical_key="execution:Q1:naive:1:none")
    started = asyncio.Event()
    cancelled = asyncio.Event()
    work_item_ids: list[str] = []

    async def execute(_claim) -> None:  # noqa: ANN001
        work_item_ids.append(_claim.work_item_id)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    worker = EvaluationJobWorker(store=store, execution_handler=execute, ragas_handler=AsyncMock())
    await worker.start()
    worker.notify()
    await asyncio.wait_for(started.wait(), timeout=1)
    await worker.stop()

    assert cancelled.is_set()
    assert len(work_item_ids) == 1
    attempts = await store.list_attempts(user_id="user-a", work_item_id=work_item_ids[0])
    assert [attempt.status for attempt in attempts] == ["interrupted"]
    claim = (await store.claim_ready_items(limit=1, now=datetime.now(timezone.utc)))[0]
    assert claim.attempt_number == 2
