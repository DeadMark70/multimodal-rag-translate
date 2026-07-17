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
from evaluation.accounting_runtime import start_execution_scope
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.job_schemas import EvaluationWorkType, WorkItemSpec
from evaluation.job_store import EvaluationJobStore
from evaluation.job_worker import EvaluationJobWorker, get_evaluation_job_worker


@pytest_asyncio.fixture
async def store(
    monkeypatch: pytest.MonkeyPatch,
) -> EvaluationJobStore:
    artifacts_dir = Path(__file__).resolve().parent.parent / ".test-artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    database_path = (
        Path(mkdtemp(prefix="evaluation-worker-", dir=artifacts_dir)) / "worker.db"
    )
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
        for path in (
            database_path,
            database_path.with_suffix(".db-shm"),
            database_path.with_suffix(".db-wal"),
        ):
            path.unlink(missing_ok=True)
        rmtree(database_path.parent, ignore_errors=True)


async def _seed_work(
    store: EvaluationJobStore,
    *,
    logical_key: str,
    work_type: EvaluationWorkType = EvaluationWorkType.DATASET_EXECUTION,
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
                work_type=work_type,
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
        await store.claim_ready_items(
            limit=1, now=datetime(2026, 7, 14, tzinfo=timezone.utc)
        )
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
    worker = EvaluationJobWorker(
        store=store, execution_handler=executor, ragas_handler=AsyncMock()
    )

    await worker.start()
    worker.notify()
    await _wait_until(lambda: executor.await_count == 1)
    await worker.stop()

    assert executor.await_args.args[0].logical_key == "execution:Q1:naive:1:none"


@pytest.mark.asyncio
async def test_start_interrupts_running_accounting_scopes_before_recovery(
    store: EvaluationJobStore,
) -> None:
    await _seed_work(store, logical_key="execution:Q1:naive:1:none")
    claim = (
        await store.claim_ready_items(
            limit=1, now=datetime(2026, 7, 14, tzinfo=timezone.utc)
        )
    )[0]
    accounting_store = EvaluationAccountingStore()
    scope = await start_execution_scope(
        store=accounting_store,
        campaign_id="cmp-1",
        run_id="run-1",
        job_id=claim.job_id,
        work_item_id=claim.work_item_id,
        attempt_id=claim.attempt_id,
    )
    worker = EvaluationJobWorker(
        store=store,
        execution_handler=AsyncMock(),
        ragas_handler=AsyncMock(),
        accounting_store=accounting_store,
    )

    await worker.start()
    await worker.stop()

    assert (await accounting_store.get_scope(scope.scope_id)).status == "interrupted"


@pytest.mark.asyncio
async def test_stop_does_not_claim_new_work(store: EvaluationJobStore) -> None:
    worker = EvaluationJobWorker(
        store=store, execution_handler=AsyncMock(), ragas_handler=AsyncMock()
    )

    await worker.start()
    await worker.stop()
    await _seed_work(store, logical_key="execution:Q1:naive:1:none")
    worker.notify()

    assert await worker.run_once() == 0


@pytest.mark.asyncio
async def test_start_restarts_idle_loop_before_claiming_new_work(
    store: EvaluationJobStore,
) -> None:
    """Work arriving as an idle loop shuts down must still be claimed."""
    executor = AsyncMock()
    worker = EvaluationJobWorker(
        store=store,
        execution_handler=executor,
        ragas_handler=AsyncMock(),
        stop_when_idle=True,
    )

    await worker.start()
    await _wait_until(
        lambda: worker._loop_task is not None and worker._loop_task.done()
    )
    await _seed_work(store, logical_key="execution:Q1:naive:1:none")

    await worker.start()
    worker.notify()
    await _wait_until(lambda: executor.await_count == 1)
    await worker.stop()


@pytest.mark.asyncio
async def test_execution_dispatch_never_exceeds_its_limit(
    store: EvaluationJobStore,
) -> None:
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

    worker = EvaluationJobWorker(
        store=store, execution_handler=execute, ragas_handler=AsyncMock()
    )
    await worker.start()
    worker.notify()
    await _wait_until(lambda: active == 4)
    release.set()
    await worker.stop()

    assert peak == 4


@pytest.mark.asyncio
async def test_mixed_dispatch_claims_four_execution_and_two_ragas_items(
    store: EvaluationJobStore,
) -> None:
    for number in range(5):
        await _seed_work(store, logical_key=f"execution:Q{number}:naive:1:none")
    for number in range(3):
        await _seed_work(
            store,
            logical_key=f"ragas:R{number}:faithfulness:signature",
            work_type=EvaluationWorkType.RAGAS_METRIC,
        )

    release = asyncio.Event()
    execution_active = 0
    ragas_active = 0

    async def execute(_claim) -> None:  # noqa: ANN001
        nonlocal execution_active
        execution_active += 1
        await release.wait()
        execution_active -= 1

    async def evaluate_ragas(_claim) -> None:  # noqa: ANN001
        nonlocal ragas_active
        ragas_active += 1
        await release.wait()
        ragas_active -= 1

    worker = EvaluationJobWorker(
        store=store,
        execution_handler=execute,
        ragas_handler=evaluate_ragas,
    )

    await worker.start()
    try:
        await _wait_until(lambda: execution_active == 4 and ragas_active == 2)
    finally:
        release.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_unconfigured_singleton_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluation.job_worker as job_worker

    monkeypatch.setattr(job_worker, "_worker", None)
    worker = get_evaluation_job_worker()

    with pytest.raises(RuntimeError, match="handlers"):
        await worker.start()


@pytest.mark.asyncio
async def test_stop_cancels_active_handler_and_recovers_its_attempt(
    store: EvaluationJobStore,
) -> None:
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

    worker = EvaluationJobWorker(
        store=store, execution_handler=execute, ragas_handler=AsyncMock()
    )
    await worker.start()
    worker.notify()
    await asyncio.wait_for(started.wait(), timeout=1)
    await worker.stop()

    assert cancelled.is_set()
    assert len(work_item_ids) == 1
    attempts = await store.list_attempts(
        user_id="user-a", work_item_id=work_item_ids[0]
    )
    assert [attempt.status for attempt in attempts] == ["interrupted"]
    claim = (await store.claim_ready_items(limit=1, now=datetime.now(timezone.utc)))[0]
    assert claim.attempt_number == 2
