"""Runtime integration tests for durable evaluation usage accounting."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest
import pytest_asyncio

from core.llm_usage_context import RawLlmUsageEvent
import evaluation.db as evaluation_db
from evaluation.accounting_runtime import (
    EvaluationAccountingSink,
    start_execution_scope,
)
from evaluation.accounting_store import EvaluationAccountingStore


TEST_PRICE_SNAPSHOT = {
    "snapshot_id": "test-v1",
    "currency": "USD",
    "usd_to_twd": None,
    "models": {
        "test-model": {
            "input_per_1m_usd": 1.0,
            "output_per_1m_usd": 2.0,
            "reasoning_per_1m_usd": 0.0,
        }
    },
}


@pytest_asyncio.fixture
async def accounting_store(
    monkeypatch: pytest.MonkeyPatch,
) -> EvaluationAccountingStore:
    database_path = (
        Path("output") / "test_tmp" / f"accounting-runtime-{uuid4().hex}" / "worker.db"
    )
    database_path.parent.mkdir(parents=True)
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", database_path)
    try:
        await evaluation_db.force_init_db()
        async with evaluation_db.connect_db() as connection:
            now = datetime.now(UTC).isoformat()
            await connection.execute(
                """INSERT INTO campaigns (id, user_id, name, status, config_json, created_at, updated_at)
                   VALUES ('cmp-1', 'user-a', NULL, 'pending', '{}', ?, ?)""",
                (now, now),
            )
            await connection.commit()
        yield EvaluationAccountingStore()
    finally:
        for path in (
            database_path,
            database_path.with_suffix(".db-shm"),
            database_path.with_suffix(".db-wal"),
        ):
            path.unlink(missing_ok=True)
        rmtree(database_path.parent, ignore_errors=True)


@pytest.mark.asyncio
async def test_sink_normalizes_prices_and_preserves_callback_event_id(
    accounting_store: EvaluationAccountingStore,
) -> None:
    store = accounting_store
    scope = await start_execution_scope(
        store=store,
        campaign_id="cmp-1",
        run_id="run-1",
        job_id="job-1",
        work_item_id="item-1",
        attempt_id="attempt-1",
        mode="naive",
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )
    stored_scope = await store.get_scope(scope.scope_id)
    assert [target.mode for target in stored_scope.targets] == ["naive"]
    raw = RawLlmUsageEvent(
        usage_event_id="event-1",
        scope_id=scope.scope_id,
        campaign_id="cmp-1",
        scope_type="execution_run",
        scope_key="run-1",
        run_id="run-1",
        provider_run_id=None,
        phase="answer_generation",
        purpose="rag_qa",
        metric_name=None,
        provider="google",
        model_name="test-model",
        raw_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        latency_ms=12.5,
        status="success",
        error={},
        created_at=datetime.now(UTC),
    )

    await scope.context.sink.record(raw)
    await scope.context.sink.record(raw)

    events = await store.list_campaign_events("cmp-1")
    assert len(events) == 1
    assert events[0].usage_event_id == "event-1"
    assert events[0].estimated_cost_usd == pytest.approx(0.00002)
    assert events[0].pricing_status == "priced"


@pytest.mark.asyncio
async def test_sink_keeps_partial_usage_unpriced(
    accounting_store: EvaluationAccountingStore,
) -> None:
    store = accounting_store
    scope = await start_execution_scope(
        store=store,
        campaign_id="cmp-1",
        run_id="run-1",
        job_id="job-1",
        work_item_id="item-1",
        attempt_id="attempt-1",
        mode="naive",
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )
    sink = EvaluationAccountingSink(store=store, price_snapshot=TEST_PRICE_SNAPSHOT)

    await sink.record(
        RawLlmUsageEvent(
            usage_event_id="event-1",
            scope_id=scope.scope_id,
            campaign_id="cmp-1",
            scope_type="execution_run",
            scope_key="run-1",
            run_id="run-1",
            provider_run_id=None,
            phase="answer_generation",
            purpose="rag_qa",
            metric_name=None,
            provider="google",
            model_name="test-model",
            raw_usage={"input_tokens": 10},
            latency_ms=None,
            status="success",
            error={},
            created_at=datetime.now(UTC),
        )
    )

    event = (await store.list_campaign_events("cmp-1"))[0]
    assert event.estimated_cost_usd is None
    assert event.pricing_status == "unavailable_usage"
