"""Tests for durable evaluation usage accounting persistence."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

from evaluation import db as evaluation_db
from evaluation import accounting_store as accounting_store_module
from evaluation.accounting_schemas import AccountingScopeStart, UsageEventCreate
from evaluation.accounting_store import EvaluationAccountingStore


async def seed_campaign(campaign_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """INSERT INTO campaigns (
                   id, user_id, name, status, phase, config_json,
                   completed_units, total_units, evaluation_completed_units,
                   evaluation_total_units, cancel_requested, created_at, updated_at
               ) VALUES (?, 'user-1', 'Accounting test', 'running', 'execution', '{}',
                         0, 1, 0, 0, 0, ?, ?)""",
            (campaign_id, now, now),
        )
        await connection.commit()


@pytest_asyncio.fixture
async def accounting_store(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    await seed_campaign("campaign-1")
    return EvaluationAccountingStore()


def _execution_scope() -> AccountingScopeStart:
    return AccountingScopeStart(
        scope_id="scope-1",
        campaign_id="campaign-1",
        scope_type="execution_run",
        scope_key="run-1",
        run_id="run-1",
        targets=[
            {
                "job_id": "job-1",
                "work_item_id": "work-1",
                "attempt_id": "attempt-1",
                "mode": "naive",
            }
        ],
    )


def _ragas_scope(target_count: int) -> AccountingScopeStart:
    return AccountingScopeStart(
        scope_id="ragas-scope",
        campaign_id="campaign-1",
        scope_type="ragas_batch",
        scope_key="faithfulness:batch-1",
        metric_name="faithfulness",
        targets=[
            {
                "campaign_result_id": f"result-{index}",
                "job_id": "job-1",
                "work_item_id": f"work-{index}",
                "attempt_id": f"attempt-{index}",
                "metric_name": "faithfulness",
            }
            for index in range(target_count)
        ],
    )


def _usage_event(*, phase: str) -> UsageEventCreate:
    return UsageEventCreate(
        usage_event_id=str(uuid4()),
        scope_id="scope-1",
        campaign_id="campaign-1",
        scope_type="execution_run",
        scope_key="run-1",
        run_id="run-1",
        phase=phase,
        purpose="rag_qa",
        provider="google",
        model_name="fake-model",
        input_tokens=10,
        output_text_tokens=5,
        reasoning_tokens=0,
        other_tokens=0,
        reported_total_tokens=15,
        raw_usage={"total_tokens": 15},
        usage_status="measured",
        reconciliation_status="balanced",
        estimated_cost_usd=0.01,
        pricing_status="priced",
        status="success",
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_record_event_updates_scope_counters_atomically(accounting_store):
    await accounting_store.start_scope(_execution_scope())
    await accounting_store.record_event(_usage_event(phase="answer_generation"))
    scope = await accounting_store.get_scope("scope-1")
    assert scope.observed_call_count == 1
    assert scope.measured_call_count == 1
    assert scope.missing_usage_call_count == 0


@pytest.mark.asyncio
async def test_fresh_scope_starts_with_known_zero_retries(accounting_store):
    scope = await accounting_store.start_scope(_ragas_scope(target_count=1))

    assert scope.retry_count == 0


@pytest.mark.asyncio
async def test_ragas_scope_retry_increments_are_durable_and_atomic(accounting_store):
    await accounting_store.start_scope(_ragas_scope(target_count=1))

    await accounting_store.increment_scope_retry("ragas-scope")
    await accounting_store.increment_scope_retry("ragas-scope")

    assert (await accounting_store.get_scope("ragas-scope")).retry_count == 2


@pytest.mark.asyncio
async def test_execution_target_mode_round_trips_durably(accounting_store):
    await accounting_store.start_scope(_execution_scope())

    scope = await accounting_store.get_scope("scope-1")

    assert [target.mode for target in scope.targets] == ["naive"]


@pytest.mark.asyncio
async def test_record_event_is_idempotent_with_its_counter_update(accounting_store):
    await accounting_store.start_scope(_execution_scope())
    event = _usage_event(phase="answer_generation")
    await accounting_store.record_event(event)
    await accounting_store.record_event(event)
    scope = await accounting_store.get_scope("scope-1")
    assert scope.observed_call_count == 1


@pytest.mark.asyncio
async def test_record_event_rejects_scope_metadata_mismatch(accounting_store):
    await accounting_store.start_scope(_execution_scope())
    event = _usage_event(phase="answer_generation")
    event.campaign_id = "other-campaign"

    with pytest.raises(ValueError, match="does not match"):
        await accounting_store.record_event(event)

    scope = await accounting_store.get_scope("scope-1")
    assert scope.observed_call_count == 0
    assert await accounting_store.list_campaign_events("campaign-1") == []


@pytest.mark.asyncio
async def test_mark_targets_official_is_atomic_when_an_attempt_is_unknown(
    accounting_store,
):
    await accounting_store.start_scope(_ragas_scope(target_count=2))

    with pytest.raises(ValueError, match="do not belong"):
        await accounting_store.mark_targets_official(
            "ragas-scope", {"attempt-0": "result-0", "missing-attempt": "result-x"}
        )

    scope = await accounting_store.get_scope("ragas-scope")
    assert all(target.is_official is False for target in scope.targets)


@pytest.mark.asyncio
async def test_ragas_scope_keeps_multiple_targets_without_cost_allocation(
    accounting_store,
):
    scope = _ragas_scope(target_count=3)
    await accounting_store.start_scope(scope)
    stored = await accounting_store.get_scope(scope.scope_id)
    assert [target.attempt_id for target in stored.targets] == [
        "attempt-0",
        "attempt-1",
        "attempt-2",
    ]
    assert all(target.is_official is False for target in stored.targets)


@pytest.mark.asyncio
async def test_failed_callback_with_measured_usage_makes_summary_partial(
    accounting_store,
):
    await accounting_store.start_scope(_execution_scope())
    event = _usage_event(phase="answer_generation")
    event.status = "failed"
    await accounting_store.record_event(event)

    summary = await accounting_store.summarize_scope_tokens("scope-1")

    assert summary.failed_call_count == 1
    assert summary.reconciliation_status == "partial"
    assert (
        summary.as_legacy_usage(accounting_schema_version="2")["total_tokens"] is None
    )


@pytest.mark.asyncio
async def test_missing_only_scope_keeps_legacy_token_categories_unknown(
    accounting_store,
) -> None:
    await accounting_store.start_scope(_execution_scope())
    event = _usage_event(phase="answer_generation")
    event.input_tokens = 0
    event.output_text_tokens = 0
    event.reported_total_tokens = None
    event.usage_status = "missing"
    event.reconciliation_status = "unavailable"
    await accounting_store.record_event(event)

    usage = (await accounting_store.summarize_scope_tokens("scope-1")).as_legacy_usage(
        accounting_schema_version="2"
    )

    assert usage == {
        "accounting_schema_version": "2",
        "input_tokens": None,
        "output_tokens": None,
        "output_text_tokens": None,
        "reasoning_tokens": None,
        "other_tokens": None,
        "total_tokens": None,
    }


@pytest.mark.asyncio
async def test_partial_scope_projects_only_balanced_measured_categories(
    accounting_store,
) -> None:
    await accounting_store.start_scope(_execution_scope())
    await accounting_store.record_event(_usage_event(phase="answer_generation"))
    missing = _usage_event(phase="answer_generation")
    missing.input_tokens = 0
    missing.output_text_tokens = 0
    missing.reported_total_tokens = None
    missing.usage_status = "missing"
    missing.reconciliation_status = "unavailable"
    await accounting_store.record_event(missing)

    summary = await accounting_store.summarize_scope_tokens("scope-1")
    usage = summary.as_legacy_usage(accounting_schema_version="2")

    assert summary.reconciliation_status == "partial"
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5
    assert usage["total_tokens"] is None


@pytest.mark.asyncio
async def test_balanced_zero_usage_remains_known_in_legacy_projection(
    accounting_store,
) -> None:
    await accounting_store.start_scope(_execution_scope())
    event = _usage_event(phase="answer_generation")
    event.input_tokens = 0
    event.output_text_tokens = 0
    event.reported_total_tokens = 0
    await accounting_store.record_event(event)

    usage = (await accounting_store.summarize_scope_tokens("scope-1")).as_legacy_usage(
        accounting_schema_version="2"
    )

    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["output_text_tokens"] == 0
    assert usage["reasoning_tokens"] == 0
    assert usage["other_tokens"] == 0
    assert usage["total_tokens"] == 0


@pytest.mark.asyncio
async def test_list_campaign_scopes_bulk_loads_ordered_targets_in_two_queries(
    accounting_store, monkeypatch
) -> None:
    for scope_number in range(3):
        await accounting_store.start_scope(
            AccountingScopeStart(
                scope_id=f"scope-{scope_number}",
                campaign_id="campaign-1",
                scope_type="ragas_batch",
                scope_key=f"faithfulness:batch-{scope_number}",
                metric_name="faithfulness",
                targets=[
                    {
                        "campaign_result_id": f"result-{scope_number}-{target_number}",
                        "job_id": "job-1",
                        "work_item_id": f"work-{scope_number}-{target_number}",
                        "attempt_id": f"attempt-{target_number}",
                        "metric_name": "faithfulness",
                    }
                    for target_number in (1, 0)
                ],
            )
        )

    statements: list[str] = []
    original_connect_db = accounting_store_module.connect_db

    @asynccontextmanager
    async def traced_connect_db():
        async with original_connect_db() as connection:
            await connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(accounting_store_module, "connect_db", traced_connect_db)

    scopes = await accounting_store.list_campaign_scopes("campaign-1")

    select_statements = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
    ]
    assert len(select_statements) == 2
    assert "FROM evaluation_accounting_scopes" in select_statements[0]
    assert "FROM evaluation_accounting_scope_targets" in select_statements[1]
    assert "WHERE scope_id IN" in select_statements[1]
    assert [scope.scope_id for scope in scopes] == ["scope-0", "scope-1", "scope-2"]
    assert [[target.attempt_id for target in scope.targets] for scope in scopes] == [
        ["attempt-0", "attempt-1"]
    ] * 3
