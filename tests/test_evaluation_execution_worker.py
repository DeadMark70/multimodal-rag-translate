"""Durable dataset execution checkpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_asyncio
from langchain_core.documents import Document

import evaluation.db as evaluation_db
from core.llm_usage_callback import emit_direct_usage
from core.llm_usage_context import llm_accounting_phase
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.execution_worker import DatasetExecutionWorker
from evaluation.analytics import EvaluationAnalyticsService
from evaluation.agentic_v9_campaign_runtime import AgenticV9CampaignRuntime
from evaluation.job_schemas import (
    ClaimedEvaluationWork,
    EvaluationWorkType,
    WorkItemSpec,
)
from evaluation.job_store import EvaluationJobStore
from evaluation.rag_modes import BenchmarkExecutionResult
from evaluation.retrieval_profiles import (
    ADVANCED_EVAL_PROFILE,
    AGENTIC_EVAL_PROFILE,
)


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
async def store(
    monkeypatch: pytest.MonkeyPatch,
) -> EvaluationJobStore:
    database_path = (
        Path(os.environ["EVALUATION_TEST_TMPDIR"])
        / f"dataset-execution-{uuid4().hex}"
        / "worker.db"
    )
    database_path.parent.mkdir(parents=True)
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", database_path)
    try:
        await evaluation_db.force_init_db()
        async with evaluation_db.connect_db() as connection:
            now = "2026-07-14T00:00:00+00:00"
            config = json.dumps(
                {
                    "test_case_ids": ["Q1"],
                    "modes": ["naive"],
                    "model_config": {
                        "id": "cfg-1",
                        "name": "test",
                        "model_name": "test-model",
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1,
                        "max_input_tokens": 1,
                        "max_output_tokens": 1,
                        "thinking_mode": False,
                        "thinking_budget": 0,
                    },
                }
            )
            await connection.execute(
                """
                INSERT INTO campaigns (id, user_id, name, status, config_json, created_at, updated_at)
                VALUES ('cmp-1', 'user-a', NULL, 'pending', ?, ?, ?)
                """,
                (config, now, now),
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


async def _claim_seeded_execution(
    store: EvaluationJobStore,
    *,
    mode: str = "naive",
    agentic_execution_version: str = "v8",
    source_docs: list[str] | None = None,
    model_config: dict | None = None,
):  # noqa: ANN202
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[
            WorkItemSpec(
                work_type=EvaluationWorkType.DATASET_EXECUTION,
                logical_key=f"execution:Q1:{mode}:1:none",
                input_snapshot={
                    "user_id": "user-a",
                    "campaign_id": "cmp-1",
                    "test_case": {
                        "id": "Q1",
                        "question": "What is the answer?",
                        "ground_truth": "42",
                        "source_docs": source_docs or [],
                        "requires_multi_doc_reasoning": False,
                    },
                    "mode": mode,
                    "run_number": 1,
                    "repeat_number": 1,
                    "model_config": model_config or {},
                    "agentic_execution_version": agentic_execution_version,
                },
            )
        ],
    )
    return (await store.claim_ready_items(limit=1, now=datetime.now(timezone.utc)))[0]


def _successful_payload(
    *, question_id: str, mode: str, answer: str
) -> BenchmarkExecutionResult:
    return BenchmarkExecutionResult(
        question_id=question_id,
        question="What is the answer?",
        ground_truth="42",
        mode=mode,
        answer=answer,
    )


@pytest.mark.asyncio
async def test_derive_campaign_state_notifies_materialized_ragas_without_late_transition() -> (
    None
):
    campaign_repository = SimpleNamespace(
        derive_execution_state=AsyncMock(
            return_value=SimpleNamespace(
                status=SimpleNamespace(value="completed"), config=None
            )
        ),
        mark_evaluating=AsyncMock(),
    )
    store = SimpleNamespace(
        get_job=AsyncMock(return_value=SimpleNamespace(config_snapshot={})),
        ensure_ragas_work=AsyncMock(return_value=1),
    )
    notify = Mock()
    worker = DatasetExecutionWorker(
        store=store,
        campaign_repository=campaign_repository,
        ragas_evaluator=SimpleNamespace(
            enabled_metrics=("faithfulness",), evaluator_model="judge-v1"
        ),
        notify=notify,
    )
    claim = ClaimedEvaluationWork(
        job_id="job-1",
        job_item_id="job-item-1",
        work_item_id="work-item-1",
        attempt_id="attempt-1",
        input_snapshot={"user_id": "user-a", "campaign_id": "cmp-1"},
    )

    await worker._derive_campaign_state(claim)

    campaign_repository.mark_evaluating.assert_not_awaited()
    notify.assert_called_once()


@pytest.mark.asyncio
async def test_execution_worker_promotes_ledger_total_not_payload_total(
    store: EvaluationJobStore,
) -> None:
    async def runner(**runtime_inputs):  # noqa: ANN003
        for phase, usage in [
            (
                "query_expansion",
                {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            ),
            (
                "answer_generation",
                {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            ),
        ]:
            with llm_accounting_phase(phase):
                await emit_direct_usage(
                    purpose="rag_qa",
                    provider="google",
                    model_name="test-model",
                    raw_usage=usage,
                    status="success",
                    error={},
                )
        payload = _successful_payload(
            question_id=runtime_inputs["test_case"].id,
            mode=runtime_inputs["mode"],
            answer="42",
        )
        payload.token_usage = {"total_tokens": 999}
        return payload

    worker = DatasetExecutionWorker(
        store=store,
        runner=runner,
        accounting_store=EvaluationAccountingStore(),
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )
    await worker.execute(await _claim_seeded_execution(store))

    results = await evaluation_db.CampaignResultRepository().list_for_campaign(
        user_id="user-a", campaign_id="cmp-1"
    )
    assert results[0].total_tokens == 37
    assert results[0].token_usage["accounting_schema_version"] == "2"


@pytest.mark.asyncio
async def test_failed_attempt_scope_is_not_official(store: EvaluationJobStore) -> None:
    accounting_store = EvaluationAccountingStore()
    worker = DatasetExecutionWorker(
        store=store,
        runner=AsyncMock(side_effect=RuntimeError("provider failed")),
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    await worker.execute(await _claim_seeded_execution(store))

    scope = (await accounting_store.list_campaign_scopes("cmp-1"))[0]
    assert scope.status == "failed"
    assert all(not target.is_official for target in scope.targets)
    assert [target.mode for target in scope.targets] == ["naive"]
    assert (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
        == []
    )


@pytest.mark.asyncio
async def test_cancelled_attempt_scope_is_cancelled(store: EvaluationJobStore) -> None:
    async def runner(**_runtime_inputs):  # noqa: ANN003
        raise asyncio.CancelledError

    accounting_store = EvaluationAccountingStore()
    worker = DatasetExecutionWorker(
        store=store,
        runner=runner,
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    with pytest.raises(asyncio.CancelledError):
        await worker.execute(await _claim_seeded_execution(store))

    scope = (await accounting_store.list_campaign_scopes("cmp-1"))[0]
    assert scope.status == "cancelled"
    assert (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
        == []
    )


@pytest.mark.asyncio
async def test_usage_persistence_error_keeps_completed_scope_and_partial_result(
    store: EvaluationJobStore,
) -> None:
    class FailingEventStore(EvaluationAccountingStore):
        async def record_event(self, event) -> None:  # noqa: ANN001
            raise RuntimeError("accounting unavailable")

    async def runner(**runtime_inputs):  # noqa: ANN003
        await emit_direct_usage(
            purpose="rag_qa",
            provider="google",
            model_name="test-model",
            raw_usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )
        return _successful_payload(
            question_id=runtime_inputs["test_case"].id,
            mode=runtime_inputs["mode"],
            answer="42",
        )

    accounting_store = FailingEventStore()
    worker = DatasetExecutionWorker(
        store=store,
        runner=runner,
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )
    await worker.execute(await _claim_seeded_execution(store))

    scope = (await accounting_store.list_campaign_scopes("cmp-1"))[0]
    result = (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
    )[0]
    assert scope.status == "completed"
    assert result.token_usage["token_accounting_status"] == "partial"


@pytest.mark.asyncio
async def test_missing_usage_does_not_project_synthetic_zero_tokens(
    store: EvaluationJobStore,
) -> None:
    async def runner(**runtime_inputs):  # noqa: ANN003
        return _successful_payload(
            question_id=runtime_inputs["test_case"].id,
            mode=runtime_inputs["mode"],
            answer="42",
        )

    worker = DatasetExecutionWorker(
        store=store,
        runner=runner,
        accounting_store=EvaluationAccountingStore(),
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )
    await worker.execute(await _claim_seeded_execution(store))

    result = (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
    )[0]
    assert result.total_tokens is None
    assert result.token_usage["total_tokens"] is None
    assert "input_tokens" not in result.token_usage
    assert result.token_usage["token_accounting_status"] == "unavailable"


@pytest.mark.asyncio
async def test_failed_unit_records_attempt_without_failed_official_result(
    store: EvaluationJobStore,
) -> None:
    runner = AsyncMock(side_effect=RuntimeError("temporary outage"))
    execution_worker = DatasetExecutionWorker(store=store, runner=runner)
    claim = await _claim_seeded_execution(store)

    await execution_worker.execute(claim)

    attempts = await store.list_attempts(
        user_id="user-a", work_item_id=claim.work_item_id
    )
    assert attempts[-1].status.value == "failed"
    assert (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
        == []
    )


@pytest.mark.parametrize(
    ("mode", "expected_profile"),
    [
        ("advanced", ADVANCED_EVAL_PROFILE),
        ("agentic", AGENTIC_EVAL_PROFILE),
    ],
)
@pytest.mark.asyncio
async def test_failed_durable_execution_persists_current_evaluation_profile(
    store: EvaluationJobStore,
    mode: str,
    expected_profile: str,
) -> None:
    payload = _successful_payload(question_id="Q1", mode=mode, answer="")
    payload.error_message = "retrieval failed"
    worker = DatasetExecutionWorker(store=store, runner=AsyncMock(return_value=payload))
    claim = await _claim_seeded_execution(store, mode=mode)

    await worker.execute(claim)

    results = await evaluation_db.CampaignResultRepository().list_for_campaign(
        user_id="user-a", campaign_id="cmp-1"
    )
    assert len(results) == 1
    assert results[0].status.value == "failed"
    assert results[0].execution_profile == expected_profile


@pytest.mark.asyncio
async def test_failed_durable_execution_prefers_captured_trace_profile(
    store: EvaluationJobStore,
) -> None:
    payload = _successful_payload(question_id="Q1", mode="agentic", answer="")
    payload.error_message = "agentic synthesis failed"
    payload.agent_trace = {"execution_profile": "captured-agentic-profile"}
    worker = DatasetExecutionWorker(store=store, runner=AsyncMock(return_value=payload))
    claim = await _claim_seeded_execution(store, mode="agentic")

    await worker.execute(claim)

    results = await evaluation_db.CampaignResultRepository().list_for_campaign(
        user_id="user-a", campaign_id="cmp-1"
    )
    assert len(results) == 1
    assert results[0].execution_profile == "captured-agentic-profile"


@pytest.mark.asyncio
async def test_ablation_conditions_with_a_shared_mode_keep_distinct_official_results(
    store: EvaluationJobStore,
) -> None:
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[
            WorkItemSpec(
                work_type=EvaluationWorkType.DATASET_EXECUTION,
                logical_key=f"execution:Q1:naive:1:{condition_id}",
                input_snapshot={
                    "user_id": "user-a",
                    "campaign_id": "cmp-1",
                    "test_case": {
                        "id": "Q1",
                        "question": "What is the answer?",
                        "ground_truth": "42",
                        "source_docs": [],
                        "requires_multi_doc_reasoning": False,
                    },
                    "mode": "naive",
                    "run_number": 1,
                    "repeat_number": 1,
                    "condition_id": condition_id,
                    "condition_label": condition_id,
                    "ablation_flags": {"condition": condition_id},
                    "model_config": {},
                },
            )
            for condition_id in ("without_graph", "with_graph")
        ],
    )
    claims = await store.claim_ready_items(limit=2, now=datetime.now(timezone.utc))
    assert len(claims) == 2

    async def runner(**kwargs):  # noqa: ANN003
        condition_id = kwargs["ablation_flags"]["condition"]
        return _successful_payload(
            question_id=kwargs["test_case"].id,
            mode=kwargs["mode"],
            answer=condition_id,
        )

    worker = DatasetExecutionWorker(store=store, runner=runner)
    for claim in claims:
        await worker.execute(claim)

    results = await evaluation_db.CampaignResultRepository().list_for_campaign(
        user_id="user-a", campaign_id="cmp-1"
    )
    assert len(results) == 2
    assert {result.condition_id for result in results} == {
        "without_graph",
        "with_graph",
    }
    campaign = await evaluation_db.CampaignRepository().get(
        user_id="user-a", campaign_id="cmp-1"
    )
    assert campaign.status.value == "completed"


@pytest.mark.asyncio
async def test_worker_completion_after_campaign_cancellation_exits_cleanly(
    store: EvaluationJobStore,
) -> None:
    claim = await _claim_seeded_execution(store)
    worker = DatasetExecutionWorker(
        store=store,
        runner=AsyncMock(
            return_value=_successful_payload(
                question_id="Q1", mode="naive", answer="42"
            )
        ),
    )
    await store.cancel_campaign_jobs(user_id="user-a", campaign_id="cmp-1")

    await worker.execute(claim)

    attempts = await store.list_attempts(
        user_id="user-a", work_item_id=claim.work_item_id
    )
    assert attempts[-1].status.value == "cancelled"
    assert await store.get_job_item_status(claim.job_item_id) == "cancelled"
    assert (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
        == []
    )


@pytest.mark.asyncio
async def test_v9_worker_materializes_the_real_core_trace_for_run_detail(
    store: EvaluationJobStore,
) -> None:
    provider = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=SimpleNamespace(
                content="The source reports 0.91.",
                usage_metadata={"input_tokens": 10, "output_tokens": 5},
            )
        )
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="The source reports 0.91.",
                    metadata={"doc_id": "doc-1", "page_number": 1},
                )
            ]
        ),
        provider_factory=lambda _purpose: provider,
    )

    async def runner(**kwargs):  # noqa: ANN003
        v9 = await runtime.execute(
            question=kwargs["test_case"].question,
            user_id=kwargs["user_id"],
            authorized_doc_ids=list(kwargs["test_case"].source_docs),
            setup_snapshot=kwargs["model_config"],
            trace_id="worker-v9-trace",
        )
        return BenchmarkExecutionResult(
            question_id=kwargs["test_case"].id,
            question=kwargs["test_case"].question,
            ground_truth=kwargs["test_case"].ground_truth,
            mode=kwargs["mode"],
            answer=v9.answer,
            contexts=[document.page_content for document in v9.documents],
            source_doc_ids=v9.source_doc_ids,
            agent_trace=v9.agent_trace,
            agentic_execution_version="v9",
        )

    worker = DatasetExecutionWorker(store=store, runner=runner)
    claim = await _claim_seeded_execution(
        store,
        mode="agentic",
        agentic_execution_version="v9",
        source_docs=["doc-1"],
        model_config={
            "max_input_tokens": 4096,
            "max_output_tokens": 256,
            "thinking_mode": False,
        },
    )
    await worker.execute(claim)

    result = (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
    )[0]
    detail = await EvaluationAnalyticsService().run_detail(
        user_id="user-a", run_id=result.id
    )
    assert detail.agentic_v9 is not None
    assert detail.agentic_v9.contract is not None
    assert detail.agentic_v9.evidence_packets
    assert detail.agentic_v9.slot_resolutions
