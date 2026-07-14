"""Durable dataset execution checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from shutil import rmtree
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

import evaluation.db as evaluation_db
from evaluation.execution_worker import DatasetExecutionWorker
from evaluation.job_schemas import EvaluationWorkType, WorkItemSpec
from evaluation.job_store import EvaluationJobStore
from evaluation.rag_modes import BenchmarkExecutionResult


@pytest_asyncio.fixture
async def store(
    monkeypatch: pytest.MonkeyPatch,
) -> EvaluationJobStore:
    database_path = Path("output") / "test_tmp" / f"dataset-execution-{uuid4().hex}" / "worker.db"
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
                        "id": "cfg-1", "name": "test", "model_name": "test-model",
                        "temperature": 0, "top_p": 1, "top_k": 1,
                        "max_input_tokens": 1, "max_output_tokens": 1,
                        "thinking_mode": False, "thinking_budget": 0,
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
        for path in (database_path, database_path.with_suffix(".db-shm"), database_path.with_suffix(".db-wal")):
            path.unlink(missing_ok=True)
        rmtree(database_path.parent, ignore_errors=True)


async def _claim_seeded_execution(store: EvaluationJobStore):  # noqa: ANN202
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[
            WorkItemSpec(
                work_type=EvaluationWorkType.DATASET_EXECUTION,
                logical_key="execution:Q1:naive:1:none",
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
                    "model_config": {},
                },
            )
        ],
    )
    return (await store.claim_ready_items(limit=1, now=datetime.now(timezone.utc)))[0]


def _successful_payload(*, question_id: str, mode: str, answer: str) -> BenchmarkExecutionResult:
    return BenchmarkExecutionResult(
        question_id=question_id,
        question="What is the answer?",
        ground_truth="42",
        mode=mode,
        answer=answer,
    )


@pytest.mark.asyncio
async def test_failed_unit_records_attempt_without_failed_official_result(
    store: EvaluationJobStore,
) -> None:
    runner = AsyncMock(side_effect=RuntimeError("temporary outage"))
    execution_worker = DatasetExecutionWorker(store=store, runner=runner)
    claim = await _claim_seeded_execution(store)

    await execution_worker.execute(claim)

    attempts = await store.list_attempts(user_id="user-a", work_item_id=claim.work_item_id)
    assert attempts[-1].status.value == "failed"
    assert await evaluation_db.CampaignResultRepository().list_for_campaign(
        user_id="user-a", campaign_id="cmp-1"
    ) == []


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
    campaign = await evaluation_db.CampaignRepository().get(user_id="user-a", campaign_id="cmp-1")
    assert campaign.status.value == "completed"


@pytest.mark.asyncio
async def test_worker_completion_after_campaign_cancellation_exits_cleanly(
    store: EvaluationJobStore,
) -> None:
    claim = await _claim_seeded_execution(store)
    worker = DatasetExecutionWorker(
        store=store,
        runner=AsyncMock(
            return_value=_successful_payload(question_id="Q1", mode="naive", answer="42")
        ),
    )
    await store.cancel_campaign_jobs(user_id="user-a", campaign_id="cmp-1")

    await worker.execute(claim)

    attempts = await store.list_attempts(user_id="user-a", work_item_id=claim.work_item_id)
    assert attempts[-1].status.value == "cancelled"
    assert await store.get_job_item_status(claim.job_item_id) == "cancelled"
    assert await evaluation_db.CampaignResultRepository().list_for_campaign(
        user_id="user-a", campaign_id="cmp-1"
    ) == []
