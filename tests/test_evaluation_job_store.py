from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from core.errors import AppError, ErrorCode
import evaluation.db as evaluation_db
from evaluation.campaign_schemas import (
    CampaignLifecycleStatus,
    CampaignResult,
    CampaignResultStatus,
)
from evaluation.accounting_runtime import start_execution_scope
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.db import (
    CampaignRepository,
    CampaignResultRepository,
    RagasScoreRepository,
)
from evaluation.error_policy import ErrorDecision
from evaluation.job_schemas import (
    ExecutionAttemptOutput,
    RagasAttemptOutput,
    WorkItemSpec,
)
from evaluation.job_store import build_evaluation_signature, build_ragas_batch_group_key
from evaluation.ragas_worker import RagasBatchWorker


@pytest_asyncio.fixture
async def store(monkeypatch):  # noqa: ANN001
    database_path = (
        Path(__file__).resolve().parent.parent
        / ".test-artifacts"
        / f"evaluation-ledger-{uuid4().hex}.db"
    )
    database_path.parent.mkdir(exist_ok=True)
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


def _spec(
    *, logical_key: str = "execution:Q1:naive:1:none", max_attempts: int = 3
) -> WorkItemSpec:
    return WorkItemSpec(
        work_type="dataset_execution",
        logical_key=logical_key,
        input_snapshot={"question_id": "Q1"},
        max_attempts=max_attempts,
    )


def _successful_output(answer: str) -> ExecutionAttemptOutput:
    return ExecutionAttemptOutput(
        result=CampaignResult(
            id="worker-result-id",
            campaign_id="cmp-1",
            question_id="Q1",
            question="Question 1",
            ground_truth="Ground truth 1",
            mode="naive",
            run_number=1,
            answer=answer,
            contexts=["ctx-1"],
            source_doc_ids=[],
            expected_sources=[],
            latency_ms=10,
            token_usage={"total_tokens": 10},
            status=CampaignResultStatus.COMPLETED,
            created_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
        )
    )


async def _claim_execution(store, now: datetime):  # noqa: ANN001
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec(max_attempts=1)],
    )
    return (await store.claim_ready_items(limit=1, now=now))[0]


async def _claim_execution_rerun(store, now: datetime):  # noqa: ANN001
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={"scope": "all"},
        config_snapshot={},
        items=[_spec(max_attempts=1)],
    )
    return (await store.claim_ready_items(limit=1, now=now))[0]


@pytest.mark.asyncio
async def test_complete_execution_attempt_atomically_promotes_accounting_scope(
    store, fixed_now
) -> None:  # noqa: ANN001
    claim = await _claim_execution(store, fixed_now)
    accounting_store = EvaluationAccountingStore()
    scope = await start_execution_scope(
        store=accounting_store,
        campaign_id="cmp-1",
        run_id="run-1",
        job_id=claim.job_id,
        work_item_id=claim.work_item_id,
        attempt_id=claim.attempt_id,
    )

    result = await store.complete_execution_attempt(
        claim,
        _successful_output("answer"),
        accounting_scope_id=scope.scope_id,
    )

    stored_scope = await accounting_store.get_scope(scope.scope_id)
    assert stored_scope.status == "completed"
    assert stored_scope.targets[0].is_official is True
    assert stored_scope.targets[0].campaign_result_id == result.id


@pytest.mark.asyncio
async def test_invalid_accounting_scope_rolls_back_execution_promotion(
    store, fixed_now
) -> None:  # noqa: ANN001
    claim = await _claim_execution(store, fixed_now)

    with pytest.raises(ValueError, match="accounting scope"):
        await store.complete_execution_attempt(
            claim,
            _successful_output("answer"),
            accounting_scope_id="missing-scope",
        )

    assert (
        await CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
        == []
    )
    attempts = await store.list_attempts(
        user_id="user-a", work_item_id=claim.work_item_id
    )
    assert attempts[-1].status.value == "running"


@pytest.mark.asyncio
async def test_create_job_notifies_post_commit_producer_hook(store) -> None:  # noqa: ANN001
    notifications: list[None] = []

    def notify() -> None:
        notifications.append(None)

    from evaluation.job_store import EvaluationJobStore

    producer = EvaluationJobStore(on_job_created=notify)
    await producer.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec()],
    )

    assert notifications == [None]


def test_ragas_batch_group_key_is_shared_while_result_signatures_remain_distinct() -> (
    None
):
    first = CampaignResult(
        id="result-1",
        campaign_id="cmp-1",
        question_id="Q1",
        question="Question",
        ground_truth="Ground truth",
        mode="naive",
        run_number=1,
        answer="Answer",
        contexts=["Context"],
        context_policy_version="policy-v1",
        status=CampaignResultStatus.COMPLETED,
        created_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    second = first.model_copy(update={"id": "result-2"})
    kwargs = {
        "evaluator_model": "evaluator",
        "evaluator_config": {"temperature": 0},
        "metric_name": "faithfulness",
        "metric_version": "v1",
        "ground_truth_hash": "ground-truth-hash",
        "context_metrics_enabled": False,
    }

    assert build_evaluation_signature(
        result=first, **kwargs
    ) != build_evaluation_signature(result=second, **kwargs)
    assert build_ragas_batch_group_key(
        result=first, **kwargs
    ) == build_ragas_batch_group_key(result=second, **kwargs)


@pytest.mark.asyncio
async def test_ensure_ragas_work_assigns_one_shared_batch_key_to_compatible_results(
    store, fixed_now
) -> None:  # noqa: ANN001
    repository = CampaignResultRepository()
    for index in range(4):
        result = await repository.create(
            result_id=f"ragas-result-{index}",
            user_id="user-a",
            campaign_id="cmp-1",
            question_id=f"Q{index}",
            question="Question",
            ground_truth="Ground truth",
            ground_truth_short="Ground truth",
            key_points=[],
            ragas_focus=[],
            mode="naive",
            execution_profile=None,
            context_policy_version="policy-v1",
            run_number=index + 1,
            answer="Answer",
            contexts=["Context"],
            source_doc_ids=[],
            expected_sources=[],
            latency_ms=1,
            token_usage={"total_tokens": 1},
            category=None,
            difficulty=None,
            status=CampaignResultStatus.COMPLETED,
        )
        async with evaluation_db.connect_db() as connection:
            await connection.execute(
                "UPDATE campaign_results SET source_attempt_id = ? WHERE id = ?",
                (f"attempt-{index}", result.id),
            )
            await connection.commit()

    assert (
        await store.ensure_ragas_work(
            user_id="user-a",
            campaign_id="cmp-1",
            evaluator_model="evaluator",
            evaluator_config={"temperature": 0},
            enabled_metrics=["faithfulness"],
            ragas_batch_size=4,
            ragas_parallel_batches=1,
        )
        == 4
    )

    claims = await store.claim_ready_items(
        limit=4, now=fixed_now, work_type="ragas_metric"
    )
    assert len(claims) == 4
    assert len({claim.input_snapshot["evaluation_signature"] for claim in claims}) == 4
    assert len({claim.input_snapshot["batch_group_key"] for claim in claims}) == 1

    class BatchEvaluator:
        def __init__(self) -> None:
            self.calls: list[int] = []

        async def evaluate_metric_batch(self, metric_name, rows, llm, embeddings):  # noqa: ANN001
            self.calls.append(len(rows))
            return [0.5] * len(rows)

    batch_evaluator = BatchEvaluator()
    await RagasBatchWorker(store=store, evaluator=batch_evaluator).execute(claims)
    assert batch_evaluator.calls == [4]


@pytest.mark.asyncio
async def test_derive_ragas_state_preserves_terminal_campaign_cancellation(
    store, fixed_now
) -> None:  # noqa: ANN001
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            "UPDATE campaigns SET config_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "test_case_ids": ["Q1"],
                        "modes": ["naive"],
                        "model_config": {
                            "id": "cfg-1",
                            "name": "test",
                            "model_name": "test-model",
                        },
                    }
                ),
                "cmp-1",
            ),
        )
        await connection.commit()
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[
            WorkItemSpec(
                work_type="ragas_metric",
                logical_key="ragas:result-1:faithfulness:signature-v1",
                input_snapshot={
                    "campaign_result_id": "result-1",
                    "metric_name": "faithfulness",
                },
            )
        ],
    )
    claims = await store.claim_ready_items(
        limit=1, now=fixed_now, work_type="ragas_metric"
    )
    assert len(claims) == 1

    repository = CampaignRepository()
    await repository.mark_evaluating(
        user_id="user-a", campaign_id="cmp-1", evaluation_total_units=1
    )
    await store.cancel_campaign_jobs(user_id="user-a", campaign_id="cmp-1")
    await repository.mark_cancelled(user_id="user-a", campaign_id="cmp-1")

    derived = await repository.derive_ragas_state(
        user_id="user-a", campaign_id="cmp-1", job_id=claims[0].job_id
    )

    assert derived.status is CampaignLifecycleStatus.CANCELLED


@pytest.mark.asyncio
async def test_failed_rerun_preserves_previous_success_projection(
    store, fixed_now
) -> None:  # noqa: ANN001
    first_claim = await _claim_execution(store, fixed_now)
    first = await store.complete_execution_attempt(
        first_claim, _successful_output("answer-v1")
    )
    rerun_claim = await _claim_execution_rerun(store, fixed_now)
    await store.fail_attempt(
        rerun_claim,
        ErrorDecision("transport", False, None, "The provider is unavailable."),
        next_retry_at=None,
    )

    current = await CampaignResultRepository().get(
        user_id="user-a", campaign_id="cmp-1", result_id=first.id
    )
    assert current.answer == "answer-v1"
    assert current.source_attempt_id == first_claim.attempt_id


@pytest.mark.asyncio
async def test_successful_rerun_promotion_keeps_old_attempt(store, fixed_now) -> None:  # noqa: ANN001
    first_claim = await _claim_execution(store, fixed_now)
    first = await store.complete_execution_attempt(
        first_claim, _successful_output("answer-v1")
    )
    second_claim = await _claim_execution_rerun(store, fixed_now)
    second = await store.complete_execution_attempt(
        second_claim, _successful_output("answer-v2")
    )

    assert second.id == first.id
    assert second.answer == "answer-v2"
    assert second.source_attempt_id == second_claim.attempt_id
    assert (
        len(
            await store.list_attempts(
                user_id="user-a", work_item_id=second_claim.work_item_id
            )
        )
        == 2
    )


@pytest.mark.asyncio
async def test_non_completed_execution_output_preserves_official_projection(
    store, fixed_now
) -> None:  # noqa: ANN001
    first_claim = await _claim_execution(store, fixed_now)
    first = await store.complete_execution_attempt(
        first_claim, _successful_output("answer-v1")
    )
    rerun_claim = await _claim_execution_rerun(store, fixed_now)
    failed_output = _successful_output("bad-answer").model_copy(
        update={
            "result": _successful_output("bad-answer").result.model_copy(
                update={"status": CampaignResultStatus.FAILED}
            )
        }
    )

    with pytest.raises(ValueError, match="completed"):
        await store.complete_execution_attempt(rerun_claim, failed_output)

    current = await CampaignResultRepository().get(
        user_id="user-a", campaign_id="cmp-1", result_id=first.id
    )
    attempts = await store.list_attempts(
        user_id="user-a", work_item_id=rerun_claim.work_item_id
    )
    assert current.answer == "answer-v1"
    assert [attempt.status for attempt in attempts] == ["succeeded", "running"]
    assert await store.get_job_item_status(rerun_claim.job_item_id) == "running"


@pytest.mark.asyncio
async def test_incompatible_signature_preserves_current_score(store, fixed_now) -> None:  # noqa: ANN001
    execution_claim = await _claim_execution(store, fixed_now)
    result = await store.complete_execution_attempt(
        execution_claim, _successful_output("answer-v1")
    )
    ragas_spec = WorkItemSpec(
        work_type="ragas_metric",
        logical_key="ragas:Q1:naive:1:faithfulness",
        input_snapshot={"campaign_result_id": result.id},
        max_attempts=1,
    )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[ragas_spec],
    )
    first_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    await store.complete_ragas_attempt(
        first_claim,
        RagasAttemptOutput(
            scores=[
                {
                    "campaign_result_id": result.id,
                    "metric_name": "faithfulness",
                    "metric_value": 0.8,
                    "details": {},
                    "evaluation_signature": "compatible",
                }
            ]
        ),
    )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={},
        config_snapshot={},
        items=[ragas_spec],
    )
    rerun_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    await store.complete_ragas_attempt(
        rerun_claim,
        RagasAttemptOutput(
            scores=[
                {
                    "campaign_result_id": result.id,
                    "metric_name": "faithfulness",
                    "metric_value": 0.1,
                    "details": {},
                    "evaluation_signature": "incompatible",
                }
            ]
        ),
    )

    async with evaluation_db.connect_db() as connection:
        row = await (
            await connection.execute(
                "SELECT metric_value, source_attempt_id FROM ragas_scores "
                "WHERE campaign_result_id = ? AND metric_name = ?",
                (result.id, "faithfulness"),
            )
        ).fetchone()
    assert row["metric_value"] == 0.8
    assert row["source_attempt_id"] == first_claim.attempt_id


@pytest.mark.asyncio
async def test_legacy_null_signature_preserves_current_score(store, fixed_now) -> None:  # noqa: ANN001
    execution_claim = await _claim_execution(store, fixed_now)
    result = await store.complete_execution_attempt(
        execution_claim, _successful_output("answer-v1")
    )
    ragas_spec = WorkItemSpec(
        work_type="ragas_metric",
        logical_key="ragas:Q1:naive:1:faithfulness",
        input_snapshot={"campaign_result_id": result.id},
        max_attempts=1,
    )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[ragas_spec],
    )
    first_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    await store.complete_ragas_attempt(
        first_claim,
        RagasAttemptOutput(
            scores=[
                {
                    "campaign_result_id": result.id,
                    "metric_name": "faithfulness",
                    "metric_value": 0.8,
                    "details": {},
                    "evaluation_signature": "compatible",
                }
            ]
        ),
    )
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            "UPDATE ragas_scores SET evaluation_signature = NULL "
            "WHERE campaign_result_id = ? AND metric_name = ?",
            (result.id, "faithfulness"),
        )
        await connection.commit()
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={},
        config_snapshot={},
        items=[ragas_spec],
    )
    rerun_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    await store.complete_ragas_attempt(
        rerun_claim,
        RagasAttemptOutput(
            scores=[
                {
                    "campaign_result_id": result.id,
                    "metric_name": "faithfulness",
                    "metric_value": 0.1,
                    "details": {},
                    "evaluation_signature": "compatible",
                }
            ]
        ),
    )

    async with evaluation_db.connect_db() as connection:
        row = await (
            await connection.execute(
                "SELECT metric_value, source_attempt_id FROM ragas_scores "
                "WHERE campaign_result_id = ? AND metric_name = ?",
                (result.id, "faithfulness"),
            )
        ).fetchone()
    assert row["metric_value"] == 0.8
    assert row["source_attempt_id"] == first_claim.attempt_id


@pytest.mark.asyncio
async def test_failed_ragas_rerun_preserves_current_score(store, fixed_now) -> None:  # noqa: ANN001
    execution_claim = await _claim_execution(store, fixed_now)
    result = await store.complete_execution_attempt(
        execution_claim, _successful_output("answer-v1")
    )
    ragas_spec = WorkItemSpec(
        work_type="ragas_metric",
        logical_key="ragas:Q1:naive:1:faithfulness",
        input_snapshot={"campaign_result_id": result.id},
        max_attempts=1,
    )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[ragas_spec],
    )
    first_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    await store.complete_ragas_attempt(
        first_claim,
        RagasAttemptOutput(
            scores=[
                {
                    "campaign_result_id": result.id,
                    "metric_name": "faithfulness",
                    "metric_value": 0.8,
                    "details": {},
                    "evaluation_signature": "compatible",
                }
            ]
        ),
    )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={},
        config_snapshot={},
        items=[ragas_spec],
    )
    rerun_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    await store.fail_attempt(
        rerun_claim,
        ErrorDecision("transport", False, None, "The provider is unavailable."),
        next_retry_at=None,
    )

    async with evaluation_db.connect_db() as connection:
        row = await (
            await connection.execute(
                "SELECT metric_value, source_attempt_id FROM ragas_scores "
                "WHERE campaign_result_id = ? AND metric_name = ?",
                (result.id, "faithfulness"),
            )
        ).fetchone()
    assert row["metric_value"] == 0.8
    assert row["source_attempt_id"] == first_claim.attempt_id


@pytest.mark.asyncio
async def test_compatible_ragas_rerun_replaces_current_score(store, fixed_now) -> None:  # noqa: ANN001
    execution_claim = await _claim_execution(store, fixed_now)
    result = await store.complete_execution_attempt(
        execution_claim, _successful_output("answer-v1")
    )
    ragas_spec = WorkItemSpec(
        work_type="ragas_metric",
        logical_key="ragas:Q1:naive:1:faithfulness",
        input_snapshot={"campaign_result_id": result.id},
        max_attempts=1,
    )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[ragas_spec],
    )
    first_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    await store.complete_ragas_attempt(
        first_claim,
        RagasAttemptOutput(
            scores=[
                {
                    "campaign_result_id": result.id,
                    "metric_name": "faithfulness",
                    "metric_value": 0.8,
                    "details": {},
                    "evaluation_signature": "compatible",
                }
            ]
        ),
    )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={},
        config_snapshot={},
        items=[ragas_spec],
    )
    rerun_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    await store.complete_ragas_attempt(
        rerun_claim,
        RagasAttemptOutput(
            scores=[
                {
                    "campaign_result_id": result.id,
                    "metric_name": "faithfulness",
                    "metric_value": 0.1,
                    "details": {},
                    "evaluation_signature": "compatible",
                }
            ]
        ),
    )

    async with evaluation_db.connect_db() as connection:
        row = await (
            await connection.execute(
                "SELECT metric_value, source_attempt_id FROM ragas_scores "
                "WHERE campaign_result_id = ? AND metric_name = ?",
                (result.id, "faithfulness"),
            )
        ).fetchone()
    assert row["metric_value"] == 0.1
    assert row["source_attempt_id"] == rerun_claim.attempt_id


@pytest.mark.asyncio
async def test_backfill_legacy_attempts_is_idempotent(store) -> None:  # noqa: ANN001
    legacy = _successful_output("legacy-answer").result
    await CampaignResultRepository().create(
        result_id=legacy.id,
        user_id="user-a",
        campaign_id=legacy.campaign_id,
        question_id=legacy.question_id,
        question=legacy.question,
        ground_truth=legacy.ground_truth,
        ground_truth_short=legacy.ground_truth_short,
        key_points=legacy.key_points,
        ragas_focus=legacy.ragas_focus,
        mode=legacy.mode,
        execution_profile=legacy.execution_profile,
        context_policy_version=legacy.context_policy_version,
        run_number=legacy.run_number,
        answer=legacy.answer,
        contexts=legacy.contexts,
        source_doc_ids=legacy.source_doc_ids,
        expected_sources=legacy.expected_sources,
        latency_ms=legacy.latency_ms,
        token_usage=legacy.token_usage,
        category=legacy.category,
        difficulty=legacy.difficulty,
        status=legacy.status,
    )
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-a",
        campaign_id="cmp-1",
        score_rows=[
            {
                "campaign_result_id": legacy.id,
                "metric_name": "faithfulness",
                "metric_value": 0.6,
                "details": {},
            }
        ],
    )

    await store.backfill_legacy_attempts()
    await store.backfill_legacy_attempts()

    async with evaluation_db.connect_db() as connection:
        result = await (
            await connection.execute(
                "SELECT source_attempt_id FROM campaign_results WHERE id = ?",
                (legacy.id,),
            )
        ).fetchone()
        score = await (
            await connection.execute(
                "SELECT source_attempt_id FROM ragas_scores WHERE campaign_result_id = ?",
                (legacy.id,),
            )
        ).fetchone()
        attempt_count = await (
            await connection.execute(
                "SELECT COUNT(*) AS count FROM evaluation_attempts"
            )
        ).fetchone()
    assert result["source_attempt_id"] is not None
    assert score["source_attempt_id"] is not None
    assert attempt_count["count"] == 2


@pytest.mark.asyncio
async def test_create_job_reuses_stable_work_item_and_creates_new_job_item(
    store,
) -> None:  # noqa: ANN001
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
async def test_retryable_failure_requeues_only_below_its_job_item_budget(
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
async def test_claim_skips_blocked_same_work_items_before_applying_limit(
    store, fixed_now
) -> None:  # noqa: ANN001
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec(logical_key="execution:Q1:naive:1:active")],
    )
    active_claim = (await store.claim_ready_items(limit=1, now=fixed_now))[0]
    for _ in range(2):
        await store.create_job_with_items(
            user_id="user-a",
            campaign_id="cmp-1",
            job_type="rerun",
            selection={},
            config_snapshot={},
            items=[_spec(logical_key="execution:Q1:naive:1:active")],
        )
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[_spec(logical_key="execution:Q2:naive:1:independent")],
    )

    claimed = await store.claim_ready_items(limit=2, now=fixed_now)

    assert len(claimed) == 1
    assert claimed[0].work_item_id != active_claim.work_item_id


@pytest.mark.asyncio
async def test_cancel_and_startup_recovery_preserve_attempt_history(
    store, fixed_now
) -> None:  # noqa: ANN001
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

    assert (
        await store.recover_interrupted_attempts(at=fixed_now + timedelta(minutes=5))
        == 1
    )
    attempts = await store.list_attempts(
        user_id="user-a", work_item_id=running_claim.work_item_id
    )
    assert [attempt.status for attempt in attempts] == ["interrupted"]
    assert await store.get_job_item_status(running_claim.job_item_id) == "pending"


@pytest.mark.asyncio
async def test_additive_migration_adds_result_provenance_columns(store) -> None:  # noqa: ANN001
    async with evaluation_db.connect_db() as connection:
        campaign_result_columns = await evaluation_db._table_columns(
            connection, "campaign_results"
        )
        ragas_score_columns = await evaluation_db._table_columns(
            connection, "ragas_scores"
        )

    assert "source_attempt_id" in campaign_result_columns
    assert "condition_id" in campaign_result_columns
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


@pytest.mark.asyncio
async def test_list_job_items_is_job_scoped_and_includes_latest_safe_attempt(
    store, fixed_now
) -> None:  # noqa: ANN001
    created = await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="rerun",
        selection={"scope": "all"},
        config_snapshot={},
        items=[_spec(max_attempts=2)],
    )
    claimed = (await store.claim_ready_items(limit=1, now=fixed_now))[0]

    items = await store.list_job_items(user_id="user-a", job_id=created.job_id)

    assert len(items) == 1
    assert items[0].job_item_id == claimed.job_item_id
    assert items[0].work_item_id == claimed.work_item_id
    assert items[0].work_type.value == "dataset_execution"
    assert items[0].question_id == "Q1"
    assert items[0].latest_attempt is not None
    assert items[0].latest_attempt.attempt_id == claimed.attempt_id
    assert items[0].latest_attempt.safe_error_message is None
    assert (
        await store.get_job(user_id="user-a", job_id=created.job_id)
    ).missing_items == 0
    with pytest.raises(AppError) as exc_info:
        await store.list_job_items(user_id="user-b", job_id=created.job_id)
    assert exc_info.value.code is ErrorCode.NOT_FOUND
