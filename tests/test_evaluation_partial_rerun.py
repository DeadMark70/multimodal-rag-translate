"""Selection regressions for repairing a small part of a campaign."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from evaluation.campaign_engine import CampaignEngine
from evaluation.campaign_schemas import CampaignLifecycleStatus, CampaignResultStatus
from evaluation.job_schemas import EvaluationRerunRequest
from evaluation.job_schemas import ClaimedEvaluationWork
from evaluation.execution_worker import DatasetExecutionWorker


@pytest.mark.asyncio
async def test_missing_scores_select_only_requested_mode_and_absent_metric() -> None:
    campaign = SimpleNamespace(
        status=CampaignLifecycleStatus.COMPLETED_WITH_ERRORS,
        config=SimpleNamespace(ragas_batch_size=4, ragas_parallel_batches=2),
    )
    rows = [SimpleNamespace(
        id=identity, question_id="Q30", mode=mode, agentic_execution_version="v10",
        status=CampaignResultStatus.COMPLETED, source_attempt_id="attempt",
    ) for identity, mode in (("naive-30", "naive"), ("agentic-30", "agentic"))]
    job = SimpleNamespace(job_id="new-job", created_at="2026-09-16")
    store = SimpleNamespace(
        backfill_legacy_attempts=AsyncMock(),
        ensure_ragas_work=AsyncMock(return_value=1),
        list_jobs=AsyncMock(side_effect=[[], [job]]),
    )
    engine = CampaignEngine(
        campaign_repository=SimpleNamespace(get=AsyncMock(return_value=campaign)),
        result_repository=SimpleNamespace(list_for_campaign=AsyncMock(return_value=rows)),
        ragas_evaluator=SimpleNamespace(
            enabled_metrics=["answer_correctness", "faithfulness", "answer_relevancy"],
            evaluator_model="test-judge",
        ),
        job_store=store, worker_notifier=lambda: None, configure_worker=False,
    )
    scores = [
        {"campaign_result_id": "naive-30", "metric_name": "answer_correctness", "metric_value": 0},
        {"campaign_result_id": "naive-30", "metric_name": "answer_relevancy", "metric_value": 0.8},
    ]
    with patch("evaluation.campaign_engine.RagasScoreRepository") as repository:
        repository.return_value.list_for_campaign = AsyncMock(return_value=scores)
        result = await engine.create_rerun(
            user_id="user", campaign_id="campaign",
            request=EvaluationRerunRequest(
                scope="missing_only", stages="ragas", question_ids=["Q30"], modes=["naive"],
            ),
        )
    assert result is job
    kwargs = store.ensure_ragas_work.await_args.kwargs
    assert kwargs["selected_result_ids"] == ["naive-30"]
    assert kwargs["metric_names_by_result"] == {"naive-30": ["faithfulness"]}


def test_execution_selection_preserves_other_modes_and_questions() -> None:
    rows = [{"work_item_id": identity, "status": "succeeded", "input_snapshot": {
        "test_case": {"id": question}, "mode": mode, "agentic_execution_version": "v10",
    }} for identity, question, mode in (
        ("n13", "Q13", "naive"), ("a13", "Q13", "agentic"), ("a14", "Q14", "agentic-v10"),
    )]
    selected = CampaignEngine._select_rerun_work_rows(
        rows, kind="execution", request=EvaluationRerunRequest(
            scope="selected", stages="execution_and_ragas", question_ids=["Q13"], modes=["agentic-v10"],
        ),
    )
    assert [row["work_item_id"] for row in selected] == ["a13"]


@pytest.mark.asyncio
async def test_execution_rerun_scores_only_the_selected_mode() -> None:
    store = SimpleNamespace(
        get_job=AsyncMock(return_value=SimpleNamespace(config_snapshot={
            "downstream_question_ids": ["Q13"], "downstream_modes": ["agentic-v10"],
        })),
        ensure_ragas_work=AsyncMock(return_value=1),
    )
    rows = [SimpleNamespace(id=identity, question_id="Q13", mode=mode, agentic_execution_version="v10")
            for identity, mode in (("n13", "naive"), ("a13", "agentic"))]
    worker = DatasetExecutionWorker(
        store=store,
        campaign_repository=SimpleNamespace(derive_execution_state=AsyncMock(return_value=SimpleNamespace(
            status=CampaignLifecycleStatus.COMPLETED, config=None,
        ))),
        result_repository=SimpleNamespace(list_for_campaign=AsyncMock(return_value=rows)),
        ragas_evaluator=SimpleNamespace(enabled_metrics=["faithfulness"], evaluator_model="judge"),
    )
    await worker._derive_campaign_state(ClaimedEvaluationWork(
        job_id="job", job_item_id="item", work_item_id="work", attempt_id="attempt",
        input_snapshot={"user_id": "user", "campaign_id": "campaign"},
    ))
    assert store.ensure_ragas_work.await_args.kwargs["selected_result_ids"] == ["a13"]
