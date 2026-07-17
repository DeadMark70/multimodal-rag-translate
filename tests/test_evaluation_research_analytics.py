"""Strict research-summary contract regression tests using durable repositories."""

from datetime import datetime, timezone

import pytest
import pytest_asyncio

from evaluation import db as evaluation_db
from evaluation.accounting_schemas import AccountingScopeStart, UsageEventCreate
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.campaign_schemas import CampaignConfig, CampaignResultStatus
from evaluation.db import (
    CampaignRepository,
    CampaignResultRepository,
    RagasScoreRepository,
)
from evaluation.research_analytics import ResearchAnalyticsService, nearest_rank
from evaluation.schemas import ModelConfig


def test_nearest_rank_percentiles_are_observed_values() -> None:
    assert nearest_rank([100, 200, 300, 400, 500], 0.50) == 300
    assert nearest_rank([100, 200, 300, 400, 500], 0.95) == 500


@pytest_asyncio.fixture
async def research_service(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    return ResearchAnalyticsService()


async def _campaign(campaign_id: str, modes: list[str]) -> None:
    repository = CampaignRepository()
    await repository.create(
        user_id="user-1",
        name=campaign_id,
        config=CampaignConfig(
            test_case_ids=["q-1"],
            modes=modes,
            model_config=ModelConfig(id="model-1", name="Model", model_name="model"),
        ),
    )
    # Repository-generated IDs are intentionally not assumed by the fixture.
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            "UPDATE campaigns SET id = ? WHERE name = ?", (campaign_id, campaign_id)
        )
        await connection.commit()


async def _result(
    campaign_id: str,
    mode: str,
    attempt: str,
    *,
    latency: float = 100.0,
    run_number: int = 1,
    status: CampaignResultStatus = CampaignResultStatus.COMPLETED,
) -> str:
    result = await CampaignResultRepository().create(
        user_id="user-1",
        campaign_id=campaign_id,
        question_id=f"q-{mode}",
        question="Q",
        ground_truth="A",
        ground_truth_short=None,
        key_points=[],
        ragas_focus=[],
        mode=mode,
        execution_profile="v2",
        context_policy_version="v2",
        run_number=run_number,
        answer="A",
        contexts=[],
        source_doc_ids=[],
        expected_sources=[],
        latency_ms=latency,
        token_usage={},
        category=None,
        difficulty=None,
        status=status,
        total_latency_ms=latency,
        source_attempt_id=attempt,
    )
    return result.id


async def _execution_scope(
    campaign_id: str,
    result_id: str,
    attempt: str,
    *,
    official: bool,
    scope_status: str = "completed",
    tokens: int = 15,
    cost: float | None = 0.1,
    pricing_status: str = "priced",
) -> None:
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id=f"scope-{attempt}",
            campaign_id=campaign_id,
            scope_type="execution_run",
            scope_key=attempt,
            run_id=result_id,
            targets=[
                {
                    "job_id": "job-1",
                    "work_item_id": "work-1",
                    "attempt_id": attempt,
                    "campaign_result_id": result_id,
                    "is_official": official,
                }
            ],
        )
    )
    await store.record_event(
        UsageEventCreate(
            usage_event_id=f"event-{attempt}",
            scope_id=f"scope-{attempt}",
            campaign_id=campaign_id,
            scope_type="execution_run",
            scope_key=attempt,
            run_id=result_id,
            phase="answer_generation",
            purpose="evaluation",
            input_tokens=tokens - 5,
            output_text_tokens=5,
            reported_total_tokens=tokens,
            usage_status="measured",
            reconciliation_status="balanced",
            estimated_cost_usd=cost,
            pricing_status=pricing_status,
            created_at=datetime.now(timezone.utc),
        )
    )
    await store.finalize_scope(f"scope-{attempt}", scope_status)


async def _official_scope(campaign_id: str, result_id: str, attempt: str) -> None:
    await _execution_scope(campaign_id, result_id, attempt, official=True)


@pytest.mark.asyncio
async def test_legacy_campaign_is_visible_but_not_comparable(research_service) -> None:
    await _campaign("legacy", ["naive"])
    await _result("legacy", "naive", "legacy-attempt")

    summary = await research_service.get_summary(user_id="user-1", campaign_id="legacy")

    assert summary.token_accounting_status == "incomplete_legacy"
    assert summary.tokens.total_tokens is None
    assert summary.execution_cost.benchmark_usd is None
    assert summary.modes[0].comparable is False
    assert "legacy_accounting" in summary.modes[0].not_comparable_reasons


@pytest.mark.asyncio
async def test_missing_faithfulness_stays_null_and_primary_metrics_are_present(
    research_service,
) -> None:
    await _campaign("partial", ["naive"])
    result_id = await _result("partial", "naive", "attempt-1")
    await _official_scope("partial", result_id, "attempt-1")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="partial",
        score_rows=[
            {
                "campaign_result_id": result_id,
                "metric_name": "answer_correctness",
                "metric_value": 0.8,
                "source_attempt_id": "attempt-1",
                "evaluation_signature": "ragas-v1",
                "details": {"evaluator_model": "judge", "metric_version": "v1"},
            },
            {
                "campaign_result_id": result_id,
                "metric_name": "answer_relevancy",
                "metric_value": 0.7,
                "source_attempt_id": "attempt-1",
                "evaluation_signature": "ragas-v1",
                "details": {"evaluator_model": "judge", "metric_version": "v1"},
            },
        ],
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="partial"
    )

    assert set(summary.modes[0].quality) == {
        "answer_correctness",
        "faithfulness",
        "answer_relevancy",
    }
    faithfulness = summary.modes[0].quality["faithfulness"]
    assert faithfulness.value is None
    assert faithfulness.status == "failed"
    assert summary.quality["faithfulness"].value is None
    assert summary.quality["faithfulness"].status == "failed"


@pytest.mark.asyncio
async def test_mixed_campaign_keeps_execution_and_ragas_accounting_separate(
    research_service,
) -> None:
    await _campaign("mixed", ["naive", "graph"])
    naive = await _result("mixed", "naive", "naive-official", latency=100)
    graph = await _result("mixed", "graph", "graph-official", latency=500)
    failed = await _result(
        "mixed",
        "naive",
        "naive-retry",
        run_number=2,
        status=CampaignResultStatus.FAILED,
    )
    await _execution_scope(
        "mixed", naive, "naive-official", official=True, tokens=15, cost=0.1
    )
    await _execution_scope(
        "mixed", graph, "graph-official", official=True, tokens=30, cost=0.2
    )
    await _execution_scope(
        "mixed",
        failed,
        "naive-retry",
        official=False,
        scope_status="failed",
        tokens=45,
        cost=None,
        pricing_status="missing_price",
    )
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="ragas-batch",
            campaign_id="mixed",
            scope_type="ragas_batch",
            scope_key="faithfulness",
            metric_name="faithfulness",
            targets=[
                {"job_id": "ragas", "work_item_id": "faith", "attempt_id": "ragas-1"}
            ],
        )
    )
    await store.record_event(
        UsageEventCreate(
            usage_event_id="ragas-event",
            scope_id="ragas-batch",
            campaign_id="mixed",
            scope_type="ragas_batch",
            scope_key="faithfulness",
            phase="evaluation",
            purpose="ragas",
            metric_name="faithfulness",
            model_name="judge-v1",
            input_tokens=100,
            output_text_tokens=20,
            reported_total_tokens=120,
            usage_status="measured",
            reconciliation_status="balanced",
            estimated_cost_usd=0.5,
            pricing_status="priced",
            created_at=datetime.now(timezone.utc),
        )
    )
    await store.finalize_scope("ragas-batch", "completed")

    summary = await research_service.get_summary(user_id="user-1", campaign_id="mixed")

    assert summary.tokens.total_tokens == 45
    assert summary.execution_cost.benchmark_usd == pytest.approx(0.3)
    assert summary.execution_cost.operational_usd is None
    assert summary.execution_cost.pricing_status == "partial"
    assert (
        summary.execution_cost.priced_call_count,
        summary.execution_cost.unpriced_call_count,
    ) == (2, 1)
    assert summary.evaluation_overhead.tokens.total_tokens == 120
    assert summary.evaluation_overhead.cost_usd == pytest.approx(0.5)
    graph_mode = next(mode for mode in summary.modes if mode.mode == "graph")
    naive_mode = next(mode for mode in summary.modes if mode.mode == "naive")
    assert graph_mode.execution_cost.operational_usd == pytest.approx(0.2)
    assert graph_mode.execution_cost.pricing_status == "complete"
    assert naive_mode.execution_cost.operational_usd is None


@pytest.mark.asyncio
async def test_failed_context_batch_requests_null_optional_metric(
    research_service,
) -> None:
    await _campaign("requested-context", ["naive"])
    result_id = await _result("requested-context", "naive", "attempt")
    await _official_scope("requested-context", result_id, "attempt")
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="failed-context-batch",
            campaign_id="requested-context",
            scope_type="ragas_batch",
            scope_key="context-recall",
            metric_name="context_recall",
            targets=[
                {"job_id": "ragas", "work_item_id": "context", "attempt_id": "batch"}
            ],
        )
    )
    await store.finalize_scope("failed-context-batch", "failed")

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="requested-context"
    )

    context_recall = summary.quality["context_recall"]
    assert context_recall.value is None
    assert context_recall.status == "failed"


@pytest.mark.asyncio
async def test_cross_linked_score_attempt_is_excluded_from_its_result(
    research_service,
) -> None:
    await _campaign("stale-score", ["naive"])
    first = await _result("stale-score", "naive", "first")
    second = await _result("stale-score", "naive", "second", run_number=2)
    await _official_scope("stale-score", first, "first")
    await _official_scope("stale-score", second, "second")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="stale-score",
        score_rows=[
            {
                "campaign_result_id": first,
                "metric_name": "answer_correctness",
                "metric_value": 0.9,
                "source_attempt_id": "second",
                "evaluation_signature": "sig",
                "details": {"evaluator_model": "judge"},
            },
            {
                "campaign_result_id": second,
                "metric_name": "answer_correctness",
                "metric_value": 0.4,
                "source_attempt_id": "second",
                "evaluation_signature": "sig",
                "details": {"evaluator_model": "judge"},
            },
        ],
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="stale-score"
    )
    observation = summary.quality["answer_correctness"]

    assert observation.value == pytest.approx(0.4)
    assert observation.valid_samples == 1
    assert observation.metric_version is None


@pytest.mark.asyncio
async def test_mixed_evaluator_metadata_excludes_incompatible_scores_and_marks_mode(
    research_service,
) -> None:
    await _campaign("compat", ["naive"])
    first = await _result("compat", "naive", "first")
    second = await _result("compat", "naive", "second", run_number=2)
    await _official_scope("compat", first, "first")
    await _official_scope("compat", second, "second")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="compat",
        score_rows=[
            {
                "campaign_result_id": first,
                "metric_name": "answer_correctness",
                "metric_value": 0.8,
                "source_attempt_id": "first",
                "evaluation_signature": "sig-a",
                "details": {"evaluator_model": "judge-a", "metric_version": "v1"},
            },
            {
                "campaign_result_id": second,
                "metric_name": "answer_correctness",
                "metric_value": 0.2,
                "source_attempt_id": "second",
                "evaluation_signature": "sig-a",
                "details": {"evaluator_model": "judge-b", "metric_version": "v2"},
            },
            {
                "campaign_result_id": first,
                "metric_name": "answer_correctness",
                "metric_value": 0.3,
                "source_attempt_id": "first",
                "evaluation_signature": "sig-b",
                "details": {"evaluator_model": "judge-a", "metric_version": "v1"},
            },
        ],
    )

    summary = await research_service.get_summary(user_id="user-1", campaign_id="compat")
    mode = summary.modes[0]

    assert mode.quality["answer_correctness"].value == pytest.approx(0.8)
    assert mode.quality["answer_correctness"].valid_samples == 1
    assert mode.comparable is False
    assert "evaluator_metadata_mismatch" in mode.not_comparable_reasons
    assert any(
        warning.code == "evaluator_metadata_mismatch" for warning in summary.warnings
    )


@pytest.mark.asyncio
async def test_campaign_aggregate_uses_raw_runs_and_optional_metrics_are_requested_only(
    research_service,
) -> None:
    await _campaign("raw", ["naive", "graph"])
    naive_one = await _result("raw", "naive", "n1", latency=10)
    naive_two = await _result("raw", "naive", "n2", latency=20, run_number=2)
    graph = await _result("raw", "graph", "g1", latency=100, run_number=3)
    for result_id, attempt in ((naive_one, "n1"), (naive_two, "n2"), (graph, "g1")):
        await _official_scope("raw", result_id, attempt)
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="raw",
        score_rows=[
            {
                "campaign_result_id": naive_one,
                "metric_name": "answer_correctness",
                "metric_value": 0.1,
                "source_attempt_id": "n1",
                "evaluation_signature": "sig",
                "details": {"evaluator_model": "judge", "metric_version": "v1"},
            },
            {
                "campaign_result_id": naive_two,
                "metric_name": "answer_correctness",
                "metric_value": 0.1,
                "source_attempt_id": "n2",
                "evaluation_signature": "sig",
                "details": {"evaluator_model": "judge", "metric_version": "v1"},
            },
            {
                "campaign_result_id": graph,
                "metric_name": "answer_correctness",
                "metric_value": 0.9,
                "source_attempt_id": "g1",
                "evaluation_signature": "sig",
                "details": {"evaluator_model": "judge", "metric_version": "v1"},
            },
            {
                "campaign_result_id": graph,
                "metric_name": "context_precision",
                "metric_value": 0.6,
                "source_attempt_id": "g1",
                "evaluation_signature": "sig",
                "details": {"evaluator_model": "judge", "metric_version": "v1"},
            },
        ],
    )

    summary = await research_service.get_summary(user_id="user-1", campaign_id="raw")

    assert summary.quality["answer_correctness"].value == pytest.approx(
        (0.1 + 0.1 + 0.9) / 3
    )
    assert summary.latency.mean_ms == pytest.approx(130 / 3)
    assert summary.latency.p50_ms == 20
    assert summary.latency.p95_ms == 100
    assert summary.latency.low_sample_size is True
    assert set(summary.quality) == {
        "answer_correctness",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    }
