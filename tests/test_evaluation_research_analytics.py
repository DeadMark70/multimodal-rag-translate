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
    scope_run_id: str | None = None,
    usage_status: str = "measured",
    reconciliation_status: str = "balanced",
) -> None:
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id=f"scope-{attempt}",
            campaign_id=campaign_id,
            scope_type="execution_run",
            scope_key=attempt,
            run_id=scope_run_id or result_id,
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
            run_id=scope_run_id or result_id,
            phase="answer_generation",
            purpose="evaluation",
            input_tokens=tokens - 5 if usage_status == "measured" else 0,
            output_text_tokens=5 if usage_status == "measured" else 0,
            reported_total_tokens=tokens if usage_status == "measured" else None,
            usage_status=usage_status,
            reconciliation_status=reconciliation_status,
            estimated_cost_usd=cost,
            pricing_status=pricing_status,
            created_at=datetime.now(timezone.utc),
        )
    )
    await store.finalize_scope(f"scope-{attempt}", scope_status)


async def _official_scope(campaign_id: str, result_id: str, attempt: str) -> None:
    await _execution_scope(campaign_id, result_id, attempt, official=True)


def _primary_score_rows(
    result_attempts: list[tuple[str, str]],
    *,
    evaluator_model: str = "judge-v1",
    compatibility_signature: str = "policy-a",
) -> list[dict]:
    return [
        {
            "campaign_result_id": result_id,
            "metric_name": metric_name,
            "metric_value": 0.8,
            "source_attempt_id": attempt,
            "evaluation_signature": f"{attempt}-{metric_name}",
            "details": {
                "evaluator_model": evaluator_model,
                "metric_version": "v1",
                "compatibility_signature": compatibility_signature,
            },
        }
        for result_id, attempt in result_attempts
        for metric_name in (
            "answer_correctness",
            "faithfulness",
            "answer_relevancy",
        )
    ]


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
async def test_missing_usage_keeps_all_token_categories_nullable(
    research_service,
) -> None:
    await _campaign("missing-usage", ["naive"])
    result_id = await _result("missing-usage", "naive", "missing-attempt")
    await _execution_scope(
        "missing-usage",
        result_id,
        "missing-attempt",
        official=True,
        tokens=0,
        cost=None,
        pricing_status="unavailable_usage",
        usage_status="missing",
        reconciliation_status="unavailable",
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="missing-usage"
    )

    assert summary.tokens.input_tokens is None
    assert summary.tokens.output_text_tokens is None
    assert summary.tokens.reasoning_tokens is None
    assert summary.tokens.other_tokens is None
    assert summary.tokens.total_tokens is None
    assert summary.tokens.by_phase == {}
    assert summary.tokens.accounting_status == "partial"
    assert summary.tokens.phase_attribution_status == "not_available"


@pytest.mark.asyncio
async def test_mixed_usage_reports_measured_subtotals_without_total(
    research_service,
) -> None:
    await _campaign("mixed-usage", ["naive"])
    result_id = await _result("mixed-usage", "naive", "mixed-attempt")
    await _official_scope("mixed-usage", result_id, "mixed-attempt")
    await EvaluationAccountingStore().record_event(
        UsageEventCreate(
            usage_event_id="event-mixed-missing",
            scope_id="scope-mixed-attempt",
            campaign_id="mixed-usage",
            scope_type="execution_run",
            scope_key="mixed-attempt",
            run_id=result_id,
            phase="answer_generation",
            purpose="evaluation",
            usage_status="missing",
            reconciliation_status="unavailable",
            pricing_status="unavailable_usage",
            created_at=datetime.now(timezone.utc),
        )
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="mixed-usage"
    )

    assert summary.tokens.input_tokens == 10
    assert summary.tokens.output_text_tokens == 5
    assert summary.tokens.total_tokens is None
    assert summary.tokens.by_phase == {"answer_generation": 15}
    assert summary.tokens.accounting_status == "partial"


@pytest.mark.asyncio
async def test_campaign_cohort_rejects_modes_scored_by_different_evaluators(
    research_service,
) -> None:
    await _campaign("mixed-cohorts", ["naive", "graph"])
    naive = await _result("mixed-cohorts", "naive", "naive-attempt")
    graph = await _result("mixed-cohorts", "graph", "graph-attempt")
    await _official_scope("mixed-cohorts", naive, "naive-attempt")
    await _official_scope("mixed-cohorts", graph, "graph-attempt")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="mixed-cohorts",
        score_rows=(
            _primary_score_rows([(naive, "naive-attempt")])
            + _primary_score_rows(
                [(graph, "graph-attempt")],
                evaluator_model="judge-v2",
                compatibility_signature="policy-b",
            )
        ),
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="mixed-cohorts"
    )

    assert sum(mode.comparable for mode in summary.modes) <= 1
    assert any(
        "evaluator_metadata_mismatch" in mode.not_comparable_reasons
        for mode in summary.modes
    )


@pytest.mark.asyncio
async def test_campaign_cohort_keeps_modes_comparable_when_identity_is_shared(
    research_service,
) -> None:
    await _campaign("shared-cohort", ["naive", "graph"])
    naive = await _result("shared-cohort", "naive", "naive-attempt")
    graph = await _result("shared-cohort", "graph", "graph-attempt")
    await _official_scope("shared-cohort", naive, "naive-attempt")
    await _official_scope("shared-cohort", graph, "graph-attempt")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="shared-cohort",
        score_rows=_primary_score_rows(
            [(naive, "naive-attempt"), (graph, "graph-attempt")]
        ),
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="shared-cohort"
    )

    assert all(mode.comparable for mode in summary.modes)


@pytest.mark.asyncio
async def test_terminal_ragas_failure_is_counted_per_result_not_as_missing(
    research_service,
) -> None:
    await _campaign("per-result-failure", ["naive"])
    results = [
        await _result(
            "per-result-failure", "naive", f"attempt-{index}", run_number=index
        )
        for index in range(1, 6)
    ]
    for index, result_id in enumerate(results, start=1):
        await _official_scope("per-result-failure", result_id, f"attempt-{index}")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="per-result-failure",
        score_rows=[
            {
                "campaign_result_id": result_id,
                "metric_name": "faithfulness",
                "metric_value": 0.8,
                "source_attempt_id": f"attempt-{index}",
                "evaluation_signature": f"attempt-{index}-faithfulness",
                "details": {
                    "evaluator_model": "judge-v1",
                    "metric_version": "v1",
                    "compatibility_signature": "policy-a",
                },
            }
            for index, result_id in enumerate(results[:4], start=1)
        ],
    )
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="failed-fifth-faithfulness",
            campaign_id="per-result-failure",
            scope_type="ragas_batch",
            scope_key="faithfulness-fifth",
            metric_name="faithfulness",
            targets=[
                {
                    "campaign_result_id": results[4],
                    "job_id": "ragas",
                    "work_item_id": "faithfulness-fifth",
                    "attempt_id": "attempt-5",
                }
            ],
        )
    )
    await store.finalize_scope("failed-fifth-faithfulness", "failed")

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="per-result-failure"
    )
    observation = summary.quality["faithfulness"]

    assert observation.valid_samples == 4
    assert observation.failed_samples == 1
    assert observation.missing_samples == 0
    assert observation.status == "partial"


@pytest.mark.asyncio
async def test_unattributable_older_v2_execution_scope_fails_closed(
    research_service,
) -> None:
    await _campaign("missing-mode", ["naive"])
    result_id = await _result("missing-mode", "naive", "official-attempt")
    await _official_scope("missing-mode", result_id, "official-attempt")
    await _execution_scope(
        "missing-mode",
        "unknown-result",
        "unknown-attempt",
        official=False,
        scope_status="failed",
        scope_run_id="unknown-run",
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="missing-mode"
    )

    assert summary.token_accounting_status == "partial"
    assert summary.phase_attribution_status == "partial"
    assert summary.modes[0].tokens.accounting_status == "partial"
    assert summary.modes[0].comparable is False
    assert "incomplete_accounting" in summary.modes[0].not_comparable_reasons
    assert "incomplete_pricing" in summary.modes[0].not_comparable_reasons
    assert summary.modes[0].execution_cost.benchmark_usd == pytest.approx(0.1)
    assert summary.modes[0].execution_cost.operational_usd is None
    assert summary.modes[0].execution_cost.pricing_status == "partial"
    assert summary.modes[0].execution_cost.priced_call_count == 1
    assert summary.modes[0].execution_cost.unpriced_call_count == 0
    assert summary.execution_cost.benchmark_usd == pytest.approx(0.1)
    assert summary.execution_cost.operational_usd == pytest.approx(0.2)
    assert summary.execution_cost.pricing_status == "complete"
    assert summary.execution_cost.priced_call_count == 2
    assert summary.execution_cost.unpriced_call_count == 0
    assert any(
        warning.code == "missing_mode_attribution" for warning in summary.warnings
    )


@pytest.mark.asyncio
async def test_missing_faithfulness_stays_null_and_primary_metrics_are_present(
    research_service,
) -> None:
    await _campaign("partial", ["naive"])
    result_id = await _result("partial", "naive", "attempt-1")
    await _official_scope("partial", result_id, "attempt-1")
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="failed-faithfulness",
            campaign_id="partial",
            scope_type="ragas_batch",
            scope_key="faithfulness",
            metric_name="faithfulness",
            targets=[
                {
                    "job_id": "ragas",
                    "work_item_id": "faithfulness",
                    "attempt_id": "faithfulness-attempt",
                }
            ],
        )
    )
    await store.finalize_scope("failed-faithfulness", "failed")
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
        scope_run_id="execution-worker-retry-uuid",
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
    assert naive_mode.execution_cost.pricing_status == "partial"


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
async def test_running_ragas_batches_mark_primary_and_optional_metrics_evaluating(
    research_service,
) -> None:
    await _campaign("active-work", ["naive"])
    result_id = await _result("active-work", "naive", "attempt")
    await _official_scope("active-work", result_id, "attempt")
    store = EvaluationAccountingStore()
    for metric_name in ("answer_correctness", "context_precision"):
        await store.start_scope(
            AccountingScopeStart(
                scope_id=f"running-{metric_name}",
                campaign_id="active-work",
                scope_type="ragas_batch",
                scope_key=metric_name,
                metric_name=metric_name,
                targets=[
                    {
                        "job_id": "ragas",
                        "work_item_id": metric_name,
                        "attempt_id": f"attempt-{metric_name}",
                    }
                ],
            )
        )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="active-work"
    )

    assert summary.quality["answer_correctness"].status == "evaluating"
    assert summary.quality["context_precision"].status == "evaluating"
    assert summary.quality["faithfulness"].status == "not_requested"


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
                "evaluation_signature": "input-sig-first",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "policy-a",
                },
            },
            {
                "campaign_result_id": second,
                "metric_name": "answer_correctness",
                "metric_value": 0.2,
                "source_attempt_id": "second",
                "evaluation_signature": "input-sig-second",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "policy-b",
                },
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
async def test_fully_scored_primary_metrics_compare_each_evaluator_cohort_separately(
    research_service,
) -> None:
    await _campaign("fully-scored", ["naive"])
    results = [
        await _result(
            "fully-scored",
            "naive",
            f"attempt-{index}",
            run_number=index,
        )
        for index in range(1, 6)
    ]
    for index, result_id in enumerate(results, start=1):
        await _official_scope("fully-scored", result_id, f"attempt-{index}")

    metrics = ("answer_correctness", "faithfulness", "answer_relevancy")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="fully-scored",
        score_rows=[
            {
                "campaign_result_id": result_id,
                "metric_name": metric_name,
                "metric_value": 0.8,
                "source_attempt_id": f"attempt-{index}",
                "evaluation_signature": f"input-{metric_name}-{index}",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": f"policy-{metric_name}",
                },
            }
            for index, result_id in enumerate(results, start=1)
            for metric_name in metrics
        ],
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="fully-scored"
    )
    mode = summary.modes[0]

    assert all(mode.quality[metric].status == "complete" for metric in metrics)
    assert all(mode.quality[metric].valid_samples == 5 for metric in metrics)
    assert mode.comparable is True
    assert "evaluator_metadata_mismatch" not in mode.not_comparable_reasons
    assert not any(
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
                "evaluation_signature": "input-sig-n1",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "answer-policy",
                },
            },
            {
                "campaign_result_id": naive_two,
                "metric_name": "answer_correctness",
                "metric_value": 0.1,
                "source_attempt_id": "n2",
                "evaluation_signature": "input-sig-n2",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "answer-policy",
                },
            },
            {
                "campaign_result_id": graph,
                "metric_name": "answer_correctness",
                "metric_value": 0.9,
                "source_attempt_id": "g1",
                "evaluation_signature": "input-sig-g1",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "answer-policy",
                },
            },
            {
                "campaign_result_id": graph,
                "metric_name": "context_precision",
                "metric_value": 0.6,
                "source_attempt_id": "g1",
                "evaluation_signature": "input-sig-context-g1",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "context-policy",
                },
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
