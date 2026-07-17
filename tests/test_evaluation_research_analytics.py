"""Strict research-summary contract regression tests using durable repositories."""

from datetime import datetime, timezone

import pytest
import pytest_asyncio

from evaluation import db as evaluation_db
from evaluation.accounting_schemas import AccountingScopeStart, UsageEventCreate
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.campaign_schemas import CampaignConfig, CampaignResultStatus
from evaluation.db import CampaignRepository, CampaignResultRepository, RagasScoreRepository
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
        user_id="user-1", name=campaign_id,
        config=CampaignConfig(
            test_case_ids=["q-1"], modes=modes,
            model_config=ModelConfig(id="model-1", name="Model", model_name="model"),
        ),
    )
    # Repository-generated IDs are intentionally not assumed by the fixture.
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute("UPDATE campaigns SET id = ? WHERE name = ?", (campaign_id, campaign_id))
        await connection.commit()


async def _result(campaign_id: str, mode: str, attempt: str, *, latency: float = 100.0) -> str:
    result = await CampaignResultRepository().create(
        user_id="user-1", campaign_id=campaign_id, question_id=f"q-{mode}", question="Q",
        ground_truth="A", ground_truth_short=None, key_points=[], ragas_focus=[], mode=mode,
        execution_profile="v2", context_policy_version="v2", run_number=1, answer="A", contexts=[],
        source_doc_ids=[], expected_sources=[], latency_ms=latency, token_usage={}, category=None,
        difficulty=None, status=CampaignResultStatus.COMPLETED, total_latency_ms=latency,
        source_attempt_id=attempt,
    )
    return result.id


async def _official_scope(campaign_id: str, result_id: str, attempt: str) -> None:
    store = EvaluationAccountingStore()
    await store.start_scope(AccountingScopeStart(
        scope_id=f"scope-{attempt}", campaign_id=campaign_id, scope_type="execution_run",
        scope_key=attempt, run_id=result_id,
        targets=[{"job_id": "job-1", "work_item_id": "work-1", "attempt_id": attempt, "campaign_result_id": result_id, "is_official": True}],
    ))
    await store.record_event(UsageEventCreate(
        usage_event_id=f"event-{attempt}", scope_id=f"scope-{attempt}", campaign_id=campaign_id,
        scope_type="execution_run", scope_key=attempt, run_id=result_id, phase="answer_generation",
        purpose="evaluation", input_tokens=10, output_text_tokens=5, reported_total_tokens=15,
        usage_status="measured", reconciliation_status="balanced", estimated_cost_usd=0.1,
        pricing_status="priced", created_at=datetime.now(timezone.utc),
    ))
    await store.finalize_scope(f"scope-{attempt}", "completed")


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
async def test_missing_faithfulness_stays_null_and_primary_metrics_are_present(research_service) -> None:
    await _campaign("partial", ["naive"])
    result_id = await _result("partial", "naive", "attempt-1")
    await _official_scope("partial", result_id, "attempt-1")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1", campaign_id="partial", score_rows=[
            {"campaign_result_id": result_id, "metric_name": "answer_correctness", "metric_value": .8, "source_attempt_id": "attempt-1", "evaluation_signature": "ragas-v1", "details": {"evaluator_model": "judge", "metric_version": "v1"}},
            {"campaign_result_id": result_id, "metric_name": "answer_relevancy", "metric_value": .7, "source_attempt_id": "attempt-1", "evaluation_signature": "ragas-v1", "details": {"evaluator_model": "judge", "metric_version": "v1"}},
        ],
    )

    summary = await research_service.get_summary(user_id="user-1", campaign_id="partial")

    assert set(summary.modes[0].quality) == {"answer_correctness", "faithfulness", "answer_relevancy"}
    faithfulness = summary.modes[0].quality["faithfulness"]
    assert faithfulness.value is None
    assert faithfulness.status == "failed"
    assert summary.quality["faithfulness"].value is None
    assert summary.quality["faithfulness"].status == "failed"
