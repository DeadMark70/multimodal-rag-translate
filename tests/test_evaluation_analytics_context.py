from types import SimpleNamespace

import pytest

from evaluation.analytics import EvaluationAnalyticsService


class FakeCampaignRepository:
    async def get(self, *, user_id: str, campaign_id: str):
        return SimpleNamespace(id=campaign_id)


class CountingResultRepository:
    def __init__(self):
        self.list_calls = 0
        self.results = [
            SimpleNamespace(
                id="run-1",
                campaign_id="campaign-1",
                question_id="Q1",
                mode="agentic",
                run_number=1,
                repeat_number=1,
                total_latency_ms=100,
                latency_ms=120,
                total_tokens=30,
                derived_metrics={"unsupported_claim_ratio": 0.1, "evidence_coverage": 0.9},
            ),
            SimpleNamespace(
                id="run-2",
                campaign_id="campaign-1",
                question_id="Q2",
                mode="agentic",
                run_number=1,
                repeat_number=1,
                total_latency_ms=200,
                latency_ms=220,
                total_tokens=70,
                derived_metrics={"unsupported_claim_ratio": 0.2, "evidence_coverage": 0.8},
            ),
        ]

    async def list_for_campaign(self, *, user_id: str, campaign_id: str):
        self.list_calls += 1
        return self.results


class CountingObservabilityRepository:
    def __init__(self):
        self.bulk_llm_calls: list[str] = []
        self.per_run_llm_calls: list[str] = []

    async def list_llm_calls_for_campaign(self, campaign_id: str):
        self.bulk_llm_calls.append(campaign_id)
        return {
            "run-1": [
                SimpleNamespace(
                    campaign_id=campaign_id,
                    estimated_cost_usd=0.01,
                    estimated_cost_twd=0.32,
                )
            ],
            "run-2": [
                SimpleNamespace(
                    campaign_id=campaign_id,
                    estimated_cost_usd=0.02,
                    estimated_cost_twd=0.64,
                )
            ],
        }

    async def list_llm_calls_for_run(self, run_id: str):
        self.per_run_llm_calls.append(run_id)
        return []


@pytest.mark.asyncio
async def test_campaign_overview_uses_bulk_llm_calls() -> None:
    result_repository = CountingResultRepository()
    observability_repository = CountingObservabilityRepository()
    service = EvaluationAnalyticsService(
        campaign_repository=FakeCampaignRepository(),
        result_repository=result_repository,
        observability_repository=observability_repository,
    )

    overview = await service.campaign_overview(user_id="user-a", campaign_id="campaign-1")

    assert overview.sample_count == 2
    assert overview.total_cost_usd == pytest.approx(0.03)
    assert observability_repository.bulk_llm_calls == ["campaign-1"]
    assert observability_repository.per_run_llm_calls == []
    assert result_repository.list_calls == 1


@pytest.mark.asyncio
async def test_mode_comparison_reuses_single_campaign_context() -> None:
    result_repository = CountingResultRepository()
    observability_repository = CountingObservabilityRepository()
    service = EvaluationAnalyticsService(
        campaign_repository=FakeCampaignRepository(),
        result_repository=result_repository,
        observability_repository=observability_repository,
    )

    response = await service.mode_comparison(user_id="user-a", campaign_id="campaign-1")

    assert response.sample_count == 2
    assert response.summaries["agentic"]["sample_count"] == 2
    assert observability_repository.bulk_llm_calls == ["campaign-1"]
    assert observability_repository.per_run_llm_calls == []
    assert result_repository.list_calls == 1
