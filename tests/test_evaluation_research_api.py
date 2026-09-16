"""HTTP contract tests for strict research accounting."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from evaluation.accounting_schemas import (
    CampaignResearchSummaryResponse,
    CostSummary,
    EvaluationOverheadSummary,
    LatencySummary,
    TokenBreakdown,
)
from evaluation.research_analytics import ResearchAnalyticsService
from evaluation.router import get_research_analytics_service
from evaluation.campaign_schemas import CampaignConfig
from evaluation.db import CampaignRepository
from evaluation.schemas import ModelConfig
from evaluation.campaign_schemas import V9ExecutionObservability
from evaluation.trace_schemas import EvaluationRunObservabilityDetail
from main import app


class _ResearchService(ResearchAnalyticsService):
    async def get_summary(self, *, user_id: str, campaign_id: str):
        assert user_id == "user-1"
        return CampaignResearchSummaryResponse(
            campaign_id=campaign_id,
            completed_run_count=0,
            total_run_count=0,
            failed_run_count=0,
            quality_status="not_requested",
            token_accounting_status="incomplete_legacy",
            pricing_status="unknown",
            phase_attribution_status="not_available",
            sample_count=0,
            latency=LatencySummary(),
            tokens=TokenBreakdown(
                accounting_status="incomplete_legacy",
                phase_attribution_status="not_available",
            ),
            execution_cost=CostSummary(pricing_status="unknown"),
            modes=[],
            evaluation_overhead=EvaluationOverheadSummary(
                tokens=TokenBreakdown(
                    accounting_status="partial",
                    phase_attribution_status="not_available",
                ),
                pricing_status="unknown",
            ),
        )


def test_research_summary_requires_auth_and_serializes_nulls(monkeypatch) -> None:
    async def load_analysis(**kwargs):
        return await kwargs["loader"]()

    monkeypatch.setattr("evaluation.analysis_cache.read_analysis", load_analysis)
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_research_analytics_service] = (
        lambda: _ResearchService()
    )
    try:
        with (
            patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
            patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
            TestClient(app) as client,
        ):
            response = client.get(
                "/api/evaluation/campaigns/campaign-1/research-summary"
            )
        assert response.status_code == 200
        body = response.json()
        assert body["tokens"]["total_tokens"] is None
        assert body["execution_cost"]["benchmark_usd"] is None
        assert body["evaluation_overhead"]["retry_count"] is None
    finally:
        app.dependency_overrides = {}


def test_research_summary_hides_campaign_owned_by_another_user(tmp_path, run_db) -> None:

    async def seed_campaign() -> str:
        campaign = await CampaignRepository().create(
            user_id="other-user",
            name="private",
            config=CampaignConfig(
                test_case_ids=["q-1"],
                modes=["naive"],
                model_config=ModelConfig(
                    id="model-1", name="Model", model_name="model"
                ),
            ),
        )
        return campaign.id

    campaign_id = run_db(seed_campaign())
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_research_analytics_service] = (
        lambda: ResearchAnalyticsService()
    )
    try:
        with (
            patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
            patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
            TestClient(app) as client,
        ):
            response = client.get(
                f"/api/evaluation/campaigns/{campaign_id}/research-summary"
            )
        assert response.status_code == 404
    finally:
        app.dependency_overrides = {}


def test_campaign_run_observability_delegates_and_hides_unowned_runs() -> None:
    class ObservabilityService(ResearchAnalyticsService):
        async def get_run_observability(
            self, *, user_id: str, campaign_id: str, run_id: str
        ):
            if user_id != "user-1":
                raise AppError(
                    code=ErrorCode.NOT_FOUND,
                    message="Campaign result not found",
                    status_code=404,
                )
            assert (campaign_id, run_id) == ("cmp-1", "run-1")
            return EvaluationRunObservabilityDetail(
                run_id=run_id,
                campaign_id=campaign_id,
                accounting_diagnostics=TokenBreakdown(
                    accounting_status="complete", phase_attribution_status="complete"
                ),
                agentic_v9=V9ExecutionObservability(schema_version="1"),
            )

    service = ObservabilityService()
    app.dependency_overrides[get_research_analytics_service] = lambda: service
    try:
        with (
            patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
            patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        ):
            app.dependency_overrides[get_current_user_id] = lambda: "user-1"
            with TestClient(app) as client:
                owned = client.get(
                    "/api/evaluation/campaigns/cmp-1/runs/run-1/observability"
                )
            assert owned.status_code == 200
            assert owned.json()["agentic_v9"]["schema_version"] == "1"

            app.dependency_overrides[get_current_user_id] = lambda: "other-user"
            with TestClient(app) as client:
                unowned = client.get(
                    "/api/evaluation/campaigns/cmp-1/runs/run-1/observability"
                )
            assert unowned.status_code == 404
    finally:
        app.dependency_overrides = {}
