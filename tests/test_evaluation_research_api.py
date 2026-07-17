"""HTTP contract tests for strict research accounting."""

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from evaluation.accounting_schemas import (
    CampaignResearchSummaryResponse,
    CostSummary,
    EvaluationOverheadSummary,
    LatencySummary,
    TokenBreakdown,
)
from evaluation.research_analytics import ResearchAnalyticsService
from evaluation.router import get_research_analytics_service
from evaluation import db as evaluation_db
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


def test_research_summary_requires_auth_and_serializes_nulls() -> None:
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_research_analytics_service] = (
        lambda: _ResearchService()
    )
    try:
        db_path = (
            Path("output")
            / "test_tmp"
            / f"research_api_{uuid4().hex}"
            / "evaluation.db"
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
            patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
            patch.object(evaluation_db, "EVALUATION_DB_PATH", db_path),
            TestClient(app) as client,
        ):
            response = client.get(
                "/api/evaluation/campaigns/campaign-1/research-summary"
            )
        assert response.status_code == 200
        body = response.json()
        assert body["tokens"]["total_tokens"] is None
        assert body["execution_cost"]["benchmark_usd"] is None
    finally:
        app.dependency_overrides = {}
