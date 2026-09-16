"""Bound page payloads while retaining campaign-wide denominators."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from core.auth import get_current_user_id
from evaluation.campaign_schemas import QuestionComparisonRow, ResearchQuestionComparisonResponse
from evaluation.router import router, get_research_analytics_service


def test_question_pages_preserve_totals_and_bound_duplicate_summaries(monkeypatch):
    monkeypatch.delenv("EVALUATION_DATABASE_URL", raising=False)
    rows = [QuestionComparisonRow(question_id=f"Q{i}") for i in range(103)]
    service = AsyncMock()
    service.get_question_comparison.return_value = ResearchQuestionComparisonResponse(
        campaign_id="campaign", analysis_unit="question", sample_count=206,
        independent_question_count=103, rows=rows,
        summaries={row.question_id: row.model_dump(mode="json") for row in rows},
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/evaluation")
    app.dependency_overrides[get_current_user_id] = lambda: "owner"
    app.dependency_overrides[get_research_analytics_service] = lambda: service
    with TestClient(app) as client:
        path = "/api/evaluation/campaigns/campaign/research-question-comparison"
        received = []
        for offset in (0, 50, 100):
            response = client.get(path, params={"offset": offset})
            assert response.status_code == 200
            page = response.json()
            assert page["sample_count"] == 206
            assert page["independent_question_count"] == 103
            assert len(page["rows"]) <= 50
            assert len(page["summaries"]) == len(page["rows"])
            received.extend(row["question_id"] for row in page["rows"])
        assert page["next_offset"] is None
        assert received == [row.question_id for row in rows]
        assert client.get(path, params={"limit": 101}).status_code == 422
        assert client.get(path, params={"offset": -1}).status_code == 422
    service.get_question_comparison.assert_awaited_with(user_id="owner", campaign_id="campaign")
