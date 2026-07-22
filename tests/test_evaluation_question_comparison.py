from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from evaluation.research_analytics import ResearchAnalyticsService


def _result(result_id: str, mode: str, score: float, *, tokens: int | None = 100):
    return SimpleNamespace(
        id=result_id,
        campaign_id="campaign-1",
        question_id="Q1",
        question="Which answer?",
        mode=mode,
        run_number=1,
        repeat_number=1,
        source_attempt_id=f"attempt-{result_id}",
        status="completed",
        category="medical",
        difficulty="hard",
        total_tokens=tokens,
        total_latency_ms=100.0 if mode == "naive" else 150.0,
        latency_ms=100.0 if mode == "naive" else 150.0,
        question_snapshot={"required_modalities": ["text"]},
    )


def _scope(result_id: str, *, partial: bool = False):
    return SimpleNamespace(
        scope_id=f"scope-{result_id}",
        scope_type="execution_run",
        status="completed",
        accounting_schema_version="2",
        run_id=result_id,
        observed_call_count=1,
        measured_call_count=1 if not partial else 0,
        missing_usage_call_count=0 if not partial else 1,
        targets=[
            SimpleNamespace(
                campaign_result_id=result_id,
                attempt_id=f"attempt-{result_id}",
                mode="naive" if result_id == "naive-1" else "agentic",
                is_official=True,
            )
        ],
    )


def _event(result_id: str, *, partial: bool = False):
    return SimpleNamespace(
        scope_id=f"scope-{result_id}",
        usage_status="missing" if partial else "measured",
        reconciliation_status="partial" if partial else "balanced",
        phase="answer_generation",
        input_tokens=60,
        output_text_tokens=40,
        reasoning_tokens=0,
        other_tokens=0,
    )


class _Campaigns:
    def __init__(self, modes=None):
        self.modes = modes

    async def get(self, *, user_id: str, campaign_id: str):
        return SimpleNamespace(
            id=campaign_id,
            status="completed",
            updated_at=datetime.now(timezone.utc),
            config=SimpleNamespace(modes=self.modes or []),
        )


class _Results:
    def __init__(self, rows):
        self.rows = rows

    async def list_for_campaign(self, **kwargs):
        return self.rows


class _Scores:
    async def list_for_campaign(self, **kwargs):
        return [
            self._row("naive-1", "answer_correctness", 0.5),
            self._row("naive-1", "faithfulness", 0.8),
            self._row("naive-1", "answer_relevancy", 0.7),
            self._row("agentic-1", "answer_correctness", 0.9),
            self._row("agentic-1", "faithfulness", 0.8),
            self._row("agentic-1", "answer_relevancy", 0.6),
        ]

    @staticmethod
    def _row(result_id: str, metric_name: str, metric_value: float):
        return {
            "campaign_result_id": result_id,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "source_attempt_id": f"attempt-{result_id}",
            "evaluation_signature": "sig-1",
            "details": {
                "evaluator_model": "eval-v1",
                "metric_version": "v1",
                "compatibility_signature": "sig-1",
                "compatibility_signature_version": "2",
            },
        }

    async def list_work_metadata_for_campaign(self, **kwargs):
        return []


class _Accounting:
    async def list_campaign_scopes(self, campaign_id):
        return [_scope("naive-1"), _scope("agentic-1", partial=True)]

    async def list_campaign_events(self, campaign_id):
        return [_event("naive-1"), _event("agentic-1", partial=True)]


@pytest.mark.asyncio
async def test_question_comparison_is_measured_and_fail_closed():
    service = ResearchAnalyticsService(
        campaigns=_Campaigns(),
        results=_Results([_result("naive-1", "naive", 0.5), _result("agentic-1", "agentic", 0.9)]),
        ragas_scores=_Scores(),
        accounting=_Accounting(),
    )

    response = await service.get_question_comparison(user_id="user-1", campaign_id="campaign-1")
    row = response.rows[0]

    assert row.delta_correctness == pytest.approx(0.4)
    assert row.delta_faithfulness == pytest.approx(0.0)
    assert row.delta_latency_ms == pytest.approx(50.0)
    assert row.delta_tokens is None
    assert row.ecr_correctness is None
    assert row.best_quality_mode == "agentic"
    assert row.category == "medical"
    assert row.difficulty == "hard"
    assert row.required_modalities == ["text"]
    assert row.evidence_coverage is None
    assert row.unsupported_claim_ratio is None
    assert row.comparability_reason == "incomplete_accounting"


@pytest.mark.asyncio
async def test_missing_comparison_mode_never_becomes_zero_delta():
    service = ResearchAnalyticsService(
        campaigns=_Campaigns(["naive", "agentic"]),
        results=_Results([_result("naive-1", "naive", 0.5)]),
        ragas_scores=_Scores(),
        accounting=_Accounting(),
    )

    row = (await service.get_question_comparison(user_id="user-1", campaign_id="campaign-1")).rows[0]

    assert row.best_quality_mode == "naive"
    assert row.delta_correctness is None
    assert row.delta_faithfulness is None
    assert row.comparability_reason == "comparison_mode_missing"


@pytest.mark.asyncio
async def test_question_comparison_uses_agentic_when_naive_is_best_quality_mode():
    class NaiveWinsScores(_Scores):
        async def list_for_campaign(self, **kwargs):
            rows = await super().list_for_campaign(**kwargs)
            for row in rows:
                if row["campaign_result_id"] == "agentic-1":
                    row["metric_value"] = 0.4
            return rows

    class CompleteAccounting(_Accounting):
        async def list_campaign_scopes(self, campaign_id):
            return [_scope("naive-1"), _scope("agentic-1")]

        async def list_campaign_events(self, campaign_id):
            return [_event("naive-1"), _event("agentic-1")]

    service = ResearchAnalyticsService(
        campaigns=_Campaigns(["naive", "agentic"]),
        results=_Results([_result("naive-1", "naive", 0.5), _result("agentic-1", "agentic", 0.4)]),
        ragas_scores=NaiveWinsScores(),
        accounting=CompleteAccounting(),
    )

    row = (await service.get_question_comparison(user_id="user-1", campaign_id="campaign-1")).rows[0]

    assert row.best_quality_mode == "naive"
    assert row.delta_correctness == pytest.approx(-0.1)
    assert row.delta_faithfulness == pytest.approx(-0.4)
    assert row.delta_tokens == pytest.approx(0.0)
    assert row.comparability_reason is None


@pytest.mark.asyncio
async def test_partial_quality_mode_cannot_win_best_mode():
    class PartialScores(_Scores):
        async def list_for_campaign(self, **kwargs):
            return [
                row
                for row in await super().list_for_campaign(**kwargs)
                if not (
                    row["campaign_result_id"] == "agentic-1"
                    and row["metric_name"] == "faithfulness"
                )
            ]

    service = ResearchAnalyticsService(
        campaigns=_Campaigns(["naive", "agentic"]),
        results=_Results([_result("naive-1", "naive", 0.5), _result("agentic-1", "agentic", 0.9)]),
        ragas_scores=PartialScores(),
        accounting=_Accounting(),
    )

    row = (await service.get_question_comparison(user_id="user-1", campaign_id="campaign-1")).rows[0]

    assert row.best_quality_mode == "naive"
    assert row.delta_correctness is None
    assert row.comparability_reason == "incomplete_quality"
