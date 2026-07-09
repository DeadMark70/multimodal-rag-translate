from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from evaluation.analytics import EvaluationAnalyticsService
from evaluation.campaign_schemas import ExportCampaignRequest
from evaluation.trace_schemas import (
    EvaluationClaim,
    EvaluationLlmCall,
    EvaluationRetrievalChunk,
    EvaluationTraceEvent,
)


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


class FakeCampaign:
    id = "campaign-1"

    def model_dump(self, mode: str = "json"):
        return {"id": self.id}


class FakeResult:
    id = "run-1"
    campaign_id = "campaign-1"
    question_id = "Q1"
    question = "Question?"
    ground_truth = "Answer"
    mode = "agentic"
    run_number = 1
    repeat_number = 1
    answer = "Answer"
    contexts = ["context"]
    source_doc_ids = ["doc-1"]
    expected_sources = ["doc-1"]
    latency_ms = 100
    total_latency_ms = 120
    total_tokens = 42
    token_usage = {"total_tokens": 42}
    derived_metrics = {}
    execution_profile = None
    error_message = None
    created_at = datetime(2026, 7, 8, tzinfo=timezone.utc)

    def model_dump(self, mode: str = "json"):
        return {
            "id": self.id,
            "campaign_id": self.campaign_id,
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "mode": self.mode,
            "run_number": self.run_number,
            "answer": self.answer,
            "contexts": self.contexts,
            "source_doc_ids": self.source_doc_ids,
            "expected_sources": self.expected_sources,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "status": "completed",
            "created_at": self.created_at.isoformat(),
            "derived_metrics": self.derived_metrics,
        }


class SingleRunCampaignRepository:
    async def get(self, *, user_id: str, campaign_id: str):
        return FakeCampaign()


class SingleRunResultRepository:
    async def list_for_campaign(self, *, user_id: str, campaign_id: str):
        return [FakeResult()]


class BulkOnlyObservabilityRepository:
    def __init__(self):
        self.bulk_trace_calls: list[str] = []
        self.bulk_llm_calls: list[str] = []
        self.bulk_chunk_calls: list[str] = []
        self.bulk_claim_calls: list[str] = []
        self.per_run_calls: list[str] = []
        now = datetime(2026, 7, 8, tzinfo=timezone.utc)
        self.trace_event = EvaluationTraceEvent(
            event_id="trace-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            parent_event_id=None,
            parent_span_id=None,
            event_type="span_completed",
            event_schema_version="1.0",
            sequence=1,
            stage_type="generation",
            stage_name="answer_generation",
            started_at=now,
            ended_at=now,
            duration_ms=10,
            status="failed",
            retry_count=0,
            payload={"full": "payload"},
            error={"code": "PROVIDER_ERROR", "message": "apiKey=sk-secret stack trace"},
            created_at=now,
        )
        self.llm_call = EvaluationLlmCall(
            llm_call_id="llm-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            provider="gemini",
            model_name="gemini-2.5-flash",
            purpose="generation",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.01,
            estimated_cost_twd=0.32,
            prompt_hash="prompt-hash",
            prompt_preview="prompt preview",
            response_hash="response-hash",
            latency_ms=20,
            status="failed",
            error={"code": "LLM_FAILED", "message": "provider failed"},
            payload={"full_prompt": "SECRET FULL PROMPT"},
            created_at=now,
        )
        self.chunk = EvaluationRetrievalChunk(
            retrieval_chunk_id="chunk-row-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            retrieval_event_id="retrieval-1",
            chunk_id="chunk-1",
            doc_id="paper.pdf",
            page_start=1,
            page_end=1,
            modality="text",
            rank_before_rerank=1,
            rank_after_rerank=1,
            dense_score=None,
            bm25_score=None,
            rerank_score=None,
            used_in_context=True,
            used_in_answer=True,
            expected_evidence_match=False,
            excerpt="retrieved excerpt",
            content_hash="content-hash",
            payload={},
            created_at=now,
        )
        self.claim = EvaluationClaim(
            claim_id="claim-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            claim_text="Claim",
            claim_type="fact",
            support_status="supported",
            evidence=[],
            unsupported_reason=None,
            payload={},
            created_at=now,
        )

    async def list_trace_events_for_campaign(self, campaign_id: str):
        self.bulk_trace_calls.append(campaign_id)
        return {"run-1": [self.trace_event]}

    async def list_llm_calls_for_campaign(self, campaign_id: str):
        self.bulk_llm_calls.append(campaign_id)
        return {"run-1": [self.llm_call]}

    async def list_retrieval_chunks_for_campaign(self, campaign_id: str):
        self.bulk_chunk_calls.append(campaign_id)
        return {"run-1": [self.chunk]}

    async def list_claims_for_campaign(self, campaign_id: str):
        self.bulk_claim_calls.append(campaign_id)
        return {"run-1": [self.claim]}

    async def list_trace_events_for_run(self, run_id: str):
        self.per_run_calls.append(f"trace:{run_id}")
        raise AssertionError("campaign analytics must not use per-run trace reads")

    async def list_llm_calls_for_run(self, run_id: str):
        self.per_run_calls.append(f"llm:{run_id}")
        raise AssertionError("campaign analytics must not use per-run llm reads")

    async def list_retrieval_chunks_for_run(self, run_id: str):
        self.per_run_calls.append(f"chunks:{run_id}")
        raise AssertionError("campaign analytics must not use per-run retrieval chunk reads")

    async def list_claims_for_run(self, run_id: str):
        self.per_run_calls.append(f"claims:{run_id}")
        raise AssertionError("campaign analytics must not use per-run claim reads")


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


@pytest.mark.asyncio
async def test_campaign_errors_uses_bulk_observability() -> None:
    observability_repository = BulkOnlyObservabilityRepository()
    service = EvaluationAnalyticsService(
        campaign_repository=SingleRunCampaignRepository(),
        result_repository=SingleRunResultRepository(),
        observability_repository=observability_repository,
    )

    response = await service.campaign_errors(user_id="user-a", campaign_id="campaign-1")

    assert [row.source for row in response.rows] == ["trace", "llm_call"]
    assert observability_repository.bulk_trace_calls == ["campaign-1"]
    assert observability_repository.bulk_llm_calls == ["campaign-1"]
    assert observability_repository.per_run_calls == []


@pytest.mark.asyncio
async def test_export_campaign_uses_bulk_observability() -> None:
    observability_repository = BulkOnlyObservabilityRepository()
    service = EvaluationAnalyticsService(
        campaign_repository=SingleRunCampaignRepository(),
        result_repository=SingleRunResultRepository(),
        observability_repository=observability_repository,
    )

    response = await service.export_campaign(
        user_id="user-a",
        campaign_id="campaign-1",
        request=ExportCampaignRequest(),
    )

    assert response.llm_calls[0]["llm_call_id"] == "llm-1"
    assert "full_prompt" not in response.llm_calls[0]["payload"]
    assert response.retrieval_summary[0]["chunk_count"] == 1
    assert response.claim_summary[0]["claims"][0]["claim_id"] == "claim-1"
    assert observability_repository.bulk_trace_calls == ["campaign-1"]
    assert observability_repository.bulk_llm_calls == ["campaign-1"]
    assert observability_repository.bulk_chunk_calls == ["campaign-1"]
    assert observability_repository.bulk_claim_calls == ["campaign-1"]
    assert observability_repository.per_run_calls == []
