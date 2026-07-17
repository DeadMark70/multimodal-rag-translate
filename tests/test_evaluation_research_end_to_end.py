"""End-to-end proof for version-2 evaluation research accounting."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest
import pytest_asyncio

from core.auth import get_current_user_id
from core.llm_usage_context import emit_direct_usage, llm_accounting_phase
import evaluation.db as evaluation_db
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.execution_worker import DatasetExecutionWorker
from evaluation.job_schemas import (
    ClaimedEvaluationWork,
    EvaluationWorkType,
    WorkItemSpec,
)
from evaluation.job_store import EvaluationJobStore
from evaluation.rag_modes import BenchmarkExecutionResult
from evaluation.ragas_worker import RagasBatchWorker
from evaluation.db import RagasScoreRepository
from main import app


EXPECTED_PHASES = {
    "naive": ["answer_generation"],
    "advanced": ["query_expansion", "answer_generation"],
    "graph": ["query_expansion", "graph_reasoning", "answer_generation"],
    "agentic": ["agent_planning", "answer_generation", "agent_synthesis"],
}

PRICE_SNAPSHOT = {
    "snapshot_id": "e2e-accounting-v1",
    "currency": "USD",
    "usd_to_twd": None,
    "models": {
        "test-model": {
            "input_per_1m_usd": 1.0,
            "output_per_1m_usd": 2.0,
            "reasoning_per_1m_usd": 0.0,
        }
    },
}


class _RagasPromotionStore:
    async def complete_ragas_attempt(self, _claim, output):  # noqa: ANN001
        return len(output.scores)

    async def fail_attempt(self, _claim, _decision, *, next_retry_at=None):  # noqa: ANN001
        raise AssertionError("the deterministic RAGAS provider must not fail")


class _BatchedRagasProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def evaluate_metric_batch(self, metric_name, rows, _llm, _embeddings):  # noqa: ANN001
        self.calls.append((metric_name, len(rows)))
        await emit_direct_usage(
            purpose="ragas_evaluator",
            provider="google",
            model_name="test-model",
            raw_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        return [0.8] * len(rows)


@pytest_asyncio.fixture
async def durable_store(tmp_path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    await evaluation_db.force_init_db()
    now = datetime.now(UTC).isoformat()
    config = json.dumps(
        {
            "test_case_ids": ["question-1"],
            "modes": list(EXPECTED_PHASES),
            "model_config": {
                "id": "test-model-config",
                "name": "Test model",
                "model_name": "test-model",
            },
        }
    )
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """INSERT INTO campaigns (id, user_id, name, status, config_json, created_at, updated_at)
               VALUES ('campaign-e2e', 'user-e2e', 'E2E', 'pending', ?, ?, ?)""",
            (config, now, now),
        )
        await connection.commit()
    return EvaluationJobStore()


async def _seed_claims(store: EvaluationJobStore) -> list[ClaimedEvaluationWork]:
    await store.create_job_with_items(
        user_id="user-e2e",
        campaign_id="campaign-e2e",
        job_type="initial",
        selection={},
        config_snapshot={"skip_ragas": True},
        items=[
            WorkItemSpec(
                work_type=EvaluationWorkType.DATASET_EXECUTION,
                logical_key=f"execution:question-1:{mode}:1:none",
                input_snapshot={
                    "user_id": "user-e2e",
                    "campaign_id": "campaign-e2e",
                    "test_case": {
                        "id": "question-1",
                        "question": "What is the evaluated answer?",
                        "ground_truth": "The evaluated answer.",
                        "source_docs": [],
                        "requires_multi_doc_reasoning": False,
                    },
                    "mode": mode,
                    "run_number": 1,
                    "repeat_number": 1,
                    "model_config": {"model_name": "test-model"},
                },
            )
            for mode in EXPECTED_PHASES
        ],
    )
    return await store.claim_ready_items(limit=4, now=datetime.now(UTC))


def _runner(**inputs):  # noqa: ANN003
    async def run() -> BenchmarkExecutionResult:
        for phase in EXPECTED_PHASES[inputs["mode"]]:
            with llm_accounting_phase(phase):
                await emit_direct_usage(
                    purpose="evaluation_runner",
                    provider="google",
                    model_name="test-model",
                    raw_usage={
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                )
        return BenchmarkExecutionResult(
            question_id=inputs["test_case"].id,
            question=inputs["test_case"].question,
            ground_truth=inputs["test_case"].ground_truth,
            mode=inputs["mode"],
            answer="The evaluated answer.",
        )

    return run()


def _ragas_claim(result, metric_name: str) -> ClaimedEvaluationWork:  # noqa: ANN001
    return ClaimedEvaluationWork(
        job_id=f"ragas-{metric_name}",
        job_item_id=f"ragas-item-{metric_name}-{result.id}",
        work_item_id=f"ragas-work-{metric_name}-{result.id}",
        attempt_id=f"ragas-attempt-{metric_name}-{result.id}",
        logical_key=f"ragas:{result.id}:{metric_name}:sig-v1",
        work_type="ragas_metric",
        input_snapshot={
            "user_id": "user-e2e",
            "campaign_id": "campaign-e2e",
            "campaign_result_id": result.id,
            "metric_name": metric_name,
            "evaluation_signature": "sig-v1",
            "result": {
                "id": result.id,
                "question_id": result.question_id,
                "question": result.question,
                "answer": result.answer,
                "contexts": result.contexts,
                "ground_truth": result.ground_truth,
                "context_policy_version": "v2",
            },
        },
    )


@pytest.mark.asyncio
async def test_research_summary_accounts_for_every_mode_and_ragas_overhead(
    durable_store: EvaluationJobStore,
) -> None:
    claims = await _seed_claims(durable_store)
    accounting_store = EvaluationAccountingStore()
    worker = DatasetExecutionWorker(
        store=durable_store,
        runner=_runner,
        accounting_store=accounting_store,
        price_snapshot=PRICE_SNAPSHOT,
    )
    for claim in claims:
        await worker.execute(claim)

    results = await evaluation_db.CampaignResultRepository().list_for_campaign(
        user_id="user-e2e", campaign_id="campaign-e2e"
    )
    assert {result.mode for result in results} == set(EXPECTED_PHASES)

    provider = _BatchedRagasProvider()
    ragas_worker = RagasBatchWorker(
        store=_RagasPromotionStore(),
        evaluator=provider,
        accounting_store=accounting_store,
        price_snapshot=PRICE_SNAPSHOT,
    )
    for metric_name in ("faithfulness", "answer_correctness"):
        await ragas_worker.execute(
            [_ragas_claim(result, metric_name) for result in results]
        )
    assert provider.calls == [("faithfulness", 4), ("answer_correctness", 4)]

    await RagasScoreRepository().replace_for_campaign(
        user_id="user-e2e",
        campaign_id="campaign-e2e",
        score_rows=[
            {
                "campaign_result_id": result.id,
                "metric_name": metric_name,
                "metric_value": 0.8,
                "source_attempt_id": result.source_attempt_id,
                "evaluation_signature": "sig-v1",
                "details": {"evaluator_model": "judge", "metric_version": "v1"},
            }
            for result in results
            for metric_name in ("faithfulness", "answer_correctness")
        ],
    )

    app.dependency_overrides[get_current_user_id] = lambda: "user-e2e"
    try:
        with (
            patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
            patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
            TestClient(app) as client,
        ):
            response = client.get(
                "/api/evaluation/campaigns/campaign-e2e/research-summary"
            )
    finally:
        app.dependency_overrides = {}

    assert response.status_code == 200
    body = response.json()
    assert body["research_schema_version"] == "2"
    assert body["token_accounting_status"] == "complete"
    assert body["pricing_status"] == "complete"
    assert body["phase_attribution_status"] == "complete"
    assert body["tokens"]["total_tokens"] == 135
    assert body["execution_cost"] == {
        "benchmark_usd": pytest.approx(0.00018),
        "operational_usd": pytest.approx(0.00018),
        "pricing_status": "complete",
        "priced_call_count": 9,
        "unpriced_call_count": 0,
    }
    assert body["tokens"]["by_phase"] == {
        "agent_planning": 15,
        "agent_synthesis": 15,
        "answer_generation": 60,
        "graph_reasoning": 15,
        "query_expansion": 30,
    }
    assert body["evaluation_overhead"]["tokens"]["total_tokens"] == 30
    assert body["evaluation_overhead"]["batch_count"] == 2
    assert body["evaluation_overhead"]["cost_usd"] == pytest.approx(0.00004)
    assert body["quality"]["faithfulness"]["value"] == pytest.approx(0.8)
    assert body["quality"]["answer_correctness"]["value"] == pytest.approx(0.8)
    assert body["quality"]["answer_relevancy"]["status"] == "not_requested"
    assert {mode["mode"] for mode in body["modes"]} == set(EXPECTED_PHASES)
    assert all(mode["comparable"] is False for mode in body["modes"])
    assert all(
        "incomplete_quality" in mode["not_comparable_reasons"] for mode in body["modes"]
    )
