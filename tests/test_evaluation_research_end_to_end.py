"""End-to-end proof for version-2 evaluation research accounting."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest
import pytest_asyncio

from core.auth import get_current_user_id
from core.llm_usage_context import (
    emit_direct_usage,
    llm_accounting_phase,
)
import evaluation.db as evaluation_db
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.db import RagasScoreRepository
from evaluation.execution_worker import DatasetExecutionWorker
from evaluation.job_schemas import (
    ClaimedEvaluationWork,
    EvaluationWorkType,
    WorkItemSpec,
)
from evaluation.job_store import EvaluationJobStore
from evaluation.rag_modes import BenchmarkExecutionResult
from evaluation.ragas_worker import RagasBatchWorker
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


class _BatchedRagasProvider:
    evaluator_model = "test-model"

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
                max_attempts=2 if mode == "naive" else 1,
            )
            for mode in EXPECTED_PHASES
        ],
    )
    return await store.claim_ready_items(limit=4, now=datetime.now(UTC))


class _ExecutionRunner:
    def __init__(self) -> None:
        self.attempts_by_mode: dict[str, int] = {}

    async def __call__(self, **inputs) -> BenchmarkExecutionResult:  # noqa: ANN003
        mode = inputs["mode"]
        attempt_number = self.attempts_by_mode.get(mode, 0) + 1
        self.attempts_by_mode[mode] = attempt_number
        fail = mode == "naive" and attempt_number == 1
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
                    status="failed" if fail else "success",
                    error={"type": "deterministic_timeout"} if fail else {},
                )
        if fail:
            raise TimeoutError("deterministic retryable execution failure")
        return BenchmarkExecutionResult(
            question_id=inputs["test_case"].id,
            question=inputs["test_case"].question,
            ground_truth=inputs["test_case"].ground_truth,
            mode=inputs["mode"],
            answer="The evaluated answer.",
        )


@pytest.mark.asyncio
async def test_research_summary_accounts_for_every_mode_and_ragas_overhead(
    durable_store: EvaluationJobStore,
) -> None:
    claims = await _seed_claims(durable_store)
    accounting_store = EvaluationAccountingStore()
    runner = _ExecutionRunner()
    worker = DatasetExecutionWorker(
        store=durable_store,
        runner=runner,
        accounting_store=accounting_store,
        price_snapshot=PRICE_SNAPSHOT,
    )
    for claim in claims:
        await worker.execute(claim)

    first_naive_claim = next(
        claim for claim in claims if claim.input_snapshot["mode"] == "naive"
    )
    assert (
        await durable_store.get_job_item_status(first_naive_claim.job_item_id)
        == "retry_wait"
    )
    first_attempts = await durable_store.list_attempts(
        user_id="user-e2e", work_item_id=first_naive_claim.work_item_id
    )
    assert [attempt.status for attempt in first_attempts] == ["failed"]
    assert [attempt.error_type for attempt in first_attempts] == ["timeout"]

    retry_at = await durable_store.next_ready_at()
    assert retry_at is not None
    retry_claims = await durable_store.claim_ready_items(limit=1, now=retry_at)
    assert len(retry_claims) == 1
    retry_claim = retry_claims[0]
    assert retry_claim.work_item_id == first_naive_claim.work_item_id
    assert retry_claim.job_item_id == first_naive_claim.job_item_id
    assert retry_claim.attempt_number == 2
    await worker.execute(retry_claim)
    assert (
        await durable_store.get_job_item_status(retry_claim.job_item_id) == "succeeded"
    )
    attempts = await durable_store.list_attempts(
        user_id="user-e2e", work_item_id=retry_claim.work_item_id
    )
    assert [attempt.status for attempt in attempts] == ["failed", "succeeded"]

    results = await evaluation_db.CampaignResultRepository().list_for_campaign(
        user_id="user-e2e", campaign_id="campaign-e2e"
    )
    assert {result.mode for result in results} == set(EXPECTED_PHASES)
    retried_result = next(result for result in results if result.mode == "naive")
    assert retried_result.source_attempt_id == retry_claim.attempt_id
    assert retried_result.source_attempt_id != first_naive_claim.attempt_id

    assert (
        await durable_store.ensure_ragas_work(
            user_id="user-e2e",
            campaign_id="campaign-e2e",
            evaluator_model="test-model",
            evaluator_config={},
            enabled_metrics=["faithfulness", "answer_correctness"],
            metric_version="v1",
            ragas_batch_size=4,
            ragas_parallel_batches=1,
        )
        == 8
    )
    ragas_claims = await durable_store.claim_ready_items(
        limit=8,
        now=datetime.now(UTC),
        work_type=EvaluationWorkType.RAGAS_METRIC,
    )
    assert len(ragas_claims) == 8

    provider = _BatchedRagasProvider()
    ragas_worker = RagasBatchWorker(
        store=durable_store,
        evaluator=provider,
        accounting_store=accounting_store,
        price_snapshot=PRICE_SNAPSHOT,
    )
    await ragas_worker.execute(ragas_claims)
    assert sorted(provider.calls) == [
        ("answer_correctness", 4),
        ("faithfulness", 4),
    ]
    score_rows = await RagasScoreRepository().list_for_campaign(
        user_id="user-e2e", campaign_id="campaign-e2e"
    )
    execution_attempts = {result.id: result.source_attempt_id for result in results}
    assert len(score_rows) == 8
    assert all(
        score["source_attempt_id"] == execution_attempts[score["campaign_result_id"]]
        for score in score_rows
    )
    assert len({score["evaluation_signature"] for score in score_rows}) == 8
    assert (
        len(
            {
                (score["metric_name"], score["details"]["compatibility_signature"])
                for score in score_rows
            }
        )
        == 2
    )
    assert {score["details"]["metric_version"] for score in score_rows} == {"v1"}

    scopes = await accounting_store.list_campaign_scopes("campaign-e2e")
    events = await accounting_store.list_campaign_events("campaign-e2e")
    events_by_scope = {
        scope.scope_id: [event for event in events if event.scope_id == scope.scope_id]
        for scope in scopes
    }
    for result in results:
        scope = next(
            scope
            for scope in scopes
            if scope.scope_type == "execution_run"
            and scope.status == "completed"
            and any(
                target.is_official and target.campaign_result_id == result.id
                for target in scope.targets
            )
        )
        assert [event.phase for event in events_by_scope[scope.scope_id]] == (
            EXPECTED_PHASES[result.mode]
        )
        assert all(
            (event.input_tokens, event.output_text_tokens, event.reported_total_tokens)
            == (10, 5, 15)
            for event in events_by_scope[scope.scope_id]
        )
        assert [target.mode for target in scope.targets] == [result.mode]
    ragas_scopes = [scope for scope in scopes if scope.scope_type == "ragas_batch"]
    failed_retry_scope = next(
        scope
        for scope in scopes
        if scope.scope_type == "execution_run"
        and any(
            target.attempt_id == first_naive_claim.attempt_id
            for target in scope.targets
        )
    )
    assert failed_retry_scope.status == "failed"
    assert not any(target.is_official for target in failed_retry_scope.targets)
    assert [target.mode for target in failed_retry_scope.targets] == ["naive"]
    assert [event.status for event in events_by_scope[failed_retry_scope.scope_id]] == [
        "failed"
    ]
    assert {scope.metric_name for scope in ragas_scopes} == {
        "faithfulness",
        "answer_correctness",
    }
    assert all(
        scope.status == "completed"
        and len(scope.targets) == 4
        and all(target.is_official for target in scope.targets)
        for scope in ragas_scopes
    )
    assert all(
        [event.phase for event in events_by_scope[scope.scope_id]] == ["ragas_scoring"]
        and events_by_scope[scope.scope_id][0].run_id is None
        for scope in ragas_scopes
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
        "operational_usd": pytest.approx(0.00020),
        "pricing_status": "complete",
        "priced_call_count": 10,
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
    assert body["evaluation_overhead"]["tokens"]["accounting_status"] == "complete"
    assert body["evaluation_overhead"]["tokens"]["phase_attribution_status"] == (
        "complete"
    )
    assert body["evaluation_overhead"]["batch_count"] == 2
    assert body["evaluation_overhead"]["retry_count"] == 0
    assert body["evaluation_overhead"]["cost_usd"] == pytest.approx(0.00004)
    assert body["evaluation_overhead"]["pricing_status"] == "complete"
    assert body["evaluation_overhead"]["evaluator_models"] == ["test-model"]
    assert body["quality_status"] == "partial"
    assert body["quality"]["faithfulness"]["value"] == pytest.approx(0.8)
    assert body["quality"]["faithfulness"]["status"] == "complete"
    assert body["quality"]["faithfulness"]["valid_samples"] == 4
    assert body["quality"]["faithfulness"]["missing_samples"] == 0
    assert body["quality"]["answer_correctness"]["value"] == pytest.approx(0.8)
    assert body["quality"]["answer_correctness"]["status"] == "complete"
    assert body["quality"]["answer_correctness"]["valid_samples"] == 4
    assert body["quality"]["answer_correctness"]["missing_samples"] == 0
    assert body["quality"]["answer_relevancy"]["status"] == "not_requested"
    assert body["quality"]["answer_relevancy"]["value"] is None
    assert body["quality"]["answer_relevancy"]["valid_samples"] == 0
    assert body["quality"]["answer_relevancy"]["missing_samples"] == 4
    assert {mode["mode"] for mode in body["modes"]} == set(EXPECTED_PHASES)
    modes = {mode["mode"]: mode for mode in body["modes"]}
    assert modes["naive"]["execution_cost"]["benchmark_usd"] == pytest.approx(0.00002)
    assert modes["naive"]["execution_cost"]["operational_usd"] == pytest.approx(0.00004)
    assert all(
        mode["execution_cost"]["benchmark_usd"]
        == mode["execution_cost"]["operational_usd"]
        for name, mode in modes.items()
        if name != "naive"
    )
    assert all(
        mode["quality"][metric]["status"] == "complete"
        and mode["quality"][metric]["valid_samples"] == 1
        for mode in body["modes"]
        for metric in ("faithfulness", "answer_correctness")
    )
    assert all(mode["comparable"] is False for mode in body["modes"])
    assert all(
        "incomplete_quality" in mode["not_comparable_reasons"] for mode in body["modes"]
    )
