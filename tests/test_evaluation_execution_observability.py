"""Durable dataset execution observability coverage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from langchain_core.documents import Document
import pytest
import pytest_asyncio

import evaluation.db as evaluation_db
from core.llm_usage_context import emit_direct_usage
from core.providers import configure_providers
from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.budgeted_llm import invoke_budgeted_llm
from data_base.agentic_v9.schemas import QueryContract, RequiredSlot, ResolvedSourceScope
from evaluation.agentic_v9_admission import V9AdmissionContract
from evaluation.agentic_v9_campaign_runtime import AgenticV9CampaignRuntime
from evaluation.analytics import reconcile_official_tokens
from evaluation.campaign_schemas import CampaignLifecycleStatus, CampaignResultStatus
from evaluation.execution_worker import DatasetExecutionWorker
from evaluation.job_schemas import (
    ClaimedEvaluationWork,
    EvaluationWorkType,
    WorkItemSpec,
)
from evaluation.job_store import EvaluationJobStore
from evaluation.observability import current_llm_call_observer
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.rag_modes import BenchmarkExecutionResult


@pytest_asyncio.fixture
async def store(monkeypatch: pytest.MonkeyPatch) -> EvaluationJobStore:
    database_path = (
        Path(os.environ["EVALUATION_TEST_TMPDIR"])
        / f"dataset-observability-{uuid4().hex}"
        / "worker.db"
    )
    database_path.parent.mkdir(parents=True)
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", database_path)
    await evaluation_db.force_init_db()
    async with evaluation_db.connect_db() as connection:
        now = "2026-08-10T00:00:00+00:00"
        config = json.dumps(
            {
                "test_case_ids": ["Q1"],
                "modes": ["agentic"],
                "model_config": {
                    "id": "cfg-1",
                    "name": "test",
                    "model_name": "test-model",
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "max_input_tokens": 8192,
                    "max_output_tokens": 2048,
                    "thinking_mode": False,
                    "thinking_budget": 0,
                },
            }
        )
        await connection.execute(
            """
            INSERT INTO campaigns
              (id, user_id, name, status, config_json, created_at, updated_at)
            VALUES ('cmp-1', 'user-a', NULL, 'pending', ?, ?, ?)
            """,
            (config, now, now),
        )
        await connection.commit()
    try:
        yield EvaluationJobStore()
    finally:
        for path in (
            database_path,
            database_path.with_suffix(".db-shm"),
            database_path.with_suffix(".db-wal"),
        ):
            path.unlink(missing_ok=True)
        rmtree(database_path.parent, ignore_errors=True)


async def _claim_execution(
    store: EvaluationJobStore,
    *,
    mode: str = "agentic",
    agentic_execution_version: str = "v9",
    source_docs: list[str] | None = None,
    model_config: dict[str, object] | None = None,
) -> ClaimedEvaluationWork:
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[
            WorkItemSpec(
                work_type=EvaluationWorkType.DATASET_EXECUTION,
                logical_key=f"execution:Q1:{mode}:1:none",
                input_snapshot={
                    "user_id": "user-a",
                    "campaign_id": "cmp-1",
                    "test_case": {
                        "id": "Q1",
                        "question": "What is the answer?",
                        "ground_truth": "42",
                        "source_docs": source_docs or [],
                        "requires_multi_doc_reasoning": False,
                    },
                    "mode": mode,
                    "run_number": 1,
                    "repeat_number": 1,
                    "condition_id": None,
                    "condition_label": None,
                    "ablation_flags": None,
                    "budget": None,
                    "model_config": model_config or {},
                    "agentic_execution_version": agentic_execution_version,
                    "shadow_evaluation_policy": None,
                },
            )
        ],
    )
    claims = await store.claim_ready_items(
        limit=1,
        now=datetime.now(timezone.utc),
    )
    assert len(claims) == 1
    return claims[0]


@pytest.mark.asyncio
async def test_completed_run_persists_snapshots_and_root_observability_span(
    store: EvaluationJobStore,
) -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:  # noqa: ANN003
        await asyncio.sleep(0.01)
        await emit_direct_usage(
            purpose="campaign_generation",
            provider="fake",
            model_name="gemini-2.5-flash",
            raw_usage={"total_tokens": 77, "input_tokens": 55, "output_tokens": 22},
        )
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            ground_truth_short=test_case.ground_truth_short,
            key_points=list(test_case.key_points),
            ragas_focus=list(test_case.ragas_focus),
            mode=kwargs["mode"],
            answer="Snapshot answer",
            contexts=["ctx-1", "ctx-2"],
            source_doc_ids=["doc-a"],
            expected_sources=list(test_case.source_docs),
            latency_ms=18,
            token_usage={"total_tokens": 77, "input_tokens": 55, "output_tokens": 22},
            category=test_case.category,
            difficulty=test_case.difficulty,
            execution_profile="snapshot-profile",
            context_policy_version="ctx-policy-v1",
        )

    model_config = {
        "id": "cfg-1",
        "name": "Balanced",
        "model_name": "gemini-2.5-flash",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_input_tokens": 8192,
        "max_output_tokens": 2048,
        "thinking_mode": False,
        "thinking_budget": 8192,
    }
    claim = await _claim_execution(
        store,
        mode="naive",
        source_docs=["doc-a", "doc-b"],
        model_config=model_config,
    )
    worker = DatasetExecutionWorker(store=store, runner=runner)
    await worker.execute(claim)

    campaign_repo = evaluation_db.CampaignRepository()
    result_repo = evaluation_db.CampaignResultRepository()
    observability_repo = EvaluationObservabilityRepository()
    latest = await campaign_repo.get(user_id="user-a", campaign_id="cmp-1")
    assert latest.status == CampaignLifecycleStatus.COMPLETED
    assert latest.phase == "evaluation"

    results = await result_repo.list_for_campaign(
        user_id="user-a", campaign_id="cmp-1"
    )
    assert len(results) == 1
    result = results[0]
    assert result.status == CampaignResultStatus.COMPLETED
    assert result.question_version is None
    assert result.request_id
    assert result.started_at is not None
    assert result.completed_at is not None
    assert result.completed_at >= result.started_at
    assert result.total_latency_ms is not None
    assert result.total_latency_ms >= 0
    assert result.total_tokens == 77
    assert result.question_snapshot == {
        "id": "Q1",
        "question": "What is the answer?",
        "ground_truth": "42",
        "ground_truth_short": None,
        "key_points": [],
        "ragas_focus": [],
        "category": None,
        "difficulty": None,
        "question_version": None,
        "required_modalities": [],
        "atomic_facts": [],
        "expected_evidence": [],
        "source_docs": ["doc-a", "doc-b"],
    }
    assert result.model_config_snapshot == model_config
    assert result.system_version_snapshot["execution_profile"] == "snapshot-profile"
    assert result.system_version_snapshot["context_policy_version"] == "ctx-policy-v1"
    assert isinstance(result.derived_metrics, dict)
    assert result.final_answer_hash

    trace_events = await observability_repo.list_trace_events_for_run(result.id)
    assert [event.status for event in trace_events] == ["running", "success"]
    assert all(event.stage_name == "campaign_unit_execution" for event in trace_events)
    assert all(event.parent_event_id is None for event in trace_events)
    assert all(event.parent_span_id is None for event in trace_events)
    assert trace_events[0].payload["request_id"] == result.request_id
    assert trace_events[1].duration_ms == pytest.approx(
        result.total_latency_ms, rel=0.2, abs=20
    )

    llm_calls = await observability_repo.list_llm_calls_for_run(result.id)
    assert len(llm_calls) == 1
    assert llm_calls[0].purpose == "campaign_generation"
    assert llm_calls[0].model_name == "gemini-2.5-flash"
    assert llm_calls[0].prompt_tokens == 55
    assert llm_calls[0].completion_tokens == 22
    assert llm_calls[0].total_tokens == 77
    assert llm_calls[0].span_id == trace_events[0].span_id


@pytest.mark.asyncio
async def test_llm_observer_write_failure_preserves_answer_and_marks_run_partial(
    store: EvaluationJobStore,
) -> None:
    configure_providers(use_fake=True)

    class Provider:
        async def ainvoke(self, messages: object) -> object:
            return {
                "content": "observed answer",
                "usage_metadata": {
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
            }

    async def runner(**kwargs) -> BenchmarkExecutionResult:  # noqa: ANN003
        test_case = kwargs["test_case"]
        observer = current_llm_call_observer()
        assert observer is not None
        response = await invoke_budgeted_llm(
            controller=RunBudgetController(
                max_llm_calls=1,
                runtime_token_budget=200,
                setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
                final_input_tokens=100,
            ),
            provider=Provider(),
            observer=observer,
            model_name="gemini-2.5-flash",
            phase="final_answer",
            purpose="synthesizer",
            messages=[{"role": "user", "content": "answer"}],
            estimated_input_tokens=100,
        )
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer=response["content"],
            token_usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        )

    claim = await _claim_execution(store, mode="naive")
    worker = DatasetExecutionWorker(store=store, runner=runner)
    with patch.object(
        EvaluationObservabilityRepository,
        "record_llm_call",
        new=AsyncMock(side_effect=OSError("storage unavailable")),
    ) as record_llm_call:
        await worker.execute(claim)

    result = (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
    )[0]
    assert result.answer == "observed answer"
    assert result.status == CampaignResultStatus.COMPLETED
    assert result.derived_metrics["observability_status"] == "partial"
    assert set(result.derived_metrics["observability_partial_reasons"]) == {
        "observability_write_failed",
        "llm_call_observer_failed",
    }
    attempted_call = record_llm_call.await_args_list[0].args[0]
    assert attempted_call.provider == "fake"


@pytest.mark.asyncio
async def test_v9_campaign_persists_default_visual_and_final_provider_attempts(
    store: EvaluationJobStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Provider:
        async def ainvoke(self, messages: object) -> object:
            user_message = messages[1]
            content = user_message["content"]
            if isinstance(content, list):
                binding = json.loads(content[0]["text"])
                return SimpleNamespace(
                    content={
                        "schema_version": "1",
                        "evidence_id": "visual-evidence-1",
                        "task_id": binding["task_id"],
                        "round_id": binding["round_id"],
                        "query_id": binding["query_id"],
                        "slot_ids": binding["target_slot_ids"],
                        "statement": "The table reports 0.91.",
                        "support_type": "direct",
                        "source": binding["source"],
                        "scope": {},
                        "locator": binding["locator"],
                        "validation_status": "deterministic_valid",
                    },
                    usage_metadata={
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                )
            return SimpleNamespace(
                content="The table reports 0.91.",
                usage_metadata={
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                },
            )

    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="exact_structured",
        intent="table value",
        required_slots=[RequiredSlot(slot_id="S1", description="table value")],
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs) -> V9AdmissionContract:  # noqa: ANN003
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )

    async def runner(**kwargs) -> BenchmarkExecutionResult:  # noqa: ANN003
        runtime_result = await AgenticV9CampaignRuntime(
            retrieve_documents=AsyncMock(
                return_value=[
                    Document(
                        page_content="The table reports 0.91.",
                        metadata={
                            "doc_id": "doc-1",
                            "chunk_id": "chunk-1",
                            "page_number": 0,
                            "page_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M/wHwAF/gL+f4eP0QAAAABJRU5ErkJggg==",
                            "page_encoded_bytes": 70,
                            "page_width": 1,
                            "page_height": 1,
                            "table_id": "table-1",
                        },
                    )
                ]
            ),
            provider_factory=lambda _purpose: Provider(),
            document_reference_resolver=lambda _user_id, references: asyncio.sleep(
                0, result={reference: reference for reference in references}
            ),
        ).execute(
            question=kwargs["test_case"].question,
            user_id=kwargs["user_id"],
            authorized_doc_ids=["doc-1"],
            setup_snapshot={
                "max_input_tokens": 4096,
                "max_output_tokens": 256,
                "thinking_mode": False,
                "provider": "gemini",
                "model_name": "gemini-2.5-flash",
            },
            trace_id="visual-phase-trace",
        )
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer=runtime_result.answer,
            contexts=[document.page_content for document in runtime_result.documents],
            source_doc_ids=list(runtime_result.source_doc_ids),
            token_usage=dict(runtime_result.usage),
            execution_profile="agentic-v9-eval",
            agentic_execution_version="v9",
            agent_trace=runtime_result.agent_trace,
        )

    claim = await _claim_execution(
        store,
        mode="agentic",
        agentic_execution_version="v9",
        source_docs=["doc-1"],
        model_config={
            "provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "max_input_tokens": 4096,
            "max_output_tokens": 256,
            "thinking_mode": False,
        },
    )
    worker = DatasetExecutionWorker(store=store, runner=runner)
    await worker.execute(claim)

    result = (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
    )[0]
    calls = await EvaluationObservabilityRepository().list_llm_calls_for_run(result.id)

    assert [(call.phase, call.provider, call.model_name) for call in calls] == [
        ("visual_extract", "gemini", "gemini-2.5-flash"),
        ("final_answer", "gemini", "gemini-2.5-flash"),
    ]
    reconciliation = reconcile_official_tokens(
        runtime_total_tokens=result.total_tokens,
        calls=calls,
        observability_partial_reasons=result.derived_metrics[
            "observability_partial_reasons"
        ],
    )
    assert result.total_tokens == 9
    assert reconciliation.status == "complete"
    assert reconciliation.provider_total_tokens == 9
    assert reconciliation.by_phase == {"visual_extract": 3, "final_answer": 6}


@pytest.mark.asyncio
async def test_agentic_trace_persists_routing_and_tool_observability(
    store: EvaluationJobStore,
) -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:  # noqa: ANN003
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="agentic answer",
            contexts=["ctx"],
            source_doc_ids=["doc-a"],
            expected_sources=["doc-a"],
            latency_ms=9,
            token_usage={"total_tokens": 12},
            category=test_case.category,
            difficulty=test_case.difficulty,
            execution_profile="agentic-eval",
            agent_trace={
                "classifier_decision": {
                    "router_version": "semantic-v1",
                    "router_type": "semantic_gate",
                    "selected_strategy_tier": "tier_3",
                    "complexity_score": 4,
                    "modality_score": 2,
                    "multi_doc_score": 3,
                    "conflict_score": 1,
                    "exact_value_score": 0,
                    "hallucination_risk_score": 2,
                    "retrieval_uncertainty_score": 1,
                    "routing_reason": "needs visual verification",
                    "routing_features": {"has_figure": True},
                },
                "strategy_tier": "tier_3",
                "route_profile": "visual_verify",
                "steps": [
                    {
                        "step_id": "s1",
                        "phase": "execution",
                        "step_type": "visual",
                        "title": "Verify figure",
                        "status": "completed",
                        "tool_calls": [
                            {
                                "tool_name": "visual_verifier",
                                "tool_type": "visual",
                                "action": "VERIFY_IMAGE",
                                "status": "completed",
                                "subtask_id": "1",
                                "input_summary": {"image": "figure-1"},
                                "output_summary": {"finding": "supported"},
                            }
                        ],
                    }
                ],
            },
        )

    claim = await _claim_execution(store, mode="agentic")
    worker = DatasetExecutionWorker(store=store, runner=runner)
    await worker.execute(claim)

    result = (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
    )[0]
    observability_repo = EvaluationObservabilityRepository()
    trace_events = await observability_repo.list_trace_events_for_run(result.id)
    routing_events = [event for event in trace_events if event.stage_type == "routing"]
    assert [event.status for event in routing_events] == ["running", "success"]

    decisions = await observability_repo.list_routing_decisions_for_run(result.id)
    assert len(decisions) == 1
    assert decisions[0].selected_mode == "agentic"
    assert decisions[0].payload["router_version"] == "semantic-v1"
    assert decisions[0].payload["selected_strategy_tier"] == "tier_3"
    assert decisions[0].payload["routing_features"] == {"has_figure": True}
    assert decisions[0].payload["actual_router_execution_enabled"] is False

    tool_calls = await observability_repo.list_tool_calls_for_run(result.id)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "visual_verifier"
    assert tool_calls[0].action == "VERIFY_IMAGE"
    assert tool_calls[0].payload["tool_type"] == "visual"
    assert tool_calls[0].payload["subtask_id"] == "1"
    assert tool_calls[0].payload["input_summary"] == {"image": "figure-1"}


@pytest.mark.asyncio
async def test_v9_actual_route_is_persisted_separately_from_retrospective_route(
    store: EvaluationJobStore,
) -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:  # noqa: ANN003
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="bounded answer",
            contexts=["ctx"],
            source_doc_ids=["doc-a"],
            expected_sources=["doc-a"],
            latency_ms=9,
            token_usage={"total_tokens": 12},
            category=test_case.category,
            difficulty=test_case.difficulty,
            execution_profile="agentic-v9-eval",
            agent_trace={
                "classifier_decision": {
                    "routing_reason": "retrospective policy view",
                    "confidence": 0.4,
                },
                "agentic_v9": {
                    "query_contract": {
                        "contract_version": "2",
                        "route": "multi_document_exact",
                        "intent": "Resolve exact facts.",
                        "slot_plan_status": "complete",
                        "route_decision": {
                            "selected_route": "multi_document_exact",
                            "decision_source": "deterministic",
                            "matched_rules": ["multiple_named_sources"],
                            "candidate_routes": [
                                "multi_document_exact",
                                "exact_structured",
                            ],
                            "route_reason": "Multiple named sources.",
                            "planner_call_used": False,
                            "fallback_reason": None,
                            "confidence": 1.0,
                        },
                    }
                },
            },
        )

    claim = await _claim_execution(store, mode="agentic")
    worker = DatasetExecutionWorker(store=store, runner=runner)
    await worker.execute(claim)

    result = (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a", campaign_id="cmp-1"
        )
    )[0]
    decisions = await EvaluationObservabilityRepository().list_routing_decisions_for_run(
        result.id
    )

    assert [decision.analysis_type for decision in decisions] == [
        "retrospective",
        "actual",
    ]
    actual = decisions[1]
    assert actual.decision_source == "deterministic"
    assert actual.candidate_routes == [
        "multi_document_exact",
        "exact_structured",
    ]
    assert actual.matched_rules == ["multiple_named_sources"]
    assert actual.reason == "Multiple named sources."
    assert actual.confidence == 1.0
    assert actual.fallback_reason is None
