"""Durable execution adapter for immutable dataset work snapshots."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from core.llm_usage_context import llm_accounting_scope
from evaluation.accounting_runtime import (
    EvaluationAccountingSink,
    start_execution_scope,
)
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.campaign_engine import (
    CampaignUnit,
    CampaignRunner,
    ExecutedCampaignUnit,
    _build_derived_metrics,
    _build_question_snapshot,
    _build_system_version_snapshot,
    _duration_ms,
    _enrich_agent_trace_payload,
    _final_answer_hash,
    _record_unit_llm_usage,
    _record_unit_research_observability,
    _record_unit_root_span,
)
from evaluation.campaign_schemas import CampaignResult, CampaignResultStatus
from evaluation.db import (
    AgentTraceRepository,
    CampaignRepository,
    CampaignResultRepository,
)
from evaluation.error_policy import classify_evaluation_error
from evaluation.job_schemas import ClaimedEvaluationWork, ExecutionAttemptOutput
from evaluation.job_store import EvaluationJobStore
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.rag_modes import BenchmarkExecutionResult, run_campaign_case
from evaluation.retrieval_profiles import evaluation_failure_execution_profile
from evaluation.schemas import TestCase
from evaluation.trace_schemas import (
    EvaluationClaim,
    EvaluationEvidencePacket,
    EvaluationSlotResolution,
)
from data_base.agentic_v9.schemas import EvidencePacket, FinalClaim, SlotResolution

logger = logging.getLogger(__name__)


class DatasetExecutionWorker:
    """Execute a claimed dataset unit exclusively from its stored snapshot."""

    def __init__(
        self,
        *,
        store: EvaluationJobStore | None = None,
        runner: CampaignRunner = run_campaign_case,
        campaign_repository: CampaignRepository | None = None,
        result_repository: CampaignResultRepository | None = None,
        trace_repository: AgentTraceRepository | None = None,
        ragas_evaluator: Any | None = None,
        notify: Any | None = None,
        accounting_store: EvaluationAccountingStore | None = None,
        price_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self._store = store or EvaluationJobStore()
        self._runner = runner
        self._campaign_repository = campaign_repository or CampaignRepository()
        self._result_repository = result_repository or CampaignResultRepository()
        self._trace_repository = trace_repository or AgentTraceRepository()
        self._observability_repository = EvaluationObservabilityRepository()
        self._ragas_evaluator = ragas_evaluator
        self._notify = notify
        self._accounting_store = accounting_store or EvaluationAccountingStore()
        self._accounting_sink = EvaluationAccountingSink(
            store=self._accounting_store, price_snapshot=price_snapshot
        )

    async def execute(self, claim: ClaimedEvaluationWork) -> None:
        """Execute one claimed unit, preserving attempts before official promotion."""
        payload: BenchmarkExecutionResult | None = None
        try:
            unit, user_id, campaign_id, model_config = self._snapshot_inputs(claim)
            started_at = datetime.now(timezone.utc)
            started_perf = time.perf_counter()
            run_id = str(uuid4())
            scope = await start_execution_scope(
                store=self._accounting_store,
                sink=self._accounting_sink,
                campaign_id=campaign_id,
                run_id=run_id,
                job_id=claim.job_id,
                work_item_id=claim.work_item_id,
                attempt_id=claim.attempt_id,
                mode=unit.mode,
            )
            try:
                with llm_accounting_scope(scope.context):
                    payload = await self._runner(
                        test_case=unit.test_case,
                        user_id=user_id,
                        mode=unit.mode,
                        model_config=model_config,
                        run_number=unit.repeat_number,
                        ablation_flags=unit.ablation_flags,
                        budget=unit.budget,
                        agentic_execution_version=unit.agentic_execution_version,
                        shadow_evaluation_policy=unit.shadow_evaluation_policy,
                    )
                completed_at = datetime.now(timezone.utc)
                total_latency_ms = max((time.perf_counter() - started_perf) * 1000, 0)
                if total_latency_ms <= 0:
                    total_latency_ms = _duration_ms(started_at, completed_at)
                if payload.error_message:
                    raise RuntimeError(payload.error_message)
                token_summary = await self._accounting_store.summarize_scope_tokens(
                    scope.scope_id
                )
                payload.token_usage = token_summary.as_legacy_usage(
                    accounting_schema_version="2"
                )
                payload.token_usage["token_accounting_status"] = (
                    "partial"
                    if scope.context.persistence_error_count
                    else token_summary.reconciliation_status
                )
                execution = ExecutedCampaignUnit(
                    unit=unit,
                    payload=payload,
                    run_id=run_id,
                    request_id=str(uuid4()),
                    started_at=started_at,
                    completed_at=completed_at,
                    total_latency_ms=total_latency_ms,
                    model_config=model_config,
                )
                result = self._successful_result(
                    campaign_id=campaign_id,
                    execution=execution,
                    total_tokens=token_summary.total_tokens,
                )
                promoted = await self._store.complete_execution_attempt(
                    claim,
                    ExecutionAttemptOutput(result=result),
                    accounting_scope_id=scope.scope_id,
                )
                if unit.agentic_execution_version == "v9":
                    await self._materialize_v9_attempt(
                        claim=claim,
                        run_id=promoted.id,
                        campaign_id=campaign_id,
                        condition_id=unit.condition_id or "",
                        payload=payload,
                    )
            except asyncio.CancelledError:
                await self._accounting_store.finalize_scope(scope.scope_id, "cancelled")
                raise
            except Exception:
                await self._accounting_store.finalize_scope(scope.scope_id, "failed")
                raise
        except Exception as exc:  # noqa: BLE001
            if await self._claim_was_cancelled(claim):
                return
            decision = classify_evaluation_error(exc)
            if self._runner is run_campaign_case or isinstance(
                payload, BenchmarkExecutionResult
            ):
                try:
                    await self._persist_failed_result(claim, exc, payload=payload)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Failed to persist execution failure projection",
                        extra={"work_item_id": claim.work_item_id},
                        exc_info=True,
                    )
            try:
                await self._store.fail_attempt(claim, decision, next_retry_at=None)
            except ValueError:
                if await self._claim_was_cancelled(claim):
                    return
                raise
            await self._derive_campaign_state(claim)
            return

        try:
            await self._record_observability(
                user_id=user_id,
                campaign_id=campaign_id,
                promoted_result_id=promoted.id,
                execution=execution,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Campaign execution trace persistence failed after durable promotion",
                extra={"campaign_id": campaign_id, "result_id": promoted.id},
                exc_info=True,
            )
        await self._derive_campaign_state(claim)

    async def _materialize_v9_attempt(
        self,
        *,
        claim: ClaimedEvaluationWork,
        run_id: str,
        campaign_id: str,
        condition_id: str,
        payload: BenchmarkExecutionResult,
    ) -> None:
        """Persist the terminal core state under the promoted attempt identity."""
        trace = payload.agent_trace or {}
        v9 = trace.get("agentic_v9") if isinstance(trace, dict) else None
        if not isinstance(v9, dict):
            raise ValueError("v9 campaign payload is missing its typed execution trace")
        schema_version = str(v9.get("schema_version") or "1")
        evidence_rows: list[EvaluationEvidencePacket] = []
        for raw_packet in v9.get("evidence_packets", []):
            packet = EvidencePacket.model_validate(raw_packet)
            evidence_rows.append(
                EvaluationEvidencePacket(
                    attempt_id=claim.attempt_id,
                    run_id=run_id,
                    campaign_id=campaign_id,
                    condition_id=condition_id,
                    schema_version=schema_version,
                    evidence_id=packet.evidence_id,
                    packet=packet.model_dump(mode="json"),
                )
            )
        resolution_rows: list[EvaluationSlotResolution] = []
        for raw_resolution in v9.get("slot_resolutions", []):
            resolution = SlotResolution.model_validate(raw_resolution)
            resolution_rows.append(
                EvaluationSlotResolution(
                    attempt_id=claim.attempt_id,
                    run_id=run_id,
                    campaign_id=campaign_id,
                    condition_id=condition_id,
                    schema_version=schema_version,
                    slot_id=resolution.slot_id,
                    resolution_stage=resolution.resolution_stage or "sufficiency_gate",
                    resolution=resolution.model_dump(mode="json"),
                )
            )
        claims: list[EvaluationClaim] = []
        for raw_claim in v9.get("final_claims", []):
            final_claim = FinalClaim.model_validate(raw_claim)
            claims.append(
                EvaluationClaim(
                    claim_id=final_claim.claim_id,
                    attempt_id=claim.attempt_id,
                    run_id=run_id,
                    campaign_id=campaign_id,
                    condition_id=condition_id,
                    schema_version=schema_version,
                    claim_text=final_claim.statement,
                    claim_type=final_claim.support_type,
                    support_status=(
                        "supported" if final_claim.evidence_ids else "unsupported"
                    ),
                    evidence=[
                        {"evidence_id": evidence_id}
                        for evidence_id in final_claim.evidence_ids
                    ],
                    unsupported_reason=final_claim.qualified_reason,
                )
            )
        trace_payload = dict(v9)
        trace_payload.pop("evidence_packets", None)
        trace_payload.pop("slot_resolutions", None)
        await self._observability_repository.materialize_v9_attempt(
            attempt_id=claim.attempt_id,
            run_id=run_id,
            campaign_id=campaign_id,
            condition_id=condition_id,
            schema_version=schema_version,
            trace_payload=trace_payload,
            evidence_packets=evidence_rows,
            slot_resolutions=resolution_rows,
            claims=claims,
        )

    async def _claim_was_cancelled(self, claim: ClaimedEvaluationWork) -> bool:
        return await self._store.get_job_item_status(claim.job_item_id) == "cancelled"

    async def _persist_failed_result(
        self,
        claim: ClaimedEvaluationWork,
        exc: Exception,
        *,
        payload: BenchmarkExecutionResult | None = None,
    ) -> None:
        """Keep a visible failed result while leaving the attempt non-successful.

        The durable attempt remains the source of truth for retry and promotion;
        this projection only makes an execution failure visible to the existing
        campaign results API.  A later successful attempt atomically replaces
        the same unit projection in ``complete_execution_attempt``.
        """
        unit, user_id, campaign_id, model_config = self._snapshot_inputs(claim)
        now = datetime.now(timezone.utc)
        test_case = unit.test_case
        await self._result_repository.create(
            result_id=claim.attempt_id,
            user_id=user_id,
            campaign_id=campaign_id,
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            ground_truth_short=test_case.ground_truth_short,
            key_points=list(test_case.key_points),
            ragas_focus=list(test_case.ragas_focus),
            mode=unit.mode,
            execution_profile=evaluation_failure_execution_profile(
                unit.mode,
                payload if payload is not None else exc,
            ),
            context_policy_version=None,
            run_number=unit.run_number,
            condition_id=unit.condition_id,
            answer=f"ERROR: {exc}",
            contexts=[],
            source_doc_ids=[],
            expected_sources=list(test_case.source_docs),
            latency_ms=0,
            token_usage={},
            category=test_case.category,
            difficulty=test_case.difficulty,
            status=CampaignResultStatus.FAILED,
            error_message=str(exc),
            question_version=test_case.question_version,
            request_id=None,
            started_at=now.isoformat(),
            completed_at=now.isoformat(),
            total_latency_ms=0,
            total_tokens=0,
            question_snapshot=_build_question_snapshot(test_case),
            model_config_snapshot=model_config,
            system_version_snapshot={
                "agentic_execution_version": unit.agentic_execution_version,
            },
            derived_metrics={
                "agentic_execution_version": unit.agentic_execution_version,
                "response_status": "failed",
            },
            final_answer_hash=_final_answer_hash(f"ERROR: {exc}"),
            source_attempt_id=claim.attempt_id,
        )

    def _snapshot_inputs(
        self, claim: ClaimedEvaluationWork
    ) -> tuple[CampaignUnit, str, str, dict[str, Any]]:
        snapshot = claim.input_snapshot
        test_case = TestCase.model_validate(snapshot["test_case"])
        user_id = str(snapshot["user_id"])
        campaign_id = str(snapshot["campaign_id"])
        model_config = dict(snapshot["model_config"])
        return (
            CampaignUnit(
                test_case=test_case,
                mode=str(snapshot["mode"]),
                run_number=int(snapshot["run_number"]),
                repeat_number=int(
                    snapshot.get("repeat_number", snapshot["run_number"])
                ),
                condition_id=(
                    str(snapshot["condition_id"])
                    if snapshot.get("condition_id")
                    else None
                ),
                condition_label=(
                    str(snapshot["condition_label"])
                    if snapshot.get("condition_label")
                    else None
                ),
                ablation_flags=(
                    dict(snapshot["ablation_flags"])
                    if snapshot.get("ablation_flags")
                    else None
                ),
                budget=dict(snapshot["budget"]) if snapshot.get("budget") else None,
                agentic_execution_version=(
                    str(snapshot.get("agentic_execution_version", "v8"))
                    if snapshot.get("agentic_execution_version") in {"v8", "v9"}
                    else "v8"
                ),
                shadow_evaluation_policy=(
                    str(snapshot["shadow_evaluation_policy"])
                    if snapshot.get("shadow_evaluation_policy") in {"operational", "research"}
                    else None
                ),
            ),
            user_id,
            campaign_id,
            model_config,
        )

    def _successful_result(
        self,
        *,
        campaign_id: str,
        execution: ExecutedCampaignUnit,
        total_tokens: int | None = None,
    ) -> CampaignResult:
        unit = execution.unit
        payload = execution.payload
        assert isinstance(payload, BenchmarkExecutionResult)
        return CampaignResult(
            id=execution.run_id,
            campaign_id=campaign_id,
            question_id=payload.question_id,
            question=payload.question,
            ground_truth=payload.ground_truth,
            ground_truth_short=payload.ground_truth_short,
            key_points=list(payload.key_points),
            ragas_focus=list(payload.ragas_focus),
            mode=payload.mode,
            execution_profile=payload.execution_profile,
            context_policy_version=payload.context_policy_version,
            run_number=unit.run_number,
            condition_id=unit.condition_id,
            answer=payload.answer,
            contexts=payload.contexts,
            source_doc_ids=payload.source_doc_ids,
            expected_sources=payload.expected_sources,
            latency_ms=payload.latency_ms,
            token_usage=payload.token_usage,
            category=payload.category,
            difficulty=payload.difficulty,
            question_version=unit.test_case.question_version,
            request_id=execution.request_id,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            total_latency_ms=execution.total_latency_ms,
            total_tokens=total_tokens,
            question_snapshot=_build_question_snapshot(unit.test_case),
            model_config_snapshot=execution.model_config,
            system_version_snapshot=_build_system_version_snapshot(
                unit=unit, payload=payload
            ),
            derived_metrics=_build_derived_metrics(unit=unit, payload=payload),
            final_answer_hash=_final_answer_hash(payload.answer),
            status=CampaignResultStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
        )

    async def _derive_campaign_state(self, claim: ClaimedEvaluationWork) -> None:
        snapshot = claim.input_snapshot
        try:
            job = await self._store.get_job(
                user_id=str(snapshot["user_id"]), job_id=claim.job_id
            )
        except Exception:  # noqa: BLE001
            job = None
        skip_ragas = bool(
            job is not None and job.config_snapshot.get("skip_ragas") is True
        )
        configured_metrics = getattr(self._ragas_evaluator, "enabled_metrics", None)
        if configured_metrics is None and callable(
            getattr(self._ragas_evaluator, "evaluate_campaign", None)
        ):
            configured_metrics = ("legacy_campaign",)
        enabled_metrics = list(configured_metrics or [])
        campaign = await self._campaign_repository.derive_execution_state(
            user_id=str(snapshot["user_id"]),
            campaign_id=str(snapshot["campaign_id"]),
            defer_completion=bool(enabled_metrics) and not skip_ragas,
            completion_phase="evaluation"
            if not enabled_metrics and not skip_ragas
            else "execution",
        )
        if self._ragas_evaluator is None:
            return
        # An execution-only rerun intentionally stops after promoting the
        # dataset result.  Combined and initial jobs continue to materialize
        # downstream metric work from the successful official projection.
        if skip_ragas:
            return
        if campaign.status.value not in {"completed", "completed_with_errors"}:
            return
        selected_result_ids: list[str] | None = None
        if job is not None:
            raw_question_ids = job.config_snapshot.get("downstream_question_ids")
            if isinstance(raw_question_ids, list) and raw_question_ids:
                results = await self._result_repository.list_for_campaign(
                    user_id=str(snapshot["user_id"]),
                    campaign_id=str(snapshot["campaign_id"]),
                )
                question_ids = {str(value) for value in raw_question_ids}
                selected_result_ids = [
                    result.id
                    for result in results
                    if result.question_id in question_ids
                ]
            raw_metrics = job.config_snapshot.get("metric_names")
            if isinstance(raw_metrics, list) and raw_metrics:
                enabled_metrics = [str(value) for value in raw_metrics]
        ragas_config = getattr(campaign, "config", None)
        ragas_batch_size = getattr(ragas_config, "ragas_batch_size", None)
        ragas_parallel_batches = getattr(ragas_config, "ragas_parallel_batches", None)
        if getattr(ragas_config, "shadow_evaluation_policy", None) == "operational":
            # Product shadow is diagnostic work: it never feeds authoritative
            # RAGAS comparisons or replaces the v8 answer projection.
            results = await self._result_repository.list_for_campaign(
                user_id=str(snapshot["user_id"]),
                campaign_id=str(snapshot["campaign_id"]),
            )
            selected_result_ids = [
                result.id
                for result in results
                if result.mode != "agentic-v9-shadow"
            ]
        created = await self._store.ensure_ragas_work(
            user_id=str(snapshot["user_id"]),
            campaign_id=str(snapshot["campaign_id"]),
            evaluator_model=str(getattr(self._ragas_evaluator, "evaluator_model", "")),
            evaluator_config={},
            enabled_metrics=enabled_metrics,
            selected_result_ids=selected_result_ids,
            ragas_batch_size=ragas_batch_size,
            ragas_parallel_batches=ragas_parallel_batches,
        )
        if created:
            if self._notify is not None:
                self._notify()

    async def _record_observability(
        self,
        *,
        user_id: str,
        campaign_id: str,
        promoted_result_id: str,
        execution: ExecutedCampaignUnit,
    ) -> None:
        span_id = await _record_unit_root_span(
            run_id=promoted_result_id,
            campaign_id=campaign_id,
            request_id=execution.request_id,
            unit=execution.unit,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_ms=execution.total_latency_ms,
            failed=False,
        )
        await _record_unit_llm_usage(
            run_id=promoted_result_id,
            campaign_id=campaign_id,
            user_id=user_id,
            request_id=execution.request_id,
            span_id=span_id,
            execution=execution,
        )
        await _record_unit_research_observability(
            run_id=promoted_result_id,
            campaign_id=campaign_id,
            user_id=user_id,
            request_id=execution.request_id,
            root_span_id=span_id,
            execution=execution,
        )
        payload = execution.payload
        if isinstance(payload, BenchmarkExecutionResult) and payload.agent_trace:
            await self._trace_repository.replace_for_result(
                user_id=user_id,
                campaign_id=campaign_id,
                campaign_result_id=promoted_result_id,
                trace_payload=_enrich_agent_trace_payload(
                    trace_payload=payload.agent_trace,
                    created_id=promoted_result_id,
                    unit=execution.unit,
                    payload=payload,
                ),
            )
