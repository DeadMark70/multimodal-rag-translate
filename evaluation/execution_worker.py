"""Durable execution adapter for immutable dataset work snapshots."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from evaluation.campaign_engine import (
    CampaignUnit,
    CampaignRunner,
    ExecutedCampaignUnit,
    _build_derived_metrics,
    _build_question_snapshot,
    _build_system_version_snapshot,
    _duration_ms,
    _enrich_agent_trace_payload,
    _extract_total_tokens,
    _final_answer_hash,
    _record_unit_llm_usage,
    _record_unit_research_observability,
    _record_unit_root_span,
)
from evaluation.campaign_schemas import CampaignResult, CampaignResultStatus
from evaluation.db import AgentTraceRepository, CampaignRepository, CampaignResultRepository
from evaluation.error_policy import classify_evaluation_error
from evaluation.job_schemas import ClaimedEvaluationWork, ExecutionAttemptOutput
from evaluation.job_store import EvaluationJobStore
from evaluation.rag_modes import BenchmarkExecutionResult, run_campaign_case
from evaluation.schemas import TestCase

logger = logging.getLogger(__name__)


class DatasetExecutionWorker:
    """Execute a claimed dataset unit exclusively from its stored snapshot."""

    def __init__(
        self,
        *,
        store: EvaluationJobStore | None = None,
        runner: CampaignRunner = run_campaign_case,
        campaign_repository: CampaignRepository | None = None,
        trace_repository: AgentTraceRepository | None = None,
        ragas_evaluator: Any | None = None,
        notify: Any | None = None,
    ) -> None:
        self._store = store or EvaluationJobStore()
        self._runner = runner
        self._campaign_repository = campaign_repository or CampaignRepository()
        self._trace_repository = trace_repository or AgentTraceRepository()
        self._ragas_evaluator = ragas_evaluator
        self._notify = notify

    async def execute(self, claim: ClaimedEvaluationWork) -> None:
        """Execute one claimed unit, preserving attempts before official promotion."""
        try:
            unit, user_id, campaign_id, model_config = self._snapshot_inputs(claim)
            started_at = datetime.now(timezone.utc)
            started_perf = time.perf_counter()
            payload = await self._runner(
                test_case=unit.test_case,
                user_id=user_id,
                mode=unit.mode,
                model_config=model_config,
                run_number=unit.repeat_number,
                ablation_flags=unit.ablation_flags,
                budget=unit.budget,
            )
            completed_at = datetime.now(timezone.utc)
            total_latency_ms = max((time.perf_counter() - started_perf) * 1000, 0)
            if total_latency_ms <= 0:
                total_latency_ms = _duration_ms(started_at, completed_at)
            if payload.error_message:
                raise RuntimeError(payload.error_message)
            execution = ExecutedCampaignUnit(
                unit=unit,
                payload=payload,
                run_id=str(uuid4()),
                request_id=str(uuid4()),
                started_at=started_at,
                completed_at=completed_at,
                total_latency_ms=total_latency_ms,
                model_config=model_config,
            )
            result = self._successful_result(
                campaign_id=campaign_id,
                execution=execution,
            )
            promoted = await self._store.complete_execution_attempt(
                claim, ExecutionAttemptOutput(result=result)
            )
        except Exception as exc:  # noqa: BLE001
            if await self._claim_was_cancelled(claim):
                return
            decision = classify_evaluation_error(exc)
            try:
                await self._store.fail_attempt(claim, decision, next_retry_at=None)
            except ValueError:
                if await self._claim_was_cancelled(claim):
                    return
                raise
            await self._derive_campaign_state(claim)
            return

        await self._derive_campaign_state(claim)
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

    async def _claim_was_cancelled(self, claim: ClaimedEvaluationWork) -> bool:
        return await self._store.get_job_item_status(claim.job_item_id) == "cancelled"

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
                repeat_number=int(snapshot.get("repeat_number", snapshot["run_number"])),
                condition_id=(str(snapshot["condition_id"]) if snapshot.get("condition_id") else None),
                condition_label=(str(snapshot["condition_label"]) if snapshot.get("condition_label") else None),
                ablation_flags=(dict(snapshot["ablation_flags"]) if snapshot.get("ablation_flags") else None),
                budget=dict(snapshot["budget"]) if snapshot.get("budget") else None,
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
            total_tokens=_extract_total_tokens(payload.token_usage),
            question_snapshot=_build_question_snapshot(unit.test_case),
            model_config_snapshot=execution.model_config,
            system_version_snapshot=_build_system_version_snapshot(unit=unit, payload=payload),
            derived_metrics=_build_derived_metrics(unit=unit, payload=payload),
            final_answer_hash=_final_answer_hash(payload.answer),
            status=CampaignResultStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
        )

    async def _derive_campaign_state(self, claim: ClaimedEvaluationWork) -> None:
        snapshot = claim.input_snapshot
        campaign = await self._campaign_repository.derive_execution_state(
            user_id=str(snapshot["user_id"]), campaign_id=str(snapshot["campaign_id"])
        )
        if self._ragas_evaluator is None:
            return
        # An execution-only rerun intentionally stops after promoting the
        # dataset result.  Combined and initial jobs continue to materialize
        # downstream metric work from the successful official projection.
        try:
            job = await self._store.get_job(
                user_id=str(snapshot["user_id"]), job_id=claim.job_id
            )
        except Exception:  # noqa: BLE001
            job = None
        if job is not None and job.config_snapshot.get("skip_ragas") is True:
            return
        if campaign.status.value not in {"completed", "completed_with_errors"}:
            return
        selected_result_ids: list[str] | None = None
        enabled_metrics = list(getattr(self._ragas_evaluator, "enabled_metrics", []))
        if job is not None:
            raw_question_ids = job.config_snapshot.get("downstream_question_ids")
            if isinstance(raw_question_ids, list) and raw_question_ids:
                results = await CampaignResultRepository().list_for_campaign(
                    user_id=str(snapshot["user_id"]),
                    campaign_id=str(snapshot["campaign_id"]),
                )
                question_ids = {str(value) for value in raw_question_ids}
                selected_result_ids = [
                    result.id for result in results if result.question_id in question_ids
                ]
            raw_metrics = job.config_snapshot.get("metric_names")
            if isinstance(raw_metrics, list) and raw_metrics:
                enabled_metrics = [str(value) for value in raw_metrics]
        ragas_config = getattr(campaign, "config", None)
        ragas_batch_size = getattr(ragas_config, "ragas_batch_size", None)
        ragas_parallel_batches = getattr(ragas_config, "ragas_parallel_batches", None)
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
            await self._campaign_repository.mark_evaluating(
                user_id=str(snapshot["user_id"]),
                campaign_id=str(snapshot["campaign_id"]),
                evaluation_total_units=created,
            )
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
