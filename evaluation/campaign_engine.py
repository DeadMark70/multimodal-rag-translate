"""Async campaign engine for evaluation benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from core.errors import AppError, ErrorCode
from evaluation.campaign_schemas import (
    CampaignMetricsResponse,
    CampaignConfig,
    CampaignCreateResponse,
    CampaignLifecycleStatus,
    CampaignResultStatus,
    CampaignResultsResponse,
    CampaignStatus,
)
from evaluation.db import AgentTraceRepository, CampaignRepository, CampaignResultRepository
from evaluation.observability import EvaluationRunRecorder
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.rag_modes import BenchmarkExecutionResult, run_campaign_case
from evaluation.ragas_evaluator import RagasEvaluator
from evaluation.retry import RateBudget
from evaluation.schemas import TestCase
from evaluation.storage import list_test_cases
from evaluation.trace_schemas import AgentTraceDetail, AgentTraceSummary, EvaluationTraceEvent

logger = logging.getLogger(__name__)

CampaignRunner = Callable[..., Awaitable[BenchmarkExecutionResult]]
_TERMINAL_STATUSES = {
    CampaignLifecycleStatus.COMPLETED,
    CampaignLifecycleStatus.FAILED,
    CampaignLifecycleStatus.CANCELLED,
}


@dataclass(frozen=True)
class CampaignUnit:
    """One question-mode-run execution cell."""

    test_case: TestCase
    mode: str
    run_number: int


@dataclass(frozen=True)
class ExecutedCampaignUnit:
    """One executed unit plus immutable snapshot metadata."""

    unit: CampaignUnit
    payload: BenchmarkExecutionResult | Exception
    run_id: str
    request_id: str
    started_at: datetime
    completed_at: datetime
    total_latency_ms: float
    model_config: dict[str, Any]


def _unit_key(unit: CampaignUnit) -> tuple[str, str, int]:
    return (unit.test_case.id, unit.mode, unit.run_number)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _duration_ms(started_at: datetime, completed_at: datetime) -> float:
    return max((completed_at - started_at).total_seconds() * 1000, 0)


def _extract_total_tokens(token_usage: dict[str, Any]) -> int:
    raw_total = token_usage.get("total_tokens")
    if isinstance(raw_total, dict):
        total = 0
        for value in raw_total.values():
            try:
                total += int(value or 0)
            except (TypeError, ValueError):
                continue
        return total
    try:
        if raw_total is not None:
            return int(raw_total)
    except (TypeError, ValueError):
        pass

    total = 0
    for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens"):
        try:
            total += int(token_usage.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _build_question_snapshot(test_case: TestCase) -> dict[str, Any]:
    return {
        "id": test_case.id,
        "question": test_case.question,
        "ground_truth": test_case.ground_truth,
        "ground_truth_short": test_case.ground_truth_short,
        "key_points": list(test_case.key_points),
        "ragas_focus": list(test_case.ragas_focus),
        "category": test_case.category,
        "difficulty": test_case.difficulty,
        "question_version": test_case.question_version,
        "required_modalities": list(test_case.required_modalities),
        "atomic_facts": list(test_case.atomic_facts),
        "expected_evidence": list(test_case.expected_evidence),
        "source_docs": list(test_case.source_docs),
    }


def _build_system_version_snapshot(
    *,
    unit: CampaignUnit,
    payload: BenchmarkExecutionResult | Exception,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "mode": unit.mode,
        "run_number": unit.run_number,
    }
    if isinstance(payload, BenchmarkExecutionResult):
        if payload.execution_profile:
            snapshot["execution_profile"] = payload.execution_profile
        if payload.context_policy_version:
            snapshot["context_policy_version"] = payload.context_policy_version
    return snapshot


def _build_derived_metrics(
    *,
    payload: BenchmarkExecutionResult | Exception,
) -> dict[str, Any]:
    if isinstance(payload, Exception):
        return {}
    return {
        "context_count": len(payload.contexts),
        "source_doc_count": len(payload.source_doc_ids),
        "expected_source_count": len(payload.expected_sources),
    }


def _final_answer_hash(answer: str | None) -> str | None:
    if not answer:
        return None
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


async def _record_unit_root_span(
    *,
    run_id: str,
    campaign_id: str,
    request_id: str,
    unit: CampaignUnit,
    started_at: datetime,
    completed_at: datetime,
    duration_ms: float,
    failed: bool,
) -> str | None:
    repository = EvaluationObservabilityRepository()
    span_id = str(uuid4())
    created_at = _utc_now()
    payload = {
        "request_id": request_id,
        "question_id": unit.test_case.id,
        "mode": unit.mode,
        "run_number": unit.run_number,
    }
    error = {"type": "CampaignUnitFailed", "message": "Campaign unit failed"} if failed else {}
    try:
        await repository.record_trace_events(
            [
                EvaluationTraceEvent(
                    event_id=str(uuid4()),
                    run_id=run_id,
                    campaign_id=campaign_id,
                    span_id=span_id,
                    parent_event_id=None,
                    parent_span_id=None,
                    event_type="campaign_unit_execution",
                    sequence=1,
                    stage_type="generation",
                    stage_name="campaign_unit_execution",
                    started_at=started_at,
                    ended_at=None,
                    duration_ms=None,
                    status="running",
                    payload=payload,
                    error={},
                    created_at=created_at,
                ),
                EvaluationTraceEvent(
                    event_id=str(uuid4()),
                    run_id=run_id,
                    campaign_id=campaign_id,
                    span_id=span_id,
                    parent_event_id=None,
                    parent_span_id=None,
                    event_type="campaign_unit_execution",
                    sequence=2,
                    stage_type="generation",
                    stage_name="campaign_unit_execution",
                    started_at=started_at,
                    ended_at=completed_at,
                    duration_ms=duration_ms,
                    status="failed" if failed else "success",
                    payload=payload,
                    error=error,
                    created_at=created_at,
                ),
            ]
        )
        return span_id
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to record campaign unit observability span",
            extra={
                "campaign_id": campaign_id,
                "run_id": run_id,
                "request_id": request_id,
                "question_id": unit.test_case.id,
                "mode": unit.mode,
            },
            exc_info=True,
        )
        return None


async def _record_unit_llm_usage(
    *,
    run_id: str,
    campaign_id: str,
    user_id: str,
    request_id: str,
    span_id: str | None,
    execution: ExecutedCampaignUnit,
) -> None:
    if not isinstance(execution.payload, BenchmarkExecutionResult):
        return
    if not execution.payload.token_usage:
        return

    model_name = execution.model_config.get("model_name")
    provider = execution.model_config.get("provider")
    if provider is None and isinstance(model_name, str) and model_name.startswith("gemini"):
        provider = "google"

    recorder = EvaluationRunRecorder(
        run_id=run_id,
        campaign_id=campaign_id,
        user_id=user_id,
        request_id=request_id,
    )
    await recorder.record_llm_usage(
        purpose="campaign_generation",
        provider=provider,
        model_name=str(model_name) if model_name else None,
        usage=execution.payload.token_usage,
        latency_ms=execution.payload.latency_ms,
        status="failed" if execution.payload.error_message else "success",
        error=(
            {"message": execution.payload.error_message}
            if execution.payload.error_message
            else None
        ),
        span_id=span_id,
        payload={
            "request_id": request_id,
            "question_id": execution.unit.test_case.id,
            "mode": execution.unit.mode,
            "run_number": execution.unit.run_number,
            "root_span_recorded": span_id is not None,
        },
        created_at=execution.completed_at,
    )


async def _cancel_and_drain_tasks(tasks: list[asyncio.Task]) -> None:
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


class CampaignEngine:
    """Create, run, cancel, and inspect evaluation campaigns."""

    def __init__(
        self,
        campaign_repository: Optional[CampaignRepository] = None,
        result_repository: Optional[CampaignResultRepository] = None,
        trace_repository: Optional[AgentTraceRepository] = None,
        ragas_evaluator: Optional[RagasEvaluator] = None,
        runner: CampaignRunner = run_campaign_case,
    ) -> None:
        self._campaign_repository = campaign_repository or CampaignRepository()
        self._result_repository = result_repository or CampaignResultRepository()
        self._trace_repository = trace_repository or AgentTraceRepository()
        self._ragas_evaluator = ragas_evaluator or RagasEvaluator(
            result_repository=self._result_repository,
        )
        self._runner = runner
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._task_guard = asyncio.Lock()

    async def _register_active_task(self, campaign_id: str, task: asyncio.Task[None]) -> None:
        async with self._task_guard:
            self._active_tasks[campaign_id] = task
        task.add_done_callback(lambda _: asyncio.create_task(self._drop_active_task(campaign_id)))

    async def _get_active_task(self, campaign_id: str) -> asyncio.Task[None] | None:
        async with self._task_guard:
            task = self._active_tasks.get(campaign_id)
        if task is not None and task.done():
            await self._drop_active_task(campaign_id)
            return None
        return task

    async def create_and_start(
        self,
        *,
        user_id: str,
        name: Optional[str],
        config: CampaignConfig,
    ) -> CampaignCreateResponse:
        resolved_cases = await self._resolve_test_cases(user_id=user_id, test_case_ids=config.test_case_ids)
        created = await self._campaign_repository.create(user_id=user_id, name=name, config=config)
        task = asyncio.create_task(
            self._run_campaign(
                user_id=user_id,
                campaign_id=created.id,
                config=config,
                test_cases=resolved_cases,
            ),
            name=f"evaluation-campaign-{created.id}",
        )
        await self._register_active_task(created.id, task)
        return CampaignCreateResponse(campaign_id=created.id, status=created.status)

    async def list_campaigns(self, *, user_id: str) -> list[CampaignStatus]:
        return await self._campaign_repository.list_by_user(user_id=user_id)

    async def get_campaign(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        return await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)

    async def get_results(self, *, user_id: str, campaign_id: str) -> CampaignResultsResponse:
        campaign = await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        return CampaignResultsResponse(campaign=campaign, results=results)

    async def get_metrics(self, *, user_id: str, campaign_id: str) -> CampaignMetricsResponse:
        campaign = await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        return await self._ragas_evaluator.get_metrics(user_id=user_id, campaign=campaign)

    async def list_traces(self, *, user_id: str, campaign_id: str) -> list[AgentTraceSummary]:
        await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        return await self._trace_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)

    async def get_trace(
        self,
        *,
        user_id: str,
        campaign_id: str,
        campaign_result_id: str,
    ) -> AgentTraceDetail:
        await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        return await self._trace_repository.get_for_result(
            user_id=user_id,
            campaign_id=campaign_id,
            campaign_result_id=campaign_result_id,
        )

    async def cancel_campaign(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        campaign = await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)
        if campaign.status in _TERMINAL_STATUSES:
            return campaign

        await self._campaign_repository.request_cancel(user_id=user_id, campaign_id=campaign_id)
        active_task = await self._get_active_task(campaign_id)
        if active_task is not None:
            active_task.cancel()
            return await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)
        return await self._campaign_repository.mark_cancelled(user_id=user_id, campaign_id=campaign_id)

    async def evaluate_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        question_ids: Optional[list[str]] = None,
    ) -> CampaignStatus:
        campaign = await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)
        if campaign.status in {
            CampaignLifecycleStatus.RUNNING,
            CampaignLifecycleStatus.EVALUATING,
        }:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="Campaign is already running",
                status_code=400,
            )

        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        completed_results = [row for row in results if row.status == CampaignResultStatus.COMPLETED]
        if not completed_results:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="Campaign has no completed raw results to evaluate",
                status_code=400,
            )

        selected_completed_results = completed_results
        normalized_question_ids = [question_id for question_id in (question_ids or []) if question_id]
        if normalized_question_ids:
            selected_question_id_set = set(normalized_question_ids)
            selected_completed_results = [
                row for row in completed_results if row.question_id in selected_question_id_set
            ]
            if not selected_completed_results:
                raise AppError(
                    code=ErrorCode.BAD_REQUEST,
                    message=(
                        "Requested question_ids have no completed raw results in this campaign"
                    ),
                    status_code=400,
                )

        selected_result_ids = [row.id for row in selected_completed_results]
        await self._campaign_repository.mark_evaluating(
            user_id=user_id,
            campaign_id=campaign_id,
            evaluation_total_units=len(selected_completed_results),
        )
        task = asyncio.create_task(
            self._run_evaluation_only(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=campaign.completed_units,
                evaluation_total_units=len(selected_completed_results),
                ragas_batch_size=campaign.config.ragas_batch_size,
                ragas_parallel_batches=campaign.config.ragas_parallel_batches,
                ragas_rpm_limit=campaign.config.ragas_rpm_limit,
                selected_result_ids=selected_result_ids,
            ),
            name=f"evaluation-ragas-{campaign_id}",
        )
        await self._register_active_task(campaign_id, task)
        return await self.get_campaign(user_id=user_id, campaign_id=campaign_id)

    async def recover_inflight_campaigns(self) -> None:
        """Recover non-terminal campaigns after process restart."""
        inflight = await self._campaign_repository.list_inflight()
        if not inflight:
            return

        for user_id, campaign in inflight:
            try:
                await self.ensure_campaign_task(
                    user_id=user_id,
                    campaign_id=campaign.id,
                    campaign_snapshot=campaign,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to recover campaign %s for user %s: %s",
                    campaign.id,
                    user_id,
                    exc,
                    exc_info=True,
                )

    async def ensure_campaign_task(
        self,
        *,
        user_id: str,
        campaign_id: str,
        campaign_snapshot: CampaignStatus | None = None,
    ) -> CampaignStatus:
        """Ensure one non-terminal campaign has a running task or terminal state."""
        campaign = campaign_snapshot or await self.get_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        if campaign.status in _TERMINAL_STATUSES:
            return campaign

        active_task = await self._get_active_task(campaign_id)
        if active_task is not None:
            return campaign

        if campaign.cancel_requested:
            return await self._campaign_repository.mark_cancelled(
                user_id=user_id, campaign_id=campaign_id
            )

        try:
            if campaign.status in {
                CampaignLifecycleStatus.PENDING,
                CampaignLifecycleStatus.RUNNING,
            }:
                resolved_cases = await self._resolve_test_cases(
                    user_id=user_id,
                    test_case_ids=campaign.config.test_case_ids,
                )
                all_units = self._build_units(
                    test_cases=resolved_cases,
                    modes=campaign.config.modes,
                    repeat_count=campaign.config.repeat_count,
                )
                all_unit_keys = {_unit_key(unit) for unit in all_units}
                existing_results = await self._result_repository.list_for_campaign(
                    user_id=user_id,
                    campaign_id=campaign_id,
                )
                completed_keys = {
                    (row.question_id, row.mode, row.run_number)
                    for row in existing_results
                    if (row.question_id, row.mode, row.run_number) in all_unit_keys
                }
                remaining_units = [
                    unit for unit in all_units if _unit_key(unit) not in completed_keys
                ]
                completed_units = len(completed_keys)
                await self._campaign_repository.update_progress(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    completed_units=completed_units,
                    current_question_id=campaign.current_question_id,
                    current_mode=campaign.current_mode,
                )

                task = asyncio.create_task(
                    self._run_campaign(
                        user_id=user_id,
                        campaign_id=campaign_id,
                        config=campaign.config,
                        test_cases=resolved_cases,
                        units=remaining_units,
                        initial_completed_units=completed_units,
                        total_units_override=len(all_units),
                    ),
                    name=f"evaluation-campaign-recovery-{campaign_id}",
                )
                await self._register_active_task(campaign_id, task)
                return await self.get_campaign(user_id=user_id, campaign_id=campaign_id)

            if campaign.status == CampaignLifecycleStatus.EVALUATING:
                results = await self._result_repository.list_for_campaign(
                    user_id=user_id,
                    campaign_id=campaign_id,
                )
                completed_results = [
                    row
                    for row in results
                    if row.status == CampaignResultStatus.COMPLETED
                ]
                selected_result_ids = [row.id for row in completed_results]
                await self._campaign_repository.mark_evaluating(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    evaluation_total_units=len(selected_result_ids),
                )
                task = asyncio.create_task(
                    self._run_evaluation_only(
                        user_id=user_id,
                        campaign_id=campaign_id,
                        completed_units=campaign.completed_units,
                        evaluation_total_units=len(selected_result_ids),
                        ragas_batch_size=campaign.config.ragas_batch_size,
                        ragas_parallel_batches=campaign.config.ragas_parallel_batches,
                        ragas_rpm_limit=campaign.config.ragas_rpm_limit,
                        selected_result_ids=selected_result_ids,
                    ),
                    name=f"evaluation-ragas-recovery-{campaign_id}",
                )
                await self._register_active_task(campaign_id, task)
                return await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        except Exception as exc:  # noqa: BLE001
            return await self._campaign_repository.mark_failed(
                user_id=user_id,
                campaign_id=campaign_id,
                error_message=f"Campaign recovery failed: {exc}",
                phase=campaign.phase,
            )

        return campaign

    async def _run_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        config: CampaignConfig,
        test_cases: list[TestCase],
        units: list[CampaignUnit] | None = None,
        initial_completed_units: int = 0,
        total_units_override: int | None = None,
    ) -> None:
        batch_tasks: list[asyncio.Task] = []
        try:
            await self._campaign_repository.mark_running(user_id=user_id, campaign_id=campaign_id)
            pending_units = units
            if pending_units is None:
                pending_units = self._build_units(
                    test_cases=test_cases,
                    modes=config.modes,
                    repeat_count=config.repeat_count,
                )
            rate_budget = RateBudget(rpm_limit=config.rpm_limit)
            completed_units = initial_completed_units
            total_units = total_units_override or (
                initial_completed_units + len(pending_units)
            )

            for offset in range(0, len(pending_units), config.batch_size):
                if await self._campaign_repository.is_cancel_requested(user_id=user_id, campaign_id=campaign_id):
                    await self._campaign_repository.mark_cancelled(user_id=user_id, campaign_id=campaign_id)
                    return

                batch = pending_units[offset : offset + config.batch_size]
                batch_tasks = [
                    asyncio.create_task(
                        self._execute_unit(
                            unit=unit,
                            user_id=user_id,
                            model_config=config.model_preset.model_dump(mode="json"),
                            rate_budget=rate_budget,
                            run_number=unit.run_number,
                        )
                    )
                    for unit in batch
                ]

                for completed_task in asyncio.as_completed(batch_tasks):
                    execution = await completed_task
                    unit = execution.unit
                    result = await self._persist_unit_result(
                        user_id=user_id,
                        campaign_id=campaign_id,
                        execution=execution,
                    )
                    completed_units += 1
                    await self._campaign_repository.update_progress(
                        user_id=user_id,
                        campaign_id=campaign_id,
                        completed_units=completed_units,
                        current_question_id=unit.test_case.id,
                        current_mode=unit.mode,
                    )
                    logger.info(
                        "Campaign %s progress %s/%s latest_result=%s",
                        campaign_id,
                        completed_units,
                        total_units,
                        result.id,
                    )

            await self._run_ragas_evaluation(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=completed_units,
                ragas_batch_size=config.ragas_batch_size,
                ragas_parallel_batches=config.ragas_parallel_batches,
                ragas_rpm_limit=config.ragas_rpm_limit,
            )
        except asyncio.CancelledError:
            await _cancel_and_drain_tasks(batch_tasks)
            if user_id and campaign_id:
                await self._campaign_repository.mark_cancelled(user_id=user_id, campaign_id=campaign_id)
            raise
        except Exception as exc:  # noqa: BLE001
            await _cancel_and_drain_tasks(batch_tasks)
            logger.error("Campaign %s failed: %s", campaign_id, exc, exc_info=True)
            await self._campaign_repository.mark_failed(
                user_id=user_id,
                campaign_id=campaign_id,
                error_message=str(exc),
                phase="execution",
            )

    async def _run_evaluation_only(
        self,
        *,
        user_id: str,
        campaign_id: str,
        completed_units: int,
        evaluation_total_units: int,
        ragas_batch_size: int,
        ragas_parallel_batches: int,
        ragas_rpm_limit: int,
        selected_result_ids: Optional[list[str]] = None,
    ) -> None:
        try:
            await self._evaluate_campaign_results(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=completed_units,
                evaluation_total_units=evaluation_total_units,
                ragas_batch_size=ragas_batch_size,
                ragas_parallel_batches=ragas_parallel_batches,
                ragas_rpm_limit=ragas_rpm_limit,
                selected_result_ids=selected_result_ids,
            )
            await self._campaign_repository.mark_completed(
                user_id=user_id,
                campaign_id=campaign_id,
                phase="evaluation",
            )
        except asyncio.CancelledError:
            await self._campaign_repository.mark_cancelled(user_id=user_id, campaign_id=campaign_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Campaign %s evaluation rerun failed: %s", campaign_id, exc, exc_info=True)
            await self._campaign_repository.mark_failed(
                user_id=user_id,
                campaign_id=campaign_id,
                error_message=str(exc),
                phase="evaluation",
            )

    async def _run_ragas_evaluation(
        self,
        *,
        user_id: str,
        campaign_id: str,
        completed_units: int,
        ragas_batch_size: int,
        ragas_parallel_batches: int,
        ragas_rpm_limit: int,
    ) -> None:
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        completed_results = [row for row in results if row.status == CampaignResultStatus.COMPLETED]
        if not completed_results:
            await self._campaign_repository.mark_completed(user_id=user_id, campaign_id=campaign_id)
            return

        await self._campaign_repository.mark_evaluating(
            user_id=user_id,
            campaign_id=campaign_id,
            evaluation_total_units=len(completed_results),
        )
        await self._evaluate_campaign_results(
            user_id=user_id,
            campaign_id=campaign_id,
            completed_units=completed_units,
            evaluation_total_units=len(completed_results),
            ragas_batch_size=ragas_batch_size,
            ragas_parallel_batches=ragas_parallel_batches,
            ragas_rpm_limit=ragas_rpm_limit,
        )
        await self._campaign_repository.mark_completed(
            user_id=user_id,
            campaign_id=campaign_id,
            phase="evaluation",
        )

    async def _evaluate_campaign_results(
        self,
        *,
        user_id: str,
        campaign_id: str,
        completed_units: int,
        evaluation_total_units: int,
        ragas_batch_size: int,
        ragas_parallel_batches: int,
        ragas_rpm_limit: int,
        selected_result_ids: Optional[list[str]] = None,
    ) -> None:
        async def on_progress(
            evaluation_completed_units: int,
            _evaluation_total_units: int,
            current_question_id: str | None,
            current_mode: str | None,
        ) -> None:
            await self._campaign_repository.update_progress(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=completed_units,
                evaluation_completed_units=evaluation_completed_units,
                evaluation_total_units=evaluation_total_units,
                current_question_id=current_question_id,
                current_mode=current_mode,
            )

        await self._ragas_evaluator.evaluate_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            ragas_batch_size=ragas_batch_size,
            ragas_parallel_batches=ragas_parallel_batches,
            ragas_rpm_limit=ragas_rpm_limit,
            selected_result_ids=selected_result_ids,
            on_progress=on_progress,
        )

    async def _execute_unit(
        self,
        *,
        unit: CampaignUnit,
        user_id: str,
        model_config: dict,
        rate_budget: RateBudget,
        run_number: int,
    ) -> ExecutedCampaignUnit:
        await rate_budget.acquire()
        run_id = str(uuid4())
        request_id = str(uuid4())
        started_at = _utc_now()
        runner_started_perf = time.perf_counter()
        try:
            payload = await self._runner(
                test_case=unit.test_case,
                user_id=user_id,
                mode=unit.mode,
                model_config=model_config,
                run_number=run_number,
            )
        except Exception as exc:  # noqa: BLE001
            payload = exc
        completed_at = _utc_now()
        total_latency_ms = max((time.perf_counter() - runner_started_perf) * 1000, 0)
        if total_latency_ms <= 0:
            total_latency_ms = _duration_ms(started_at, completed_at)
        return ExecutedCampaignUnit(
            unit=unit,
            payload=payload,
            run_id=run_id,
            request_id=request_id,
            started_at=started_at,
            completed_at=completed_at,
            total_latency_ms=total_latency_ms,
            model_config=dict(model_config),
        )

    async def _persist_unit_result(
        self,
        *,
        user_id: str,
        campaign_id: str,
        execution: ExecutedCampaignUnit,
    ):
        unit = execution.unit
        payload = execution.payload
        question_snapshot = _build_question_snapshot(unit.test_case)
        total_tokens = (
            _extract_total_tokens(payload.token_usage)
            if isinstance(payload, BenchmarkExecutionResult)
            else None
        )
        system_version_snapshot = _build_system_version_snapshot(unit=unit, payload=payload)
        derived_metrics = _build_derived_metrics(payload=payload)

        if isinstance(payload, Exception):
            created = await self._result_repository.create(
                result_id=execution.run_id,
                user_id=user_id,
                campaign_id=campaign_id,
                question_id=unit.test_case.id,
                question=unit.test_case.question,
                ground_truth=unit.test_case.ground_truth,
                ground_truth_short=unit.test_case.ground_truth_short,
                key_points=list(unit.test_case.key_points),
                ragas_focus=list(unit.test_case.ragas_focus),
                mode=unit.mode,
                execution_profile=(
                    getattr(payload, "agent_trace", {}) or {}
                ).get("execution_profile"),
                context_policy_version=None,
                run_number=unit.run_number,
                answer=f"ERROR: {payload}",
                contexts=[],
                source_doc_ids=[],
                expected_sources=list(unit.test_case.source_docs),
                latency_ms=0,
                token_usage={},
                category=unit.test_case.category,
                difficulty=unit.test_case.difficulty,
                status=CampaignResultStatus.FAILED,
                error_message=str(payload),
                question_version=unit.test_case.question_version,
                request_id=execution.request_id,
                started_at=execution.started_at.isoformat(),
                completed_at=execution.completed_at.isoformat(),
                total_latency_ms=execution.total_latency_ms,
                total_tokens=total_tokens,
                question_snapshot=question_snapshot,
                model_config_snapshot=execution.model_config,
                system_version_snapshot=system_version_snapshot,
                derived_metrics=derived_metrics,
            )
            span_id = await _record_unit_root_span(
                run_id=created.id,
                campaign_id=campaign_id,
                request_id=execution.request_id,
                unit=unit,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_ms=execution.total_latency_ms,
                failed=True,
            )
            await _record_unit_llm_usage(
                run_id=created.id,
                campaign_id=campaign_id,
                user_id=user_id,
                request_id=execution.request_id,
                span_id=span_id,
                execution=execution,
            )
            trace_payload = getattr(payload, "agent_trace", None)
            if trace_payload:
                await self._trace_repository.replace_for_result(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    campaign_result_id=created.id,
                    trace_payload=trace_payload,
                )
            return created

        created = await self._result_repository.create(
            result_id=execution.run_id,
            user_id=user_id,
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
            answer=payload.answer,
            contexts=payload.contexts,
            source_doc_ids=payload.source_doc_ids,
            expected_sources=payload.expected_sources,
            latency_ms=payload.latency_ms,
            token_usage=payload.token_usage,
            category=payload.category,
            difficulty=payload.difficulty,
            status=CampaignResultStatus.COMPLETED if not payload.error_message else CampaignResultStatus.FAILED,
            error_message=payload.error_message,
            question_version=unit.test_case.question_version,
            request_id=execution.request_id,
            started_at=execution.started_at.isoformat(),
            completed_at=execution.completed_at.isoformat(),
            total_latency_ms=execution.total_latency_ms,
            total_tokens=total_tokens,
            question_snapshot=question_snapshot,
            model_config_snapshot=execution.model_config,
            system_version_snapshot=system_version_snapshot,
            derived_metrics=derived_metrics,
            final_answer_hash=_final_answer_hash(payload.answer),
        )
        span_id = await _record_unit_root_span(
            run_id=created.id,
            campaign_id=campaign_id,
            request_id=execution.request_id,
            unit=unit,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_ms=execution.total_latency_ms,
            failed=payload.error_message is not None,
        )
        await _record_unit_llm_usage(
            run_id=created.id,
            campaign_id=campaign_id,
            user_id=user_id,
            request_id=execution.request_id,
            span_id=span_id,
            execution=execution,
        )
        if payload.agent_trace:
            await self._trace_repository.replace_for_result(
                user_id=user_id,
                campaign_id=campaign_id,
                campaign_result_id=created.id,
                trace_payload=payload.agent_trace,
            )
        return created

    async def _resolve_test_cases(self, *, user_id: str, test_case_ids: list[str]) -> list[TestCase]:
        available = [TestCase.model_validate(item) for item in await list_test_cases(user_id)]
        by_id = {item.id: item for item in available}
        selected: list[TestCase] = []
        missing: list[str] = []
        for test_case_id in test_case_ids:
            test_case = by_id.get(test_case_id)
            if test_case is None:
                missing.append(test_case_id)
            else:
                selected.append(test_case)

        if missing:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message=f"Unknown test case ids: {', '.join(missing)}",
                status_code=400,
            )
        return selected

    @staticmethod
    def _build_units(
        *,
        test_cases: list[TestCase],
        modes: list[str],
        repeat_count: int,
    ) -> list[CampaignUnit]:
        units: list[CampaignUnit] = []
        for run_number in range(1, repeat_count + 1):
            for test_case in test_cases:
                for mode in modes:
                    units.append(CampaignUnit(test_case=test_case, mode=mode, run_number=run_number))
        return units

    async def _drop_active_task(self, campaign_id: str) -> None:
        async with self._task_guard:
            self._active_tasks.pop(campaign_id, None)


_campaign_engine: Optional[CampaignEngine] = None


def get_campaign_engine() -> CampaignEngine:
    global _campaign_engine
    if _campaign_engine is None:
        _campaign_engine = CampaignEngine()
    return _campaign_engine

