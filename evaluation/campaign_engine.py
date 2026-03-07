"""Async campaign engine for evaluation benchmarks."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

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
from evaluation.db import CampaignRepository, CampaignResultRepository
from evaluation.rag_modes import BenchmarkExecutionResult, run_campaign_case
from evaluation.ragas_evaluator import RagasEvaluator
from evaluation.retry import RateBudget
from evaluation.schemas import TestCase
from evaluation.storage import list_test_cases

logger = logging.getLogger(__name__)

CampaignRunner = Callable[..., Awaitable[BenchmarkExecutionResult]]


@dataclass(frozen=True)
class CampaignUnit:
    """One question-mode-run execution cell."""

    test_case: TestCase
    mode: str
    run_number: int


class CampaignEngine:
    """Create, run, cancel, and inspect evaluation campaigns."""

    def __init__(
        self,
        campaign_repository: Optional[CampaignRepository] = None,
        result_repository: Optional[CampaignResultRepository] = None,
        ragas_evaluator: Optional[RagasEvaluator] = None,
        runner: CampaignRunner = run_campaign_case,
    ) -> None:
        self._campaign_repository = campaign_repository or CampaignRepository()
        self._result_repository = result_repository or CampaignResultRepository()
        self._ragas_evaluator = ragas_evaluator or RagasEvaluator(
            result_repository=self._result_repository,
        )
        self._runner = runner
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._task_guard = asyncio.Lock()

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
        async with self._task_guard:
            self._active_tasks[created.id] = task
        task.add_done_callback(lambda _: asyncio.create_task(self._drop_active_task(created.id)))
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

    async def cancel_campaign(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        campaign = await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)
        if campaign.status in {
            CampaignLifecycleStatus.COMPLETED,
            CampaignLifecycleStatus.FAILED,
            CampaignLifecycleStatus.CANCELLED,
        }:
            return campaign

        updated = await self._campaign_repository.request_cancel(user_id=user_id, campaign_id=campaign_id)
        async with self._task_guard:
            active_task = self._active_tasks.get(campaign_id)
        if active_task is not None:
            active_task.cancel()
        return updated

    async def evaluate_campaign(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
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

        await self._campaign_repository.mark_evaluating(
            user_id=user_id,
            campaign_id=campaign_id,
            evaluation_total_units=len(completed_results),
        )
        task = asyncio.create_task(
            self._run_evaluation_only(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=campaign.completed_units,
                evaluation_total_units=len(completed_results),
            ),
            name=f"evaluation-ragas-{campaign_id}",
        )
        async with self._task_guard:
            self._active_tasks[campaign_id] = task
        task.add_done_callback(lambda _: asyncio.create_task(self._drop_active_task(campaign_id)))
        return await self.get_campaign(user_id=user_id, campaign_id=campaign_id)

    async def _run_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        config: CampaignConfig,
        test_cases: list[TestCase],
    ) -> None:
        batch_tasks: list[asyncio.Task[tuple[CampaignUnit, BenchmarkExecutionResult | Exception]]] = []
        try:
            await self._campaign_repository.mark_running(user_id=user_id, campaign_id=campaign_id)
            units = self._build_units(test_cases=test_cases, modes=config.modes, repeat_count=config.repeat_count)
            rate_budget = RateBudget(rpm_limit=config.rpm_limit)
            completed_units = 0

            for offset in range(0, len(units), config.batch_size):
                if await self._campaign_repository.is_cancel_requested(user_id=user_id, campaign_id=campaign_id):
                    await self._campaign_repository.mark_cancelled(user_id=user_id, campaign_id=campaign_id)
                    return

                batch = units[offset : offset + config.batch_size]
                batch_tasks = [
                    asyncio.create_task(
                        self._execute_unit(
                            unit=unit,
                            user_id=user_id,
                            model_config=config.model_preset.model_dump(mode="json"),
                            rate_budget=rate_budget,
                        )
                    )
                    for unit in batch
                ]

                for completed_task in asyncio.as_completed(batch_tasks):
                    unit, payload = await completed_task
                    result = await self._persist_unit_result(
                        user_id=user_id,
                        campaign_id=campaign_id,
                        unit=unit,
                        payload=payload,
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
                        len(units),
                        result.id,
                    )

            await self._run_ragas_evaluation(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=completed_units,
            )
        except asyncio.CancelledError:
            for task in batch_tasks:
                task.cancel()
            if user_id and campaign_id:
                await self._campaign_repository.mark_cancelled(user_id=user_id, campaign_id=campaign_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Campaign %s failed: %s", campaign_id, exc, exc_info=True)
            await self._campaign_repository.mark_failed(
                user_id=user_id,
                campaign_id=campaign_id,
                error_message=str(exc),
                phase="evaluation",
            )

    async def _run_evaluation_only(
        self,
        *,
        user_id: str,
        campaign_id: str,
        completed_units: int,
        evaluation_total_units: int,
    ) -> None:
        try:
            await self._evaluate_campaign_results(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=completed_units,
                evaluation_total_units=evaluation_total_units,
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
            on_progress=on_progress,
        )

    async def _execute_unit(
        self,
        *,
        unit: CampaignUnit,
        user_id: str,
        model_config: dict,
        rate_budget: RateBudget,
    ) -> tuple[CampaignUnit, BenchmarkExecutionResult | Exception]:
        await rate_budget.acquire()
        try:
            payload = await self._runner(
                test_case=unit.test_case,
                user_id=user_id,
                mode=unit.mode,
                model_config=model_config,
            )
            return unit, payload
        except Exception as exc:  # noqa: BLE001
            return unit, exc

    async def _persist_unit_result(
        self,
        *,
        user_id: str,
        campaign_id: str,
        unit: CampaignUnit,
        payload: BenchmarkExecutionResult | Exception,
    ):
        if isinstance(payload, Exception):
            return await self._result_repository.create(
                user_id=user_id,
                campaign_id=campaign_id,
                question_id=unit.test_case.id,
                question=unit.test_case.question,
                ground_truth=unit.test_case.ground_truth,
                mode=unit.mode,
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
            )

        return await self._result_repository.create(
            user_id=user_id,
            campaign_id=campaign_id,
            question_id=payload.question_id,
            question=payload.question,
            ground_truth=payload.ground_truth,
            mode=payload.mode,
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
        )

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
