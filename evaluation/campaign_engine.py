"""Async campaign engine for evaluation benchmarks."""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional

from core.errors import AppError, ErrorCode
from evaluation.campaign_schemas import (
    AblationCondition,
    CampaignMetricsResponse,
    CampaignConfig,
    CampaignCreateResponse,
    CampaignLifecycleStatus,
    CampaignResultStatus,
    CampaignResultsResponse,
    CampaignStatus,
)
from evaluation.db import (
    AgentTraceRepository,
    CampaignRepository,
    CampaignResultRepository,
)
from evaluation.job_schemas import (
    EvaluationAttempt,
    EvaluationJob,
    EvaluationJobItemSummary,
    EvaluationJobType,
    EvaluationRerunRequest,
    EvaluationWorkType,
    WorkItemSpec,
)
from evaluation.job_store import EvaluationJobStore
from evaluation.agentic_campaign_adapter import effective_agentic_execution_version
from evaluation.rag_modes import run_campaign_case
from evaluation.ragas_evaluator import RagasEvaluator
from evaluation.schemas import TestCase
from evaluation.storage import list_test_cases
from evaluation.trace_schemas import (
    AgentTraceDetail,
    AgentTraceSummary,
)
from evaluation.campaign_execution import (
    CampaignRunner,
    CampaignUnit,
)


logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {
    CampaignLifecycleStatus.COMPLETED,
    CampaignLifecycleStatus.COMPLETED_WITH_ERRORS,
    CampaignLifecycleStatus.FAILED,
    CampaignLifecycleStatus.CANCELLED,
}
_LEGACY_RAGAS_METRIC = "legacy_campaign"


class CampaignEngine:
    """Create, run, cancel, and inspect evaluation campaigns."""

    def __init__(
        self,
        campaign_repository: Optional[CampaignRepository] = None,
        result_repository: Optional[CampaignResultRepository] = None,
        trace_repository: Optional[AgentTraceRepository] = None,
        ragas_evaluator: Optional[RagasEvaluator] = None,
        runner: CampaignRunner = run_campaign_case,
        job_store: EvaluationJobStore | None = None,
        worker_notifier: Callable[[], None] | None = None,
        configure_worker: bool = True,
    ) -> None:
        self._campaign_repository = campaign_repository or CampaignRepository()
        self._result_repository = result_repository or CampaignResultRepository()
        self._trace_repository = trace_repository or AgentTraceRepository()
        self._ragas_evaluator = ragas_evaluator or RagasEvaluator(
            result_repository=self._result_repository,
        )
        self._runner = runner
        self._job_store = job_store or EvaluationJobStore()
        worker = None
        self._worker = None
        self._worker_owned = False
        if worker_notifier is None:
            from evaluation.execution_worker import DatasetExecutionWorker
            from evaluation.accounting_store import EvaluationAccountingStore
            from evaluation.job_worker import configure_evaluation_job_worker
            from evaluation.job_worker import EvaluationJobWorker
            from evaluation.job_worker import get_evaluation_job_worker
            from evaluation.ragas_worker import RagasBatchWorker

            # The application singleton is reserved for the real process
            # engine.  Injected runners/evaluators (tests and embedded
            # callers) get an isolated worker so a lifespan-owned singleton
            # cannot process their ledger with stale handlers.
            use_process_worker = (
                runner is run_campaign_case
                and ragas_evaluator is None
                and job_store is None
            )
            accounting_store = EvaluationAccountingStore()
            worker = (
                get_evaluation_job_worker()
                if use_process_worker
                else EvaluationJobWorker(
                    store=self._job_store,
                    stop_when_idle=True,
                )
            )
            execution_handler = DatasetExecutionWorker(
                store=self._job_store,
                runner=runner,
                result_repository=self._result_repository,
                ragas_evaluator=self._ragas_evaluator,
                notify=worker.notify,
            ).execute
            ragas_batch_handler = RagasBatchWorker(
                store=self._job_store,
                evaluator=self._ragas_evaluator,
                campaign_repository=self._campaign_repository,
                accounting_store=accounting_store,
            ).execute
            if configure_worker:
                if use_process_worker:
                    configure_evaluation_job_worker(
                        execution_handler=execution_handler,
                        ragas_batch_handler=ragas_batch_handler,
                    )
                else:
                    worker.configure_handlers(
                        execution_handler=execution_handler,
                        ragas_batch_handler=ragas_batch_handler,
                    )
            worker_notifier = worker.notify
            self._worker = worker
            self._worker_owned = not use_process_worker
        self._worker_notifier = worker_notifier
        if getattr(self._job_store, "_on_job_created", None) is None:
            self._job_store._on_job_created = (
                worker.notify if worker is not None else worker_notifier
            )

    async def create_and_start(
        self,
        *,
        user_id: str,
        name: Optional[str],
        config: CampaignConfig,
    ) -> CampaignCreateResponse:
        if "router" in config.modes and not config.actual_router_execution_enabled:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="router mode is not implemented yet; use retrospective router analysis.",
                status_code=400,
            )
        resolved_cases = await self._resolve_test_cases(
            user_id=user_id, test_case_ids=config.test_case_ids
        )
        created = await self._campaign_repository.create(
            user_id=user_id, name=name, config=config
        )
        units = self._build_units(
            test_cases=resolved_cases,
            modes=config.modes,
            repeat_count=config.repeat_count,
            ablation_conditions=config.ablation_conditions,
            agentic_execution_version=config.agentic_execution_version,
            shadow_evaluation_policy=config.shadow_evaluation_policy,
        )
        await self._job_store.create_job_with_items(
            user_id=user_id,
            campaign_id=created.id,
            job_type=EvaluationJobType.INITIAL,
            selection={"campaign_id": created.id},
            config_snapshot=config.model_dump(mode="json", by_alias=True),
            items=[
                self._work_item_spec(
                    user_id=user_id, campaign_id=created.id, unit=unit, config=config
                )
                for unit in units
            ],
        )
        await self._start_worker_if_available()
        if self._worker_notifier is not None:
            self._worker_notifier()
        return CampaignCreateResponse(campaign_id=created.id, status=created.status)

    async def list_campaigns(self, *, user_id: str) -> list[CampaignStatus]:
        campaigns = await self._campaign_repository.list_by_user(user_id=user_id)
        return [
            await self._reconcile_read_status(user_id=user_id, campaign=campaign)
            for campaign in campaigns
        ]

    async def get_campaign(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        campaign = await self._campaign_repository.get(
            user_id=user_id, campaign_id=campaign_id
        )
        return await self._reconcile_read_status(user_id=user_id, campaign=campaign)

    async def _reconcile_read_status(
        self, *, user_id: str, campaign: CampaignStatus
    ) -> CampaignStatus:
        """Repair a stale evaluating projection from the durable metric ledger.

        Worker callbacks are intentionally best-effort around event-loop
        shutdowns. A read must still converge once every durable RAGAS item is
        terminal, while unresolved work remains visibly evaluating.
        """
        if (
            campaign.status == CampaignLifecycleStatus.EVALUATING
            and campaign.evaluation_total_units > 0
        ):
            return await self._campaign_repository.derive_ragas_state(
                user_id=user_id,
                campaign_id=campaign.id,
            )
        return campaign

    async def get_results(
        self, *, user_id: str, campaign_id: str
    ) -> CampaignResultsResponse:
        campaign = await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        return CampaignResultsResponse(campaign=campaign, results=results)

    async def get_metrics(
        self, *, user_id: str, campaign_id: str
    ) -> CampaignMetricsResponse:
        campaign = await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        return await self._ragas_evaluator.get_metrics(
            user_id=user_id, campaign=campaign
        )

    async def list_traces(
        self, *, user_id: str, campaign_id: str
    ) -> list[AgentTraceSummary]:
        await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        return await self._trace_repository.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )

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

    async def cancel_campaign(
        self, *, user_id: str, campaign_id: str
    ) -> CampaignStatus:
        campaign = await self._campaign_repository.get(
            user_id=user_id, campaign_id=campaign_id
        )
        if campaign.status in _TERMINAL_STATUSES:
            return campaign

        await self._campaign_repository.request_cancel(
            user_id=user_id, campaign_id=campaign_id
        )
        await self._job_store.cancel_campaign_jobs(
            user_id=user_id, campaign_id=campaign_id
        )
        return await self._campaign_repository.mark_cancelled(
            user_id=user_id, campaign_id=campaign_id
        )

    async def create_rerun(
        self,
        *,
        user_id: str,
        campaign_id: str,
        request: EvaluationRerunRequest,
    ) -> EvaluationJob:
        """Create one durable rerun job from immutable campaign work.

        Execution reruns reuse the original work snapshots, while metric-only
        reruns target the campaign's current successful official results.  The
        worker creates downstream RAGAS work after an execution rerun promotes
        a result, so a combined rerun never evaluates an uncommitted payload.
        """
        campaign = await self._campaign_repository.get(
            user_id=user_id, campaign_id=campaign_id
        )
        if campaign.status in {
            CampaignLifecycleStatus.RUNNING,
            CampaignLifecycleStatus.EVALUATING,
        }:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="Campaign is already running",
                status_code=400,
            )

        configured_metrics = getattr(self._ragas_evaluator, "enabled_metrics", None)
        if configured_metrics is None:
            configured_metrics = (
                [_LEGACY_RAGAS_METRIC]
                if callable(getattr(self._ragas_evaluator, "evaluate_campaign", None))
                else ["faithfulness", "answer_correctness", "answer_relevancy"]
            )
        enabled_metrics = list(configured_metrics)
        if request.stages != "execution":
            unknown_metrics = [
                name for name in request.metric_names if name not in enabled_metrics
            ]
            if unknown_metrics:
                raise AppError(
                    code=ErrorCode.BAD_REQUEST,
                    message=f"Unknown RAGAS metrics: {', '.join(unknown_metrics)}",
                    status_code=400,
                )

        includes_execution = request.stages in {"execution", "execution_and_ragas"}
        if includes_execution:
            rows = await self._job_store.list_campaign_work_items(
                user_id=user_id,
                campaign_id=campaign_id,
                work_type=EvaluationWorkType.DATASET_EXECUTION,
            )
            selected_rows = self._select_rerun_work_rows(
                rows, request=request, kind="execution"
            )
            if not selected_rows:
                raise AppError(
                    code=ErrorCode.BAD_REQUEST,
                    message="No matching execution work is available for rerun",
                    status_code=400,
                )
            specs = [
                WorkItemSpec(
                    work_item_id=str(row["work_item_id"]),
                    work_type=EvaluationWorkType.DATASET_EXECUTION,
                    logical_key=str(row["logical_key"]),
                    input_snapshot=dict(row["input_snapshot"]),
                )
                for row in selected_rows
            ]
            downstream_question_ids = sorted(
                {
                    str(row["input_snapshot"].get("test_case", {}).get("id"))
                    for row in selected_rows
                    if row["input_snapshot"].get("test_case", {}).get("id")
                }
            )
            job = await self._job_store.create_job_with_items(
                user_id=user_id,
                campaign_id=campaign_id,
                job_type=EvaluationJobType.RERUN,
                selection=request.model_dump(mode="json"),
                config_snapshot={
                    "campaign_config": campaign.config.model_dump(
                        mode="json", by_alias=True
                    ),
                    "stages": request.stages,
                    "skip_ragas": request.stages == "execution",
                    "metric_names": list(request.metric_names),
                    "downstream_question_ids": downstream_question_ids,
                },
                items=specs,
            )
            await self._campaign_repository.mark_running(
                user_id=user_id,
                campaign_id=campaign_id,
            )
            await self._start_worker_if_available()
            if self._worker_notifier is not None:
                self._worker_notifier()
            return job

        metric_names = (
            list(request.metric_names) if request.metric_names else enabled_metrics
        )
        unknown_metrics = [name for name in metric_names if name not in enabled_metrics]
        if unknown_metrics:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message=f"Unknown RAGAS metrics: {', '.join(unknown_metrics)}",
                status_code=400,
            )

        # Legacy campaigns created before the durable ledger have no attempt
        # provenance.  Backfill deterministic synthetic attempts so the
        # compatibility endpoint can still enqueue a metric-only rerun.
        await self._job_store.backfill_legacy_attempts()
        results = await self._result_repository.list_for_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
        )
        completed_results = [
            row
            for row in results
            if row.status == CampaignResultStatus.COMPLETED
            and row.source_attempt_id is not None
        ]
        metric_names_by_result: dict[str, list[str]] | None = None
        if request.scope == "selected":
            question_ids = set(request.question_ids)
            completed_results = [
                row for row in completed_results if row.question_id in question_ids
            ]
        elif request.scope == "failed_only":
            failed_rows = await self._job_store.list_campaign_work_items(
                user_id=user_id,
                campaign_id=campaign_id,
                work_type=EvaluationWorkType.RAGAS_METRIC,
            )
            failed_keys = {
                (
                    str(row["input_snapshot"].get("campaign_result_id") or ""),
                    str(row["input_snapshot"].get("metric_name") or ""),
                )
                for row in failed_rows
                if row["status"] in {"failed", "interrupted"}
            }
            completed_results = [
                row
                for row in completed_results
                if any(
                    row.id == result_id and metric in metric_names
                    for result_id, metric in failed_keys
                )
            ]
            failed_metric_names = {metric for _, metric in failed_keys}
            metric_names = [
                metric for metric in metric_names if metric in failed_metric_names
            ]
            metric_names_by_result = {}
            for result_id, metric in failed_keys:
                if metric in metric_names:
                    metric_names_by_result.setdefault(result_id, []).append(metric)

        if not completed_results or not metric_names:
            message = (
                "Requested question_ids have no completed raw results in this campaign"
                if request.scope == "selected"
                else "No matching completed results are available for RAGAS rerun"
            )
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message=message,
                status_code=400,
            )

        existing_jobs = {
            job.job_id
            for job in await self._job_store.list_jobs(
                user_id=user_id, campaign_id=campaign_id
            )
        }
        selected_result_ids = [row.id for row in completed_results]
        created_count = await self._job_store.ensure_ragas_work(
            user_id=user_id,
            campaign_id=campaign_id,
            evaluator_model=str(getattr(self._ragas_evaluator, "evaluator_model", "")),
            evaluator_config={},
            enabled_metrics=metric_names,
            selected_result_ids=selected_result_ids,
            **(
                {"metric_names_by_result": metric_names_by_result}
                if metric_names_by_result is not None
                else {}
            ),
            force=True,
            ragas_batch_size=campaign.config.ragas_batch_size,
            ragas_parallel_batches=campaign.config.ragas_parallel_batches,
        )
        if not created_count:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="No RAGAS work was created for rerun",
                status_code=400,
            )
        await self._start_worker_if_available()
        if self._worker_notifier is not None:
            self._worker_notifier()
        jobs = await self._job_store.list_jobs(user_id=user_id, campaign_id=campaign_id)
        new_jobs = [job for job in jobs if job.job_id not in existing_jobs]
        if not new_jobs:
            raise AppError(
                code=ErrorCode.INTERNAL_ERROR,
                message="RAGAS rerun job was not persisted",
                status_code=500,
            )
        return max(new_jobs, key=lambda job: job.created_at)

    @staticmethod
    def _select_rerun_work_rows(
        rows: list[dict[str, Any]],
        *,
        request: EvaluationRerunRequest,
        kind: str,
    ) -> list[dict[str, Any]]:
        if request.scope == "all":
            return rows
        if request.scope == "failed_only":
            return [row for row in rows if row["status"] in {"failed", "interrupted"}]
        selected_ids = set(request.question_ids)
        selected: list[dict[str, Any]] = []
        for row in rows:
            snapshot = row["input_snapshot"]
            if kind == "execution":
                question_id = snapshot.get("test_case", {}).get("id")
            else:
                question_id = snapshot.get("result", {}).get("question_id")
            if question_id in selected_ids:
                selected.append(row)
        return selected

    async def list_jobs(self, *, user_id: str, campaign_id: str) -> list[EvaluationJob]:
        await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)
        return await self._job_store.list_jobs(user_id=user_id, campaign_id=campaign_id)

    async def get_job(self, *, user_id: str, job_id: str) -> EvaluationJob:
        return await self._job_store.get_job(user_id=user_id, job_id=job_id)

    async def list_job_items(
        self, *, user_id: str, job_id: str
    ) -> list[EvaluationJobItemSummary]:
        return await self._job_store.list_job_items(user_id=user_id, job_id=job_id)

    async def cancel_job(self, *, user_id: str, job_id: str) -> EvaluationJob:
        job = await self._job_store.get_job(user_id=user_id, job_id=job_id)
        campaign = None
        if job.campaign_id:
            campaign = await self._campaign_repository.get(
                user_id=user_id, campaign_id=job.campaign_id
            )
        work_types = await self._job_store.get_job_work_types(
            user_id=user_id, job_id=job_id
        )
        cancelled = await self._job_store.cancel_job(user_id=user_id, job_id=job_id)
        if campaign is not None:
            if (
                EvaluationWorkType.DATASET_EXECUTION in work_types
                or campaign.status == CampaignLifecycleStatus.RUNNING
            ):
                await self._campaign_repository.derive_execution_state(
                    user_id=user_id, campaign_id=campaign.id
                )
            if (
                EvaluationWorkType.RAGAS_METRIC in work_types
                or campaign.status == CampaignLifecycleStatus.EVALUATING
            ):
                await self._campaign_repository.derive_ragas_state(
                    user_id=user_id, campaign_id=campaign.id
                )
        return cancelled

    async def list_attempts(
        self, *, user_id: str, work_item_id: str
    ) -> list[EvaluationAttempt]:
        return await self._job_store.list_attempts(
            user_id=user_id, work_item_id=work_item_id
        )

    async def evaluate_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        question_ids: Optional[list[str]] = None,
    ) -> CampaignStatus:
        request = EvaluationRerunRequest(
            scope="selected" if question_ids else "all",
            stages="ragas",
            question_ids=list(question_ids or []),
        )
        await self.create_rerun(
            user_id=user_id,
            campaign_id=campaign_id,
            request=request,
        )
        campaign = await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        if campaign.status in _TERMINAL_STATUSES or request.question_ids:
            # The local worker may finish a tiny legacy-compatible rerun
            # before the HTTP handler serializes its response.  Preserve the
            # historical contract that POST /evaluate acknowledges the new
            # evaluation phase; subsequent polling reads the durable terminal
            # state from the database.
            results = await self._result_repository.list_for_campaign(
                user_id=user_id, campaign_id=campaign_id
            )
            selected_count = (
                len(
                    [
                        row
                        for row in results
                        if row.status == CampaignResultStatus.COMPLETED
                        and (
                            not request.question_ids
                            or row.question_id in request.question_ids
                        )
                    ]
                )
                if request.question_ids
                else campaign.evaluation_total_units
            )
            campaign = campaign.model_copy(
                update={
                    "status": CampaignLifecycleStatus.EVALUATING,
                    "phase": "evaluation",
                    "evaluation_completed_units": 0,
                    "evaluation_total_units": selected_count,
                }
            )
        return campaign

    async def recover_inflight_campaigns(self) -> None:
        """Recover non-terminal campaigns after process restart."""
        inflight = await self._campaign_repository.list_inflight()
        if not inflight:
            return

        drain_owned = (
            self._worker_owned
            and self._worker is not None
            and not self._worker.is_running
        )
        for user_id, campaign in inflight:
            try:
                await self._prepare_legacy_recovery(
                    user_id=user_id,
                    campaign=campaign,
                )
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
                if isinstance(exc, AppError):
                    await self._campaign_repository.mark_failed(
                        user_id=user_id,
                        campaign_id=campaign.id,
                        error_message=str(exc),
                        phase="execution",
                    )
        if drain_owned and self._worker is not None:
            await self._worker.run_until_idle()

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

        self._worker_notifier()
        return campaign

    async def _start_worker_if_available(self) -> None:
        if self._worker is not None:
            await self._worker.start()

    async def _prepare_legacy_recovery(
        self, *, user_id: str, campaign: CampaignStatus
    ) -> None:
        """Bridge pre-ledger campaigns into the durable recovery path."""
        if campaign.cancel_requested:
            await self._campaign_repository.mark_cancelled(
                user_id=user_id, campaign_id=campaign.id
            )
            return

        durable_items = await self._job_store.list_campaign_work_items(
            user_id=user_id, campaign_id=campaign.id
        )
        if durable_items:
            # A normal ledger-backed campaign is recovered by the process
            # worker itself.  The compatibility bridge below is only for
            # campaigns that predate the ledger entirely.
            return

        await self._job_store.backfill_legacy_attempts()
        configured_metrics = getattr(self._ragas_evaluator, "enabled_metrics", None)
        if configured_metrics is None:
            configured_metrics = (
                [_LEGACY_RAGAS_METRIC]
                if callable(getattr(self._ragas_evaluator, "evaluate_campaign", None))
                else ["faithfulness", "answer_correctness", "answer_relevancy"]
            )
        metric_names = list(configured_metrics)
        if campaign.status == CampaignLifecycleStatus.EVALUATING:
            results = await self._result_repository.list_for_campaign(
                user_id=user_id, campaign_id=campaign.id
            )
            selected_ids = [
                row.id
                for row in results
                if row.status == CampaignResultStatus.COMPLETED
            ]
            created = await self._job_store.ensure_ragas_work(
                user_id=user_id,
                campaign_id=campaign.id,
                evaluator_model=str(
                    getattr(self._ragas_evaluator, "evaluator_model", "")
                ),
                evaluator_config={},
                enabled_metrics=metric_names,
                selected_result_ids=selected_ids,
                ragas_batch_size=campaign.config.ragas_batch_size,
                ragas_parallel_batches=campaign.config.ragas_parallel_batches,
            )
            if not created:
                await self._campaign_repository.mark_completed(
                    user_id=user_id,
                    campaign_id=campaign.id,
                    phase="evaluation",
                    completed_units=len(selected_ids),
                )
            return

        existing = await self._job_store.list_campaign_work_items(
            user_id=user_id,
            campaign_id=campaign.id,
            work_type=EvaluationWorkType.DATASET_EXECUTION,
        )
        test_cases = await self._resolve_test_cases(
            user_id=user_id, test_case_ids=campaign.config.test_case_ids
        )
        units = self._build_units(
            test_cases=test_cases,
            modes=campaign.config.modes,
            repeat_count=campaign.config.repeat_count,
            ablation_conditions=campaign.config.ablation_conditions,
            agentic_execution_version=campaign.config.agentic_execution_version,
            shadow_evaluation_policy=campaign.config.shadow_evaluation_policy,
        )
        existing_keys = {str(row.get("logical_key")) for row in existing}
        result_by_id = {
            row.id: row
            for row in await self._result_repository.list_for_campaign(
                user_id=user_id, campaign_id=campaign.id
            )
        }
        for row in existing:
            logical_key = str(row.get("logical_key") or "")
            if not logical_key.startswith("legacy:execution:"):
                continue
            legacy_result = result_by_id.get(logical_key.rsplit(":", 1)[-1])
            if legacy_result is not None:
                existing_keys.add(
                    f"execution:{legacy_result.question_id}:{legacy_result.mode}:"
                    f"{legacy_result.run_number}:none"
                )
        missing_specs = [
            spec
            for unit in units
            for spec in [
                self._work_item_spec(
                    user_id=user_id,
                    campaign_id=campaign.id,
                    unit=unit,
                    config=campaign.config,
                )
            ]
            if spec.logical_key not in existing_keys
        ]
        if missing_specs:
            await self._job_store.create_job_with_items(
                user_id=user_id,
                campaign_id=campaign.id,
                job_type=EvaluationJobType.INITIAL,
                selection={"campaign_id": campaign.id},
                config_snapshot=campaign.config.model_dump(mode="json", by_alias=True),
                items=missing_specs,
            )

    async def _resolve_test_cases(
        self, *, user_id: str, test_case_ids: list[str]
    ) -> list[TestCase]:
        available = [
            TestCase.model_validate(item) for item in await list_test_cases(user_id)
        ]
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
        ablation_conditions: list[AblationCondition] | None = None,
        agentic_execution_version: Literal["v8", "v9", "v10"] = "v8",
        shadow_evaluation_policy: Literal["operational", "research"] | None = None,
    ) -> list[CampaignUnit]:
        units: list[CampaignUnit] = []
        if ablation_conditions:
            condition_count = len(ablation_conditions)
            for repeat_number in range(1, repeat_count + 1):
                for test_case in test_cases:
                    for condition_index, condition in enumerate(
                        ablation_conditions, start=1
                    ):
                        stored_run_number = (
                            (repeat_number - 1) * condition_count
                        ) + condition_index
                        units.append(
                            CampaignUnit(
                                test_case=test_case,
                                mode=condition.mode,
                                run_number=stored_run_number,
                                repeat_number=repeat_number,
                                condition_id=condition.condition_id,
                                condition_label=condition.label,
                                ablation_flags=dict(condition.ablation_flags),
                                budget=dict(condition.budget)
                                if condition.budget
                                else None,
                                agentic_execution_version=effective_agentic_execution_version(
                                    condition.mode, agentic_execution_version
                                ),
                                shadow_evaluation_policy=shadow_evaluation_policy,
                            )
                        )
            return units

        for run_number in range(1, repeat_count + 1):
            for test_case in test_cases:
                for mode in modes:
                    units.append(
                        CampaignUnit(
                            test_case=test_case,
                            mode=mode,
                            run_number=run_number,
                            repeat_number=run_number,
                            agentic_execution_version=effective_agentic_execution_version(
                                mode, agentic_execution_version
                            ),
                            shadow_evaluation_policy=shadow_evaluation_policy,
                        )
                    )
        return units

    @staticmethod
    def _work_item_spec(
        *, user_id: str, campaign_id: str, unit: CampaignUnit, config: CampaignConfig
    ) -> WorkItemSpec:
        condition_key = unit.condition_id or "none"
        return WorkItemSpec(
            work_type=EvaluationWorkType.DATASET_EXECUTION,
            logical_key=f"execution:{unit.test_case.id}:{unit.mode}:{unit.run_number}:{condition_key}",
            input_snapshot={
                "user_id": user_id,
                "campaign_id": campaign_id,
                "test_case": unit.test_case.model_dump(mode="json"),
                "mode": unit.mode,
                "run_number": unit.run_number,
                "repeat_number": unit.repeat_number,
                "condition_id": unit.condition_id,
                "condition_label": unit.condition_label,
                "ablation_flags": dict(unit.ablation_flags or {}),
                "budget": dict(unit.budget or {}),
                "agentic_execution_version": unit.agentic_execution_version,
                "shadow_evaluation_policy": unit.shadow_evaluation_policy,
                "model_config": config.model_preset.model_dump(mode="json"),
                "prompt_capture_policy": config.prompt_capture_policy.model_dump(
                    mode="json"
                ),
            },
        )


_campaign_engine: Optional[CampaignEngine] = None


def get_campaign_engine() -> CampaignEngine:
    global _campaign_engine
    if _campaign_engine is None:
        from evaluation.job_worker import get_evaluation_job_worker

        _campaign_engine = CampaignEngine(
            configure_worker=not get_evaluation_job_worker().is_configured
        )
    return _campaign_engine
