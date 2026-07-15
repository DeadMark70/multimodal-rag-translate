"""Single-process lifecycle manager for durable evaluation work."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TypeAlias

from evaluation.job_schemas import ClaimedEvaluationWork, EvaluationWorkType
from evaluation.job_store import EvaluationJobStore


ClaimHandler: TypeAlias = Callable[[ClaimedEvaluationWork], Awaitable[None]]
RagasBatchHandler: TypeAlias = Callable[[list[ClaimedEvaluationWork]], Awaitable[None]]
Clock: TypeAlias = Callable[[], datetime]
Sleep: TypeAlias = Callable[[float], Awaitable[None]]

_EXECUTION_CONCURRENCY = 4
_RAGAS_CONCURRENCY = 2
_RAGAS_BATCH_SIZE = 4
_HEARTBEAT_SECONDS = 15.0


class EvaluationJobWorker:
    """Run the durable ledger locally, with explicit wakeup and shutdown control."""

    def __init__(
        self,
        *,
        store: EvaluationJobStore | None = None,
        execution_handler: ClaimHandler | None = None,
        ragas_handler: ClaimHandler | None = None,
        ragas_batch_handler: RagasBatchHandler | None = None,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
        stop_when_idle: bool = False,
    ) -> None:
        self._store = store or EvaluationJobStore(on_job_created=self.notify)
        self._execution_handler = execution_handler
        self._ragas_handler = ragas_handler
        self._ragas_batch_handler = ragas_batch_handler
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or asyncio.sleep
        self._stop_when_idle = stop_when_idle
        self._reset_loop_primitives()
        self._loop_task: asyncio.Task[None] | None = None
        self._active_tasks: dict[asyncio.Task[None], EvaluationWorkType] = {}
        self._accepting = False

    def _reset_loop_primitives(self) -> None:
        """Create asyncio synchronization objects for the current event loop.

        The process worker is a singleton, while FastAPI's TestClient (and
        some embedded hosts) may start it from a new event loop after a prior
        lifespan has stopped.  asyncio locks, semaphores, and events are
        loop-bound once awaited, so they must not leak across lifespans.
        """
        self._execution_slots = asyncio.Semaphore(_EXECUTION_CONCURRENCY)
        self._ragas_slots = asyncio.Semaphore(_RAGAS_CONCURRENCY)
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._claim_lock = asyncio.Lock()

    @property
    def is_configured(self) -> bool:
        """Whether at least one durable work handler is ready to run."""
        return not self._handlers_unavailable()

    @property
    def is_running(self) -> bool:
        """Whether the event-driven loop is currently accepting work."""
        return self._loop_task is not None and not self._loop_task.done()

    def configure_handlers(
        self,
        *,
        execution_handler: ClaimHandler | None = None,
        ragas_handler: ClaimHandler | None = None,
        ragas_batch_handler: RagasBatchHandler | None = None,
    ) -> None:
        """Attach Task 5/6 handlers before the process worker starts."""
        if self._accepting:
            raise RuntimeError("Evaluation worker handlers cannot change while it is running")
        if execution_handler is not None:
            self._execution_handler = execution_handler
        if ragas_handler is not None:
            self._ragas_handler = ragas_handler
        if ragas_batch_handler is not None:
            self._ragas_batch_handler = ragas_batch_handler

    async def start(self) -> None:
        """Recover interrupted ledger attempts and start the event-driven loop."""
        if self._loop_task is not None and not self._loop_task.done():
            return
        if self._handlers_unavailable():
            raise RuntimeError("Evaluation worker handlers must be configured before start")
        self._reset_loop_primitives()
        self._active_tasks.clear()
        self._accepting = True
        async with self._claim_lock:
            await self._store.recover_interrupted_attempts(at=self._clock())
        self._loop_task = asyncio.create_task(self._run_loop(), name="evaluation-job-worker")
        self.notify()

    async def run_until_idle(self, *, max_rounds: int = 1000) -> None:
        """Drain ready durable work synchronously for embedded recovery.

        This is intentionally separate from ``start``: compatibility callers
        that do not own an application lifespan can recover a campaign without
        leaving a background loop task attached to their short-lived event
        loop.  Production uses ``start`` and the event-driven loop instead.
        """
        if self.is_running:
            return
        if self._handlers_unavailable():
            raise RuntimeError("Evaluation worker handlers must be configured before drain")
        self._reset_loop_primitives()
        self._accepting = True
        async with self._claim_lock:
            await self._store.recover_interrupted_attempts(at=self._clock())
        try:
            for _ in range(max_rounds):
                claimed = await self.run_once()
                active = tuple(self._active_tasks)
                if active:
                    await asyncio.gather(*active, return_exceptions=True)
                if not claimed and not self._active_tasks:
                    break
        finally:
            self._accepting = False

    async def stop(self) -> None:
        """Stop accepting claims, cancel handlers, and recover unfinished attempts."""
        self._accepting = False
        self._stop_event.set()
        self.notify()

        active = tuple(self._active_tasks)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)

        async with self._claim_lock:
            await self._store.recover_interrupted_attempts(at=self._clock())

        loop_task = self._loop_task
        if loop_task is not None:
            try:
                await asyncio.wait_for(loop_task, timeout=5.0)
            except asyncio.TimeoutError:
                loop_task.cancel()
                await asyncio.gather(loop_task, return_exceptions=True)
        self._loop_task = None

    def notify(self) -> None:
        """Wake the loop after callers add work or change its readiness."""
        self._wake_event.set()

    async def run_once(self) -> int:
        """Claim and dispatch currently-ready work without exceeding local capacity."""
        if not self._accepting or self._handlers_unavailable():
            return 0
        async with self._claim_lock:
            if not self._accepting:
                return 0
            claims: list[ClaimedEvaluationWork] = []
            for work_type, capacity, handler in (
                (
                    EvaluationWorkType.DATASET_EXECUTION,
                    _EXECUTION_CONCURRENCY,
                    self._execution_handler,
                ),
                (
                    EvaluationWorkType.RAGAS_METRIC,
                    _RAGAS_CONCURRENCY,
                    self._ragas_handler,
                ),
            ):
                if handler is None and not (
                    work_type == EvaluationWorkType.RAGAS_METRIC
                    and self._ragas_batch_handler is not None
                ):
                    continue
                available = capacity - self._active_count(work_type)
                if available > 0:
                    if (
                        work_type == EvaluationWorkType.RAGAS_METRIC
                        and self._ragas_batch_handler is not None
                        and self._ragas_handler is None
                    ):
                        # One batch-handler task owns two provider batches at a
                        # time; claim up to the durable worker's batch window.
                        if self._active_count(work_type) > 0:
                            continue
                        claims.extend(
                            await self._store.claim_ready_items(
                                limit=_RAGAS_BATCH_SIZE * _RAGAS_CONCURRENCY,
                                now=self._clock(),
                                work_type=work_type,
                            )
                        )
                        continue
                    claims.extend(
                        await self._store.claim_ready_items(
                            limit=available,
                            now=self._clock(),
                            work_type=work_type,
                        )
                    )
            if not self._accepting:
                return 0
        if self._ragas_batch_handler is not None and self._ragas_handler is None:
            ragas_claims = [claim for claim in claims if self._work_type_for(claim) == EvaluationWorkType.RAGAS_METRIC]
            other_claims = [claim for claim in claims if claim not in ragas_claims]
            dispatch: list[tuple[asyncio.Task[None], EvaluationWorkType]] = []
            if ragas_claims:
                dispatch.append(
                    (
                        asyncio.create_task(
                            self._run_ragas_batch(ragas_claims),
                            name="evaluation-ragas-batch",
                        ),
                        EvaluationWorkType.RAGAS_METRIC,
                    )
                )
            dispatch.extend(
                (
                    asyncio.create_task(self._run_claim(claim), name=f"evaluation-attempt-{claim.attempt_id}"),
                    self._work_type_for(claim),
                )
                for claim in other_claims
            )
        else:
            dispatch = [
                (
                    asyncio.create_task(self._run_claim(claim), name=f"evaluation-attempt-{claim.attempt_id}"),
                    self._work_type_for(claim),
                )
                for claim in claims
            ]
        for task, work_type in dispatch:
            self._active_tasks[task] = work_type
            task.add_done_callback(self._task_finished)
        return len(claims)

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.clear()
            claimed = await self.run_once()
            if self._stop_event.is_set():
                return
            if not claimed and self._stop_when_idle and not self._active_tasks:
                self._accepting = False
                self._stop_event.set()
                return
            if claimed:
                await asyncio.sleep(0)
                continue
            await self._wait_for_wakeup_or_retry()

    async def _wait_for_wakeup_or_retry(self) -> None:
        due_at = await self._store.next_ready_at()
        timeout = None if due_at is None else max(0.0, (due_at - self._clock()).total_seconds())
        wake_task = asyncio.create_task(self._wake_event.wait())
        waiters: set[asyncio.Task[object]] = {wake_task}
        if timeout is not None:
            waiters.add(asyncio.create_task(self._sleep(timeout)))
        done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        del done
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _run_claim(self, claim: ClaimedEvaluationWork) -> None:
        handler, slots = self._handler_for(claim)
        heartbeat = asyncio.create_task(
            self._heartbeat_until_cancelled(claim.attempt_id),
            name=f"evaluation-heartbeat-{claim.attempt_id}",
        )
        try:
            async with slots:
                await handler(claim)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _run_ragas_batch(self, claims: list[ClaimedEvaluationWork]) -> None:
        assert self._ragas_batch_handler is not None
        heartbeats = [
            asyncio.create_task(
                self._heartbeat_until_cancelled(claim.attempt_id),
                name=f"evaluation-heartbeat-{claim.attempt_id}",
            )
            for claim in claims
        ]
        try:
            await self._ragas_batch_handler(claims)
        finally:
            for heartbeat in heartbeats:
                heartbeat.cancel()
            if heartbeats:
                await asyncio.gather(*heartbeats, return_exceptions=True)

    async def _heartbeat_until_cancelled(self, attempt_id: str) -> None:
        while not self._stop_event.is_set():
            await self._sleep(_HEARTBEAT_SECONDS)
            if not self._stop_event.is_set():
                await self._store.heartbeat_attempt(attempt_id, at=self._clock())

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        self._active_tasks.pop(task, None)
        if not task.cancelled():
            task.exception()
        self.notify()

    def _handlers_unavailable(self) -> bool:
        return (
            self._execution_handler is None
            and self._ragas_handler is None
            and self._ragas_batch_handler is None
        )

    def _active_count(self, work_type: EvaluationWorkType) -> int:
        return sum(active_type == work_type for active_type in self._active_tasks.values())

    def _work_type_for(self, claim: ClaimedEvaluationWork) -> EvaluationWorkType:
        if claim.work_type == EvaluationWorkType.RAGAS_METRIC or (
            claim.work_type is None and (claim.logical_key or "").startswith("ragas:")
        ):
            return EvaluationWorkType.RAGAS_METRIC
        return EvaluationWorkType.DATASET_EXECUTION

    def _handler_for(self, claim: ClaimedEvaluationWork) -> tuple[ClaimHandler, asyncio.Semaphore]:
        if self._work_type_for(claim) == EvaluationWorkType.RAGAS_METRIC:
            if self._ragas_handler is None:
                raise RuntimeError("RAGAS handler is not configured")
            return self._ragas_handler, self._ragas_slots
        if self._execution_handler is None:
            raise RuntimeError("Execution handler is not configured")
        return self._execution_handler, self._execution_slots


_worker: EvaluationJobWorker | None = None


def get_evaluation_job_worker() -> EvaluationJobWorker:
    """Return the process-local durable worker singleton."""
    global _worker
    if _worker is None:
        _worker = EvaluationJobWorker()
    return _worker


def configure_evaluation_job_worker(
    *,
    execution_handler: ClaimHandler | None = None,
    ragas_handler: ClaimHandler | None = None,
    ragas_batch_handler: RagasBatchHandler | None = None,
) -> EvaluationJobWorker:
    """Configure the process singleton when Task 5/6 adapters become available."""
    worker = get_evaluation_job_worker()
    worker.configure_handlers(
        execution_handler=execution_handler,
        ragas_handler=ragas_handler,
        ragas_batch_handler=ragas_batch_handler,
    )
    return worker
