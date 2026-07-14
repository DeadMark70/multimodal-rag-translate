"""Single-process lifecycle manager for durable evaluation work."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TypeAlias

from evaluation.job_schemas import ClaimedEvaluationWork, EvaluationWorkType
from evaluation.job_store import EvaluationJobStore


ClaimHandler: TypeAlias = Callable[[ClaimedEvaluationWork], Awaitable[None]]
Clock: TypeAlias = Callable[[], datetime]
Sleep: TypeAlias = Callable[[float], Awaitable[None]]

_EXECUTION_CONCURRENCY = 4
_RAGAS_CONCURRENCY = 2
_HEARTBEAT_SECONDS = 15.0


class EvaluationJobWorker:
    """Run the durable ledger locally, with explicit wakeup and shutdown control."""

    def __init__(
        self,
        *,
        store: EvaluationJobStore | None = None,
        execution_handler: ClaimHandler | None = None,
        ragas_handler: ClaimHandler | None = None,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._store = store or EvaluationJobStore()
        self._execution_handler = execution_handler
        self._ragas_handler = ragas_handler
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or asyncio.sleep
        self._execution_slots = asyncio.Semaphore(_EXECUTION_CONCURRENCY)
        self._ragas_slots = asyncio.Semaphore(_RAGAS_CONCURRENCY)
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._claim_lock = asyncio.Lock()
        self._loop_task: asyncio.Task[None] | None = None
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._accepting = False

    async def start(self) -> None:
        """Recover interrupted ledger attempts and start the event-driven loop."""
        if self._loop_task is not None and not self._loop_task.done():
            return
        self._stop_event.clear()
        self._accepting = True
        async with self._claim_lock:
            await self._store.recover_interrupted_attempts(at=self._clock())
        self._loop_task = asyncio.create_task(self._run_loop(), name="evaluation-job-worker")
        self.notify()

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
            await loop_task
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
            available = _EXECUTION_CONCURRENCY + _RAGAS_CONCURRENCY - len(self._active_tasks)
            if available <= 0:
                return 0
            claims = await self._store.claim_ready_items(limit=available, now=self._clock())
            if not self._accepting:
                return 0
        for claim in claims:
            task = asyncio.create_task(self._run_claim(claim), name=f"evaluation-attempt-{claim.attempt_id}")
            self._active_tasks.add(task)
            task.add_done_callback(self._task_finished)
        return len(claims)

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.clear()
            claimed = await self.run_once()
            if self._stop_event.is_set():
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

    async def _heartbeat_until_cancelled(self, attempt_id: str) -> None:
        while not self._stop_event.is_set():
            await self._sleep(_HEARTBEAT_SECONDS)
            if not self._stop_event.is_set():
                await self._store.heartbeat_attempt(attempt_id, at=self._clock())

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        self._active_tasks.discard(task)
        if not task.cancelled():
            task.exception()
        self.notify()

    def _handlers_unavailable(self) -> bool:
        return self._execution_handler is None and self._ragas_handler is None

    def _handler_for(self, claim: ClaimedEvaluationWork) -> tuple[ClaimHandler, asyncio.Semaphore]:
        if claim.work_type == EvaluationWorkType.RAGAS_METRIC or (
            claim.work_type is None and (claim.logical_key or "").startswith("ragas:")
        ):
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
