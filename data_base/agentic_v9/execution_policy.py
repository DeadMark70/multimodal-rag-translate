"""Runtime-only concurrency, timeout, and cancellation bounds for Agentic v9."""

from __future__ import annotations

import asyncio
from time import monotonic
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, TypeVar

from data_base.agentic_v9.schemas import ExecutionPolicy


_T = TypeVar("_T")
RuntimeKind = Literal["retrieval", "llm", "visual"]
_DEFAULT_TIMEOUT_S = 8.0


class ExecutionDeadline:
    """A monotonic whole-run deadline created once per evaluation attempt."""

    def __init__(
        self, total_deadline_s: float, *, monotonic: Callable[[], float] = monotonic
    ) -> None:
        if total_deadline_s <= 0:
            raise ValueError("total deadline must be positive")
        self._monotonic = monotonic
        self.started_at = monotonic()
        self.total_deadline_s = total_deadline_s

    def remaining_seconds(self) -> float:
        """Return the non-negative budget left from the original start time."""
        return max(0.0, self.total_deadline_s - (self._monotonic() - self.started_at))

    def has_time_remaining(self) -> bool:
        """Return whether another bounded operation may still be started."""
        return self.remaining_seconds() > 0


class ExecutionCancellation:
    """One cooperative cancellation signal shared by a v9 execution attempt."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str | None = None

    def cancel(self, reason: str) -> None:
        """Signal cancellation once while retaining its first terminal reason."""
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether the shared execution was cancelled."""
        return self._event.is_set()

    def is_set(self) -> bool:
        """Support asyncio.Event-compatible cancellation checks."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Wait until a campaign, task, or stream cancellation arrives."""
        await self._event.wait()


class V9ExecutionPolicyRuntime:
    """Apply typed v9 policy without changing the legacy v8 execution path."""

    def __init__(self, policy: ExecutionPolicy | None = None) -> None:
        self.policy = policy or ExecutionPolicy()
        self._semaphores = {
            "retrieval": asyncio.Semaphore(self.policy.max_retrieval_concurrency),
            "llm": asyncio.Semaphore(self.policy.max_llm_concurrency),
            "visual": asyncio.Semaphore(self.policy.max_visual_concurrency),
        }
        if any(timeout <= 0 for timeout in self.policy.phase_timeouts_s.values()):
            raise ValueError("phase timeouts must be positive")

    def start_deadline(self) -> ExecutionDeadline:
        """Create the attempt-wide deadline before any v9 stage is entered."""
        return ExecutionDeadline(self.policy.total_deadline_s)

    def timeout_for(
        self, phase: str, *, deadline: ExecutionDeadline | None = None
    ) -> float:
        """Return the phase bound clamped to the remaining whole-run budget."""
        timeout = self.policy.phase_timeouts_s.get(phase, _DEFAULT_TIMEOUT_S)
        return min(timeout, deadline.remaining_seconds()) if deadline else timeout

    def has_final_reserve(self, deadline: ExecutionDeadline) -> bool:
        """Keep the final phase's declared time available for required completion."""
        return deadline.remaining_seconds() >= self.timeout_for("final_answer")

    async def run_retrieval(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        phase: str = "evidence_extract",
        cancellation: ExecutionCancellation | None = None,
        deadline: ExecutionDeadline | None = None,
    ) -> _T:
        """Run one retrieval operation within its independent semaphore."""
        return await self._run("retrieval", operation, phase, cancellation, deadline)

    async def run_llm(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        phase: str,
        cancellation: ExecutionCancellation | None = None,
        deadline: ExecutionDeadline | None = None,
    ) -> _T:
        """Run one budget-gated LLM operation within the shared LLM cap."""
        return await self._run("llm", operation, phase, cancellation, deadline)

    async def run_visual(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        phase: str = "visual_extract",
        cancellation: ExecutionCancellation | None = None,
        deadline: ExecutionDeadline | None = None,
    ) -> _T:
        """Run one visual operation within the single-asset default cap."""
        return await self._run("visual", operation, phase, cancellation, deadline)

    async def run_provider_attempts(
        self,
        operation: Callable[[int], Awaitable[_T]],
        *,
        phase: str,
        max_attempts: int,
        cancellation: ExecutionCancellation | None = None,
        deadline: ExecutionDeadline | None = None,
    ) -> _T:
        """Retry only explicit provider attempts; each call stays budget-admissible."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        for attempt in range(1, max_attempts + 1):
            try:
                return await self.run_llm(
                    lambda: operation(attempt),
                    phase=phase,
                    cancellation=cancellation,
                    deadline=deadline,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                if attempt == max_attempts:
                    raise
        raise AssertionError("provider attempts must return or raise")

    async def run_group(
        self,
        operations: Sequence[Callable[[], Awaitable[_T]]],
        *,
        kind: RuntimeKind,
        phase: str | None = None,
        cancellation: ExecutionCancellation | None = None,
        deadline: ExecutionDeadline | None = None,
    ) -> tuple[_T, ...]:
        """Run siblings in a TaskGroup so any failure cancels the remaining work."""
        results: list[_T | None] = [None] * len(operations)

        async def collect(index: int, operation: Callable[[], Awaitable[_T]]) -> None:
            results[index] = await self._run(
                kind, operation, phase or _phase_for_kind(kind), cancellation, deadline
            )

        async with asyncio.TaskGroup() as group:
            for index, operation in enumerate(operations):
                group.create_task(collect(index, operation))
        return tuple(result for result in results if result is not None)

    async def _run(
        self,
        kind: RuntimeKind,
        operation: Callable[[], Awaitable[_T]],
        phase: str,
        cancellation: ExecutionCancellation | None,
        deadline: ExecutionDeadline | None,
    ) -> _T:
        _raise_if_cancelled(cancellation)
        async with self._semaphores[kind]:
            _raise_if_cancelled(cancellation)
            timeout = self.timeout_for(phase, deadline=deadline)
            if timeout <= 0:
                raise asyncio.TimeoutError("whole-run deadline exhausted")
            async with asyncio.timeout(timeout):
                return await _await_cancellable(operation, cancellation)


async def emit_sse_event(
    send: Callable[[object], Awaitable[None]],
    event: object,
    cancellation: ExecutionCancellation,
) -> None:
    """Forward an event or cancel the shared attempt when its SSE client leaves."""
    try:
        await send(event)
    except (BrokenPipeError, ConnectionError):
        cancellation.cancel("sse_disconnected")
        raise
    except asyncio.CancelledError:
        cancellation.cancel("sse_disconnected")
        raise


async def _await_cancellable(
    operation: Callable[[], Awaitable[_T]], cancellation: ExecutionCancellation | None
) -> _T:
    operation_task = asyncio.create_task(operation())
    cancellation_task = (
        asyncio.create_task(cancellation.wait()) if cancellation is not None else None
    )
    tasks = {operation_task}
    if cancellation_task is not None:
        tasks.add(cancellation_task)
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if cancellation_task in done:
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise asyncio.CancelledError(cancellation.reason)
        return operation_task.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _phase_for_kind(kind: RuntimeKind) -> str:
    return "visual_extract" if kind == "visual" else "evidence_extract"


def _raise_if_cancelled(cancellation: ExecutionCancellation | None) -> None:
    if cancellation is not None and cancellation.is_cancelled():
        raise asyncio.CancelledError(cancellation.reason)


__all__ = [
    "ExecutionCancellation",
    "ExecutionDeadline",
    "V9ExecutionPolicyRuntime",
    "emit_sse_event",
]
