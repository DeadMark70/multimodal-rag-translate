"""Runtime bounds and cancellation contracts for Agentic v9."""

from __future__ import annotations

import asyncio

import pytest

from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.budgeted_llm import invoke_budgeted_llm
from data_base.agentic_v9.execution_policy import (
    ExecutionDeadline,
    ExecutionCancellation,
    V9ExecutionPolicyRuntime,
    emit_sse_event,
)
from data_base.agentic_v9.schemas import ExecutionPolicy


def test_execution_policy_has_the_initial_runtime_bounds() -> None:
    policy = ExecutionPolicy()

    assert policy.max_retrieval_concurrency == 3
    assert policy.max_llm_concurrency == 2
    assert policy.max_visual_concurrency == 1
    assert policy.total_deadline_s == 24.0
    assert policy.phase_timeouts_s == {
        "route_plan": 2.0,
        "retrieval_judge": 2.0,
        "evidence_extract": 8.0,
        "visual_extract": 8.0,
        "final_answer": 15.0,
    }


def test_deadline_clamps_every_phase_timeout_without_resetting() -> None:
    now = [100.0]
    deadline = ExecutionDeadline(24.0, monotonic=lambda: now[0])
    runtime = V9ExecutionPolicyRuntime(ExecutionPolicy())

    assert runtime.timeout_for("final_answer", deadline=deadline) == 15.0
    now[0] = 122.5

    assert deadline.remaining_seconds() == 1.5
    assert runtime.timeout_for("final_answer", deadline=deadline) == 1.5
    assert runtime.timeout_for("route_plan", deadline=deadline) == 1.5


@pytest.mark.asyncio
async def test_runtime_enforces_retrieval_concurrency_and_phase_timeout() -> None:
    runtime = V9ExecutionPolicyRuntime(
        ExecutionPolicy(
            max_retrieval_concurrency=2,
            phase_timeouts_s={"route_plan": 0.01},
        )
    )
    active = 0
    peak = 0
    release = asyncio.Event()

    async def blocked() -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await release.wait()
        finally:
            active -= 1

    tasks = [
        asyncio.create_task(runtime.run_retrieval(blocked, phase="route_plan"))
        for _ in range(3)
    ]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert peak == 2

    with pytest.raises(asyncio.TimeoutError):
        await runtime.run_llm(blocked, phase="route_plan")

    release.set()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_attempt_retries_are_bounded_and_numbered() -> None:
    runtime = V9ExecutionPolicyRuntime(ExecutionPolicy())
    attempts: list[int] = []

    async def flaky(attempt: int) -> str:
        attempts.append(attempt)
        if attempt == 1:
            raise RuntimeError("temporary provider failure")
        return "accepted"

    result = await runtime.run_provider_attempts(
        flaky, phase="evidence_extract", max_attempts=2
    )

    assert result == "accepted"
    assert attempts == [1, 2]


@pytest.mark.asyncio
async def test_task_group_failure_cancels_sibling_operations() -> None:
    runtime = V9ExecutionPolicyRuntime(ExecutionPolicy())
    sibling_cancelled = asyncio.Event()

    async def failing() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("retrieval failed")

    async def sibling() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    with pytest.raises(Exception) as error:
        await runtime.run_group((failing, sibling), kind="retrieval")

    assert any(isinstance(item, RuntimeError) for item in error.value.exceptions)
    assert sibling_cancelled.is_set()


@pytest.mark.asyncio
async def test_campaign_cancellation_stops_an_inflight_operation() -> None:
    runtime = V9ExecutionPolicyRuntime(ExecutionPolicy())
    cancellation = ExecutionCancellation()
    started = asyncio.Event()

    async def blocked() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        runtime.run_retrieval(blocked, cancellation=cancellation)
    )
    await started.wait()
    cancellation.cancel("campaign_cancelled")

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancellation.reason == "campaign_cancelled"


@pytest.mark.asyncio
async def test_sse_disconnect_cancels_the_shared_execution() -> None:
    cancellation = ExecutionCancellation()

    async def disconnected(_: object) -> None:
        raise ConnectionError("client disconnected")

    with pytest.raises(ConnectionError):
        await emit_sse_event(disconnected, {"event": "phase_update"}, cancellation)

    assert cancellation.is_cancelled()
    assert cancellation.reason == "sse_disconnected"


class _CancelledProvider:
    async def ainvoke(self, messages: list[dict[str, object]]) -> object:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_cancelled_provider_attempt_reconciles_its_reservation() -> None:
    controller = RunBudgetController(
        max_llm_calls=1,
        runtime_token_budget=200,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )
    task = asyncio.create_task(
        invoke_budgeted_llm(
            controller=controller,
            provider=_CancelledProvider(),
            phase="final_answer",
            purpose="synthesizer",
            messages=[{"role": "user", "content": "answer"}],
            estimated_input_tokens=100,
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = await controller.snapshot()
    assert snapshot.provider_attempt_count == 1
    assert snapshot.reconciled_tokens == 200
