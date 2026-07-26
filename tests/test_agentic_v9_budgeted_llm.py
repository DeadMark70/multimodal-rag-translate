"""Tests for the provider boundary that reserves before invoking."""

import asyncio

import pytest

from core.llm_usage_context import current_llm_accounting_phase
from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.budgeted_llm import invoke_budgeted_llm
from data_base.agentic_v9.schemas import BudgetExceededError


class _NeverCalledProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: object) -> object:
        self.calls += 1
        return {"usage_metadata": {"total_tokens": 1}}


class _UnavailableProvider:
    async def ainvoke(self, messages: object) -> object:
        raise RuntimeError("provider unavailable")


class _PhaseRecordingProvider:
    def __init__(self) -> None:
        self.phase: str | None = None

    async def ainvoke(self, messages: object) -> object:
        self.phase = current_llm_accounting_phase()
        return {"usage_metadata": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


class _ResponseProvider:
    async def ainvoke(self, messages: object) -> object:
        return {
            "content": "provider answer",
            "usage_metadata": {
                "input_tokens": 11,
                "output_tokens": 5,
                "reasoning_tokens": 3,
                "other_tokens": 2,
                "total_tokens": 21,
            },
        }


class _TimeoutProvider:
    async def ainvoke(self, messages: object) -> object:
        raise asyncio.TimeoutError("secret timeout detail")


class _CancelledProvider:
    async def ainvoke(self, messages: object) -> object:
        raise asyncio.CancelledError


class _SecretFailureProvider:
    async def ainvoke(self, messages: object) -> object:
        raise RuntimeError("api_key=super-secret")


class _RecordingObserver:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[object] = []
        self.fail = fail
        self.partial_reasons: list[str] = []

    async def on_terminal_attempt(self, observation: object) -> bool:
        self.calls.append(observation)
        if self.fail:
            raise OSError("observability unavailable")
        return True

    def mark_partial(self, reason: str) -> None:
        self.partial_reasons.append(reason)


def _controller(*, max_llm_calls: int = 2) -> RunBudgetController:
    return RunBudgetController(
        max_llm_calls=max_llm_calls,
        runtime_token_budget=1000,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )


@pytest.mark.asyncio
async def test_rejected_reservation_prevents_provider_invocation() -> None:
    controller = RunBudgetController(
        max_llm_calls=1,
        runtime_token_budget=200,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )
    provider = _NeverCalledProvider()

    with pytest.raises(BudgetExceededError, match="final_envelope_protected"):
        await invoke_budgeted_llm(
            controller=controller,
            provider=provider,
            phase="route_plan",
            purpose="planner",
            messages=[{"role": "user", "content": "route this"}],
            estimated_input_tokens=1,
        )

    assert provider.calls == 0


@pytest.mark.asyncio
async def test_final_provider_failure_returns_deterministic_qualified_partial() -> None:
    controller = RunBudgetController(
        max_llm_calls=1,
        runtime_token_budget=200,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )

    result = await invoke_budgeted_llm(
        controller=controller,
        provider=_UnavailableProvider(),
        phase="final_answer",
        purpose="synthesizer",
        messages=[{"role": "user", "content": "answer"}],
        estimated_input_tokens=100,
    )

    assert result.response_status == "qualified_partial"
    assert result.final_generation_count == 0
    assert (
        result.answer
        == "Final generation was unavailable; evidence is returned as a qualified partial."
    )


@pytest.mark.asyncio
async def test_budgeted_invocation_exposes_actual_phase_to_provider_usage_callbacks() -> None:
    controller = RunBudgetController(
        max_llm_calls=1,
        runtime_token_budget=400,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )
    provider = _PhaseRecordingProvider()

    await invoke_budgeted_llm(
        controller=controller,
        provider=provider,
        phase="final_answer",
        purpose="synthesizer",
        messages=[{"role": "user", "content": "answer"}],
        estimated_input_tokens=100,
    )

    assert provider.phase == "final_answer"


@pytest.mark.asyncio
async def test_successful_admitted_attempt_emits_complete_terminal_observation() -> None:
    observer = _RecordingObserver()
    response = await invoke_budgeted_llm(
        controller=_controller(),
        provider=_ResponseProvider(),
        observer=observer,
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        phase="evidence_extract",
        purpose="extract_evidence",
        messages=[{"role": "user", "content": "Extract this evidence."}],
        estimated_input_tokens=10,
    )

    assert response["content"] == "provider answer"
    assert len(observer.calls) == 1
    call = observer.calls[0]
    assert call.phase == "evidence_extract"
    assert call.purpose == "extract_evidence"
    assert call.reservation_id
    assert call.provider_attempt == 1
    assert call.provider == "gemini"
    assert call.model_name == "gemini-2.5-flash"
    assert call.prompt_hash
    assert call.response_hash
    assert call.latency_ms >= 0
    assert call.status == "success"
    assert call.error == {}
    assert call.usage == {
        "input_tokens": 11,
        "output_tokens": 5,
        "reasoning_tokens": 3,
        "other_tokens": 2,
        "total_tokens": 21,
        "usage_status": "measured",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "expected_status", "expected_type"),
    [
        (_TimeoutProvider(), "timeout", "TimeoutError"),
        (_SecretFailureProvider(), "failed", "RuntimeError"),
    ],
)
async def test_provider_terminal_failures_are_observed_with_safe_errors(
    provider: object, expected_status: str, expected_type: str
) -> None:
    observer = _RecordingObserver()

    with pytest.raises(Exception):
        await invoke_budgeted_llm(
            controller=_controller(),
            provider=provider,
            observer=observer,
            provider_name="gemini",
            model_name="gemini-2.5-flash",
            phase="evidence_extract",
            purpose="extract_evidence",
            messages=[{"role": "user", "content": "secret prompt"}],
            estimated_input_tokens=10,
        )

    assert len(observer.calls) == 1
    call = observer.calls[0]
    assert call.status == expected_status
    assert call.error == {
        "type": expected_type,
        "message": "provider_attempt_failed",
    }
    assert "secret" not in str(call.error).lower()


@pytest.mark.asyncio
async def test_cancelled_admitted_attempt_emits_one_terminal_observation() -> None:
    observer = _RecordingObserver()

    with pytest.raises(asyncio.CancelledError):
        await invoke_budgeted_llm(
            controller=_controller(),
            provider=_CancelledProvider(),
            observer=observer,
            provider_name="gemini",
            model_name="gemini-2.5-flash",
            phase="evidence_extract",
            purpose="extract_evidence",
            messages=[{"role": "user", "content": "extract"}],
            estimated_input_tokens=10,
        )

    assert len(observer.calls) == 1
    assert observer.calls[0].status == "failed"
    assert observer.calls[0].error == {
        "type": "CancelledError",
        "message": "provider_attempt_cancelled",
    }


@pytest.mark.asyncio
async def test_retries_emit_append_only_attempts_one_and_two() -> None:
    observer = _RecordingObserver()
    controller = RunBudgetController(
        max_llm_calls=3,
        runtime_token_budget=1000,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )

    with pytest.raises(RuntimeError):
        await invoke_budgeted_llm(
            controller=controller,
            provider=_SecretFailureProvider(),
            observer=observer,
            provider_name="gemini",
            model_name="gemini-2.5-flash",
            phase="evidence_extract",
            purpose="extract_evidence",
            messages=[{"role": "user", "content": "extract"}],
            estimated_input_tokens=10,
        )
    await invoke_budgeted_llm(
        controller=controller,
        provider=_ResponseProvider(),
        observer=observer,
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        phase="retrieval_judge",
        purpose="extract_evidence",
        messages=[{"role": "user", "content": "extract"}],
        estimated_input_tokens=10,
    )

    assert [call.provider_attempt for call in observer.calls] == [1, 2]
    assert len({call.reservation_id for call in observer.calls}) == 2
    assert [call.status for call in observer.calls] == ["failed", "success"]


@pytest.mark.asyncio
async def test_budget_rejection_is_not_observed_as_provider_attempt() -> None:
    observer = _RecordingObserver()
    provider = _NeverCalledProvider()
    controller = RunBudgetController(
        max_llm_calls=1,
        runtime_token_budget=200,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )

    with pytest.raises(BudgetExceededError):
        await invoke_budgeted_llm(
            controller=controller,
            provider=provider,
            observer=observer,
            phase="evidence_extract",
            purpose="extract_evidence",
            messages=[{"role": "user", "content": "extract"}],
            estimated_input_tokens=10,
        )

    assert provider.calls == 0
    assert observer.calls == []


@pytest.mark.asyncio
async def test_observer_failure_preserves_answer_and_marks_partial() -> None:
    observer = _RecordingObserver(fail=True)

    response = await invoke_budgeted_llm(
        controller=_controller(),
        provider=_ResponseProvider(),
        observer=observer,
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        phase="evidence_extract",
        purpose="extract_evidence",
        messages=[{"role": "user", "content": "extract"}],
        estimated_input_tokens=10,
    )

    assert response["content"] == "provider answer"
    assert len(observer.calls) == 1
    assert observer.partial_reasons == ["llm_call_observer_failed"]


@pytest.mark.asyncio
async def test_prompt_capture_is_execution_time_bounded_and_sanitized() -> None:
    observer = _RecordingObserver()
    await invoke_budgeted_llm(
        controller=_controller(),
        provider=_ResponseProvider(),
        observer=observer,
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        capture_policy={
            "hash": True,
            "preview": True,
            "full_prompt": False,
            "preview_max_chars": 48,
        },
        phase="evidence_extract",
        purpose="extract_evidence",
        messages=[
            {
                "role": "user",
                "content": (
                    "api_key=sk-live-super-secret "
                    "Extract a deliberately long evidence statement."
                ),
            }
        ],
        estimated_input_tokens=10,
    )

    call = observer.calls[0]
    assert call.prompt_hash
    assert call.prompt_preview is not None
    assert len(call.prompt_preview) <= 48
    assert "super-secret" not in call.prompt_preview
    assert "sk-live" not in call.prompt_preview
    assert call.prompt_capture_status == "redacted"
    assert call.full_prompt is None
    assert call.full_prompt_capture_status == "not_captured_at_execution"
