"""Regression coverage for scoped LLM accounting callbacks."""

import asyncio
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)
from pydantic import BaseModel

from core.llm_usage_callback import EvaluationUsageCallback
from core.llm_usage_context import (
    LlmAccountingContext,
    current_llm_accounting_context,
    llm_accounting_phase,
    llm_accounting_scope,
)
from core.providers import configure_providers, get_llm
from evaluation.token_normalizers import normalize_provider_usage


class MemorySink:
    def __init__(self) -> None:
        self.events = []

    async def record(self, event) -> None:
        self.events.append(event)


class FailingSink:
    async def record(self, event) -> None:
        raise RuntimeError("database password=not-for-logs")


class AnswerSchema(BaseModel):
    answer: str


class CallbackFakeChatModel(BaseChatModel):
    """Minimal local chat model that drives LangChain's callback lifecycle."""

    response: AIMessage

    @property
    def _llm_type(self) -> str:
        return "callback-fake"

    def bind_tools(self, *args, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.response)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="ok", usage_metadata=self.response.usage_metadata
            )
        )


def _context(sink, scope_id: str = "scope-1") -> LlmAccountingContext:
    return LlmAccountingContext(
        scope_id=scope_id,
        campaign_id="campaign-1",
        scope_type="execution_run",
        scope_key=scope_id,
        run_id=scope_id,
        metric_name=None,
        sink=sink,
    )


async def simulate_callback(
    callback: EvaluationUsageCallback,
    *,
    usage: dict[str, int],
    run_id=None,
) -> None:
    provider_run_id = run_id or uuid4()
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, **usage}
    await callback.on_chat_model_start(
        {}, [[AIMessage(content="prompt")]], run_id=provider_run_id
    )
    response = LLMResult(
        generations=[
            [ChatGeneration(message=AIMessage(content="ok", usage_metadata=usage))]
        ]
    )
    await callback.on_llm_end(response, run_id=provider_run_id)


async def run_scoped_call(scope_id: str, sink: MemorySink) -> None:
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")
    with llm_accounting_scope(_context(sink, scope_id)):
        await simulate_callback(
            callback,
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


@pytest.mark.asyncio
async def test_callback_records_only_inside_active_scope() -> None:
    sink = MemorySink()
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")

    await simulate_callback(
        callback, usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
    )
    assert sink.events == []

    with (
        llm_accounting_scope(_context(sink)),
        llm_accounting_phase("answer_generation"),
    ):
        await simulate_callback(
            callback, usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
        )

    assert len(sink.events) == 1
    assert sink.events[0].phase == "answer_generation"
    assert sink.events[0].raw_usage["total_tokens"] == 6


@pytest.mark.asyncio
async def test_callback_uses_configured_model_name_for_token_attribution() -> None:
    sink = MemorySink()
    callback = EvaluationUsageCallback(
        purpose="graph_extraction",
        provider="google",
        model_name="gemini-3.1-flash-lite",
    )

    with llm_accounting_scope(_context(sink)):
        await simulate_callback(
            callback,
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    assert sink.events[0].model_name == "gemini-3.1-flash-lite"


@pytest.mark.asyncio
async def test_error_callback_records_redacted_diagnostic_message() -> None:
    sink = MemorySink()
    callback = EvaluationUsageCallback(
        purpose="graph_extraction",
        provider="google",
        model_name="gemini-3.1-flash-lite",
    )
    provider_run_id = uuid4()

    with llm_accounting_scope(_context(sink)):
        await callback.on_chat_model_start(
            {}, [[AIMessage(content="prompt")]], run_id=provider_run_id
        )
        await callback.on_llm_error(
            RuntimeError("400 api_key=super-secret-token: request failed"),
            run_id=provider_run_id,
        )

    assert sink.events[0].model_name == "gemini-3.1-flash-lite"
    assert sink.events[0].error["type"] == "RuntimeError"
    assert "message" in sink.events[0].error
    assert "super-secret-token" not in sink.events[0].error["message"]
    assert "[REDACTED]" in sink.events[0].error["message"]


@pytest.mark.asyncio
async def test_error_callback_redacts_json_bearer_and_query_secrets() -> None:
    sink = MemorySink()
    callback = EvaluationUsageCallback(
        purpose="graph_extraction",
        provider="google",
        model_name="gemini-3.1-flash-lite",
    )
    provider_run_id = uuid4()
    message = (
        'Authorization: Bearer bearer-secret, '
        '{"api_key":"json-secret","access_token":"token-secret"}, '
        "https://api.example.test/generate?key=url-secret&x=1"
    )

    with llm_accounting_scope(_context(sink)):
        await callback.on_chat_model_start(
            {}, [[AIMessage(content="prompt")]], run_id=provider_run_id
        )
        await callback.on_llm_error(RuntimeError(message), run_id=provider_run_id)

    safe_message = sink.events[0].error["message"]
    for secret in ("bearer-secret", "json-secret", "token-secret", "url-secret"):
        assert secret not in safe_message
    assert safe_message.count("[REDACTED]") >= 4


@pytest.mark.asyncio
async def test_concurrent_contexts_do_not_mix() -> None:
    left, right = MemorySink(), MemorySink()

    await asyncio.gather(run_scoped_call("left", left), run_scoped_call("right", right))

    assert {event.scope_id for event in left.events} == {"left"}
    assert {event.scope_id for event in right.events} == {"right"}


@pytest.mark.asyncio
async def test_scope_and_phase_reset_after_exception() -> None:
    sink = MemorySink()

    with pytest.raises(RuntimeError, match="boom"):
        with llm_accounting_scope(_context(sink)), llm_accounting_phase("temporary"):
            raise RuntimeError("boom")

    assert current_llm_accounting_context() is None
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")
    await simulate_callback(callback, usage={"total_tokens": 1})
    assert sink.events == []


@pytest.mark.asyncio
async def test_terminal_callback_emits_once_and_clears_saved_start() -> None:
    sink = MemorySink()
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")
    provider_run_id = uuid4()

    with llm_accounting_scope(_context(sink)):
        await simulate_callback(
            callback, usage={"total_tokens": 1}, run_id=provider_run_id
        )
        await callback.on_llm_error(
            RuntimeError("duplicate terminal"), run_id=provider_run_id
        )

    assert len(sink.events) == 1
    assert callback._starts == {}


@pytest.mark.asyncio
async def test_sink_failure_is_counted_without_raising_or_leaking_error(caplog) -> None:
    context = _context(FailingSink())
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")

    with llm_accounting_scope(context):
        await simulate_callback(callback, usage={"total_tokens": 1})

    assert context.persistence_error_count == 1
    assert "not-for-logs" not in caplog.text


@pytest.mark.asyncio
async def test_fake_provider_emits_direct_event_only_in_scope() -> None:
    configure_providers(use_fake=True)
    fake = get_llm("rag_qa")
    sink = MemorySink()

    await fake.ainvoke([])
    with llm_accounting_scope(_context(sink)), llm_accounting_phase("fake_call"):
        response = await fake.ainvoke([])

    assert response.content.startswith("[TEST_MODE]")
    assert len(sink.events) == 1
    assert sink.events[0].provider == "fake"
    assert sink.events[0].phase == "fake_call"


@pytest.mark.asyncio
async def test_fake_provider_preserves_missing_usage_for_normalization() -> None:
    configure_providers(use_fake=True)
    fake = get_llm("rag_qa")
    sink = MemorySink()

    with llm_accounting_scope(_context(sink)):
        response = await fake.ainvoke([])

    assert response.usage_metadata == {}
    assert sink.events[0].raw_usage == {}
    usage = normalize_provider_usage(sink.events[0].provider, sink.events[0].raw_usage)
    assert usage.usage_status == "missing"
    assert usage.reported_total_tokens is None


@pytest.mark.asyncio
async def test_error_callback_preserves_observed_partial_usage() -> None:
    sink = MemorySink()
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")
    provider_run_id = uuid4()
    partial_response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="partial",
                        usage_metadata={
                            "input_tokens": 4,
                            "output_tokens": 1,
                            "total_tokens": 5,
                        },
                    )
                )
            ]
        ]
    )

    with llm_accounting_scope(_context(sink)):
        await callback.on_chat_model_start(
            {}, [[AIMessage(content="prompt")]], run_id=provider_run_id
        )
        await callback.on_llm_error(
            RuntimeError("stream interrupted"),
            run_id=provider_run_id,
            response=partial_response,
        )

    assert len(sink.events) == 1
    assert sink.events[0].status == "failed"
    assert sink.events[0].raw_usage["total_tokens"] == 5


@pytest.mark.asyncio
async def test_structured_output_preserves_parsed_result_and_emits_once() -> None:
    sink = MemorySink()
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")
    model = CallbackFakeChatModel(
        callbacks=[callback],
        response=AIMessage(
            content="",
            tool_calls=[
                {"name": "AnswerSchema", "args": {"answer": "ok"}, "id": "call-1"}
            ],
            usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        ),
    )

    with llm_accounting_scope(_context(sink)):
        result = await model.with_structured_output(
            AnswerSchema, include_raw=True
        ).ainvoke("prompt")

    assert result["parsed"] == AnswerSchema(answer="ok")
    assert result["raw"].usage_metadata["total_tokens"] == 3
    assert len(sink.events) == 1


@pytest.mark.asyncio
async def test_streaming_terminal_usage_emits_one_combined_event() -> None:
    sink = MemorySink()
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")
    model = CallbackFakeChatModel(
        callbacks=[callback],
        response=AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        ),
    )

    with llm_accounting_scope(_context(sink)):
        chunks = [chunk async for chunk in model.astream("prompt")]

    assert "".join(str(chunk.content) for chunk in chunks) == "ok"
    assert len(sink.events) == 1
    assert sink.events[0].raw_usage["total_tokens"] == 3
