"""Architecture tests for the Agentic v9 budgeted provider boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.providers as providers_module
from core.llm_factory import current_llm_runtime_overrides, llm_runtime_override
from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.budgeted_llm import BudgetedLlmInvoker
from data_base.agentic_v9.model_paths import (
    V9ClaimVerifier,
    V9ConflictArbiter,
    V9CragJudge,
    V9EvidenceExtractor,
    V9FinalAnswerRenderer,
    V9QueryRewriter,
    V9VisualHelper,
)
from graph_rag.generic_mode import GenericGraphRouter
from evaluation.agentic_v9_admission import build_v9_admission_contract


class _RecordingInvoker:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def invoke(
        self,
        *,
        phase: str,
        purpose: str,
        messages: list[dict[str, object]],
    ) -> object:
        self.calls.append({"phase": phase, "purpose": purpose, "messages": messages})
        return self.response


class _FailingInvoker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(
        self,
        *,
        phase: str,
        purpose: str,
        messages: list[dict[str, object]],
    ) -> object:
        del purpose, messages
        self.calls.append(phase)
        raise TimeoutError("graph route unavailable")


def test_bind_json_schema_uses_native_json_configuration() -> None:
    captured: dict[str, object] = {}

    class _Bindable:
        def bind(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return "bound-provider"

    assert hasattr(providers_module, "bind_json_schema")
    result = providers_module.bind_json_schema(
        _Bindable(),
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    assert result == "bound-provider"
    assert captured == {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
    }


def test_bind_json_schema_rejects_provider_without_bind() -> None:
    assert hasattr(providers_module, "bind_json_schema")
    with pytest.raises(providers_module.ProviderError, match="native JSON schema"):
        providers_module.bind_json_schema(object(), schema={"type": "object"})


def test_v9_runtime_has_no_provider_ainvoke_bypass_outside_budget_gateway() -> None:
    runtime_dir = Path(__file__).parents[1] / "data_base" / "agentic_v9"
    bypasses: list[str] = []
    for path in runtime_dir.glob("*.py"):
        if path.name == "budgeted_llm.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bypasses.extend(
            str(path)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "ainvoke"
        )

    assert bypasses == []


@pytest.mark.asyncio
async def test_preflight_admission_never_invokes_a_provider() -> None:
    async def resolve(user_id: str, references: list[str]) -> dict[str, str]:
        assert user_id == "user-1"
        return {reference: "doc-1" for reference in references}

    admission = await build_v9_admission_contract(
        question="Please investigate this unclear request.",
        user_id="user-1",
        source_references=["paper.pdf"],
        document_reference_resolver=resolve,
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 512},
    )

    assert admission.contract.route_decision is None
    assert admission.contract.strategy_tier == "budgeted_ambiguity"
    assert admission.contract.contract_version == "1"


@pytest.mark.asyncio
async def test_admission_preserves_authoritative_name_to_id_mapping() -> None:
    async def resolve(_user_id: str, _references: list[str]) -> dict[str, str]:
        return {"nnMamba.pdf": "doc-z", "Other.pdf": "doc-a"}

    admission = await build_v9_admission_contract(
        question="From nnMamba.pdf, report the value in Table 2.",
        user_id="user-1",
        source_references=["nnMamba.pdf", "Other.pdf"],
        document_reference_resolver=resolve,
        setup_policy={},
    )

    assert admission.source_scope.source_name_to_doc_ids == {
        "nnMamba.pdf": ["doc-z"],
        "Other.pdf": ["doc-a"],
    }
    assert admission.contract.resolved_source_scope == admission.source_scope
    assert set(admission.contract.resolved_source_scope.authorized_doc_ids) == {
        "doc-z",
        "doc-a",
    }


@pytest.mark.asyncio
async def test_v9_multi_query_rewrite_uses_the_injected_budgeted_invoker() -> None:
    invoker = _RecordingInvoker(
        SimpleNamespace(content="1. first alternate query\n2. second alternate query")
    )

    queries = await V9QueryRewriter(invoker).multi_query("original question")

    assert queries == [
        "original question",
        "first alternate query",
        "second alternate query",
    ]
    assert invoker.calls[0]["phase"] == "query_rewrite"
    assert invoker.calls[0]["purpose"] == "query_rewrite"


@pytest.mark.asyncio
async def test_concrete_invoker_routes_all_v9_calls_through_budget_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    provider = object()

    async def fake_budget_gate(**kwargs: object) -> object:
        captured.update(kwargs)
        return "budgeted response"

    monkeypatch.setattr(
        "data_base.agentic_v9.budgeted_llm.invoke_budgeted_llm", fake_budget_gate
    )
    invoker = BudgetedLlmInvoker(
        controller=object(), provider_factory=lambda purpose: provider
    )

    response = await invoker.invoke(
        phase="claim_verifier",
        purpose="claim_verifier",
        messages=[{"role": "user", "content": "verify"}],
    )

    assert response == "budgeted response"
    assert captured["controller"] is invoker.controller
    assert captured["provider_factory"] is invoker.provider_factory
    assert captured["phase"] == "claim_verifier"


@pytest.mark.asyncio
async def test_concrete_invoker_applies_phase_policy_while_creating_and_invoking_provider() -> (
    None
):
    observed: list[dict[str, object]] = []

    class _Provider:
        async def ainvoke(self, messages: object) -> object:
            observed.append(current_llm_runtime_overrides())
            return {"usage_metadata": {"input_tokens": 1, "output_tokens": 1}}

    controller = RunBudgetController(
        max_llm_calls=1,
        runtime_token_budget=2_000,
        setup_snapshot={
            "max_input_tokens": 1_000,
            "max_output_tokens": 1_000,
            "thinking_enabled": False,
        },
        final_input_tokens=10,
    )
    with llm_runtime_override(
        model_name="gemini-2.5-flash-lite",
        thinking_enabled=False,
        max_input_tokens=1_000,
        max_output_tokens=1_000,
    ):
        response = await BudgetedLlmInvoker(
            controller=controller,
            provider_factory=lambda purpose: (
                observed.append(current_llm_runtime_overrides()) or _Provider()
            ),
        ).invoke(
            phase="final_answer",
            purpose="final_answer",
            messages=[{"role": "user", "content": "answer"}],
        )

    assert response["usage_metadata"]["input_tokens"] == 1
    assert len(observed) == 2
    for config in observed:
        assert config["model_name"] == "gemini-2.5-flash-lite"
        assert config["thinking_enabled"] is False
        assert config["max_output_tokens"] == 1_000
        assert (config["temperature"], config["top_p"], config["top_k"]) == (
            0.25,
            0.9,
            40,
        )


@pytest.mark.asyncio
async def test_contract_planning_provider_limits_come_from_evaluation_setup() -> None:
    observed: list[dict[str, object]] = []

    class _Provider:
        async def ainvoke(self, messages: object) -> object:
            observed.append(current_llm_runtime_overrides())
            return {"usage_metadata": {"input_tokens": 1, "output_tokens": 1}}

    controller = RunBudgetController(
        max_llm_calls=2,
        runtime_token_budget=2_000,
        setup_snapshot={
            "max_input_tokens": 777,
            "max_output_tokens": 123,
            "thinking_enabled": False,
        },
        final_input_tokens=10,
    )
    await BudgetedLlmInvoker(
        controller=controller,
        provider_factory=lambda purpose: _Provider(),
    ).invoke(
        phase="contract_planning",
        purpose="atomic_contract_planning",
        messages=[{"role": "user", "content": "plan"}],
    )

    assert observed[0]["max_input_tokens"] <= 777
    assert observed[0]["max_output_tokens"] == 123
    assert observed[0]["temperature"] == 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_type", "method", "phase", "purpose"),
    [
        (V9CragJudge, "judge", "retrieval_judge", "retrieval_judge"),
        (V9VisualHelper, "extract", "visual_extract", "visual_analysis"),
        (V9EvidenceExtractor, "extract", "evidence_extract", "evidence_extraction"),
        (
            V9ConflictArbiter,
            "arbitrate",
            "conflict_arbitration",
            "conflict_arbitration",
        ),
        (V9ClaimVerifier, "verify", "claim_verifier", "claim_verifier"),
        (V9FinalAnswerRenderer, "render", "final_answer", "final_answer"),
    ],
)
async def test_v9_model_helpers_use_the_injected_invoker(
    adapter_type: type[object], method: str, phase: str, purpose: str
) -> None:
    invoker = _RecordingInvoker(SimpleNamespace(content="accepted"))

    response = await getattr(adapter_type(invoker), method)(
        [{"role": "user", "content": "test"}]
    )

    assert response.content == "accepted"
    assert invoker.calls == [
        {
            "phase": phase,
            "purpose": purpose,
            "messages": [{"role": "user", "content": "test"}],
        }
    ]


@pytest.mark.asyncio
async def test_graph_fallback_router_uses_the_injected_budgeted_invoker() -> None:
    invoker = _RecordingInvoker(
        SimpleNamespace(
            content='{"query_kind": "relation", "path": "blended", "reason": "model"}'
        )
    )

    decision = await GenericGraphRouter(llm_invoker=invoker).route(
        "Explain the implications of this material in depth",
        has_communities=True,
    )

    assert decision.router_reason == "model"
    assert invoker.calls[0]["phase"] == "graph_route"
    assert invoker.calls[0]["purpose"] == "graph_extraction"


@pytest.mark.asyncio
async def test_graph_fast_path_does_not_invoke_provider() -> None:
    invoker = _RecordingInvoker(SimpleNamespace(content="unused"))

    decision = await GenericGraphRouter(llm_invoker=invoker).route(
        "Summarize the overall themes.",
        has_communities=True,
    )

    assert decision.router_reason == "summary_keywords"
    assert invoker.calls == []


@pytest.mark.asyncio
async def test_graph_provider_failure_returns_safe_route_fallback() -> None:
    invoker = _FailingInvoker()

    decision = await GenericGraphRouter(llm_invoker=invoker).route(
        "Explain the implications of this material in depth",
        has_communities=True,
    )

    assert invoker.calls == ["graph_route"]
    assert decision.router_reason == "llm_router_fallback"
    assert decision.path == "blended"
