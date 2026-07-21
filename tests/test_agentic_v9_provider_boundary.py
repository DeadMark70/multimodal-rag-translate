"""Architecture tests for the Agentic v9 budgeted provider boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        self.calls.append(
            {"phase": phase, "purpose": purpose, "messages": messages}
        )
        return self.response


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
async def test_v9_multi_query_rewrite_uses_the_injected_budgeted_invoker() -> None:
    invoker = _RecordingInvoker(
        SimpleNamespace(content="1. first alternate query\n2. second alternate query")
    )

    queries = await V9QueryRewriter(invoker).multi_query("original question")

    assert queries == ["original question", "first alternate query", "second alternate query"]
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
    assert captured["provider"] is provider
    assert captured["phase"] == "claim_verifier"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_type", "method", "phase", "purpose"),
    [
        (V9CragJudge, "judge", "retrieval_judge", "retrieval_judge"),
        (V9VisualHelper, "extract", "visual_extract", "visual_analysis"),
        (V9EvidenceExtractor, "extract", "evidence_extract", "evidence_extraction"),
        (V9ConflictArbiter, "arbitrate", "conflict_arbitration", "conflict_arbitration"),
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
