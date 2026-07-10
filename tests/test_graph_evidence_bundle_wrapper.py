from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_base.RAG_QA_service import (
    GraphContextDetails,
    _get_graph_context,
    _get_graph_evidence_bundle,
    _render_graph_bundle_for_legacy_prompt,
)
from graph_rag.generic_mode import GraphEvidence, GraphRouteDecision
from graph_rag.schemas import EvidenceAnchor, GraphEvidenceBundle, GraphEvidenceItem


def _resolved_item() -> GraphEvidenceItem:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=0,
        quote="Resolved graph evidence.",
        quote_hash="quote-hash",
        chunk_hash="chunk-hash",
        confidence=0.9,
    )
    return GraphEvidenceItem.from_anchor(
        item_id="edge:resolved",
        graph_mode="local",
        source="edge",
        edge_ids=["edge:resolved"],
        node_ids=["node-a", "node-b"],
        relation_type="supports",
        summary="Resolved graph evidence.",
        anchor=anchor,
        resolution_status="resolved",
    )


def _unresolved_item() -> GraphEvidenceItem:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=0,
        quote="Unresolved graph evidence.",
        quote_hash="quote-hash",
        chunk_hash="chunk-hash",
        confidence=0.9,
    )
    return GraphEvidenceItem.from_anchor(
        item_id="edge:unresolved",
        graph_mode="local",
        source="edge",
        edge_ids=["edge:unresolved"],
        node_ids=["node-a", "node-b"],
        relation_type="supports",
        summary="Unresolved graph evidence.",
        anchor=anchor,
    )


@pytest.mark.asyncio
async def test_graph_raw_current_returns_legacy_string_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = AsyncMock(return_value="=== Graph Evidence ===\nlegacy")
    monkeypatch.setattr(
        "data_base.RAG_QA_service._get_graph_context_legacy_raw", legacy
    )

    context = await _get_graph_context("q", "user-1", search_mode="generic")

    assert context == "=== Graph Evidence ===\nlegacy"
    legacy.assert_awaited_once_with(
        question="q",
        user_id="user-1",
        search_mode="generic",
        graph_execution_hints=None,
        return_evidence=False,
        return_details=False,
    )


@pytest.mark.asyncio
async def test_graph_context_preserves_return_evidence_two_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = GraphEvidence(
        evidence_id="legacy:1",
        evidence_type="local_edge",
        text="legacy",
        score=0.8,
        token_estimate=1,
    )
    monkeypatch.setattr(
        "data_base.RAG_QA_service._get_graph_context_legacy_raw",
        AsyncMock(return_value=("legacy", [evidence])),
    )

    payload = await _get_graph_context("q", "user-1", return_evidence=True)

    assert payload == ("legacy", [evidence])


@pytest.mark.asyncio
async def test_graph_context_preserves_return_details_three_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = GraphContextDetails(
        route_decision=GraphRouteDecision(
            query_kind="fact",
            path="local-first",
            hops=1,
            max_nodes=1,
            max_communities=0,
            token_budget=1,
        ),
        matched_entity_ids=[],
        community_ids=[],
        candidate_evidence_count=0,
        graph_latency_ms=0,
    )
    monkeypatch.setattr(
        "data_base.RAG_QA_service._get_graph_context_legacy_raw",
        AsyncMock(return_value=("legacy", [], details)),
    )

    payload = await _get_graph_context(
        "q", "user-1", return_evidence=True, return_details=True
    )

    assert payload == ("legacy", [], details)


@pytest.mark.asyncio
async def test_locator_wrapper_renders_only_resolved_final_context_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = GraphEvidenceBundle(
        query="q",
        route="local-first",
        evidence_items=[_resolved_item(), _unresolved_item()],
        final_context_items=[_resolved_item()],
    )
    monkeypatch.setattr(
        "data_base.RAG_QA_service._get_graph_evidence_bundle",
        AsyncMock(return_value=bundle),
    )

    payload = await _get_graph_context(
        "q",
        "user-1",
        graph_execution_hints={"graph_evidence_locator_enabled": True},
        return_evidence=True,
        return_details=True,
    )

    assert payload == ("=== Graph Evidence ===\nResolved graph evidence.", [], None)


@pytest.mark.asyncio
async def test_new_graph_bundle_path_returns_structured_bundle() -> None:
    with patch("graph_rag.store.GraphStore") as store_cls:
        store = store_cls.return_value
        store.get_status.return_value = SimpleNamespace(has_graph=False, node_count=0)

        bundle = await _get_graph_evidence_bundle("q", "user-1", search_mode="generic")

    assert isinstance(bundle, GraphEvidenceBundle)
    assert bundle.route == "none"
    assert bundle.final_context_items == []


@pytest.mark.asyncio
async def test_graph_evidence_bundle_calls_legacy_raw_once_without_recursing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MagicMock()
    store.get_status.return_value = SimpleNamespace(
        has_graph=True,
        node_count=1,
        community_count=0,
        community_level_counts={},
        needs_optimization=False,
    )
    monkeypatch.setattr("graph_rag.store.GraphStore", MagicMock(return_value=store))
    legacy = AsyncMock()
    monkeypatch.setattr(
        "data_base.RAG_QA_service._get_graph_context_legacy_raw", legacy
    )
    local_items = AsyncMock(return_value=([], []))
    monkeypatch.setattr(
        "graph_rag.local_search.local_search_evidence_items", local_items
    )

    bundle = await _get_graph_evidence_bundle("q", "user-1", search_mode="local")

    assert bundle.route == "local-first"
    assert bundle.hints == []
    local_items.assert_awaited_once()
    legacy.assert_not_awaited()


@pytest.mark.asyncio
async def test_lookup_failure_returns_empty_bundle_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingLookup:
        def by_chunk_id(self, user_id: str, chunk_id: str) -> None:
            raise OSError("vector index unavailable")

        def by_doc_and_index(self, user_id: str, doc_id: str, chunk_index: int) -> None:
            raise OSError("vector index unavailable")

        def by_chunk_hash(self, user_id: str, doc_id: str, chunk_hash: str) -> None:
            raise OSError("vector index unavailable")

        def fuzzy_by_quote(self, user_id: str, doc_id: str, quote: str) -> None:
            raise OSError("vector index unavailable")

    store = MagicMock()
    store.get_status.return_value = SimpleNamespace(
        has_graph=True,
        node_count=1,
        community_count=0,
        community_level_counts={},
        needs_optimization=False,
    )
    monkeypatch.setattr("graph_rag.store.GraphStore", MagicMock(return_value=store))

    async def raise_from_local_search(*args, **kwargs):
        kwargs["anchor_resolver"].resolve(
            "user-1",
            EvidenceAnchor(
                doc_id="doc-1",
                chunk_id="chunk-1",
                quote="Source quote.",
                quote_hash="quote-hash",
                chunk_hash="chunk-hash",
                confidence=0.9,
            ),
        )

    monkeypatch.setattr(
        "graph_rag.local_search.local_search_evidence_items",
        raise_from_local_search,
    )

    bundle = await _get_graph_evidence_bundle(
        "q",
        "user-1",
        search_mode="local",
        chunk_lookup=FailingLookup(),
    )

    assert bundle.route == "none"
    assert bundle.hints == []
    assert bundle.evidence_items == []
    assert bundle.final_context_items == []


def test_legacy_renderer_uses_only_verified_quotes_and_rejects_mismatches() -> None:
    verified = _resolved_item()
    verified.summary = "Inferred summary that must never be rendered."
    verified.evidence_quote = "Verified source quote only."
    mismatched = _resolved_item()
    mismatched.evidence_quote = "Mismatched source quote."
    mismatched.verification_status = "quote_mismatch"
    stale = _resolved_item()
    stale.evidence_quote = "Stale source quote."
    stale.verification_status = "hash_mismatch"
    stale.resolution_status = "stale"
    bundle = GraphEvidenceBundle.model_construct(
        query="q",
        route="local-first",
        hints=[],
        evidence_items=[],
        final_context_items=[verified, mismatched, stale],
        token_estimate=0,
    )

    context = _render_graph_bundle_for_legacy_prompt(bundle)

    assert "Verified source quote only." in context
    assert "Inferred summary" not in context
    assert "Mismatched source quote." not in context
    assert "Stale source quote." not in context
