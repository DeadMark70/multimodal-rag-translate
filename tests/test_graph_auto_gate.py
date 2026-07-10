from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from data_base.RAG_QA_service import (
    GraphEvidenceLifecycle,
    RAGResult,
    _build_graph_evidence_items,
    _classify_graph_need,
    _graph_execution_strategy,
    _graph_gate_inputs,
    rag_answer_question,
)
from graph_rag.feature_flags import get_graph_feature_flags
from graph_rag.generic_mode import GraphEvidence
from data_base.context_packing import GraphLocatedChunk
from graph_rag.schemas import EvidenceAnchor, GraphEvidenceBundle, GraphEvidenceItem


def test_graph_gate_uses_graph_for_claim_scope() -> None:
    decision = _classify_graph_need(
        "Compare the first claim scope of Weak-Mamba-UNet and Semi-Mamba-UNet"
    )

    assert decision.use_graph is True
    assert decision.role == "locator"
    assert decision.final_graph_context_allowed is True


@pytest.mark.parametrize(
    "question",
    [
        "What Params and FLOPs are reported in Table 1?",
        "Which formula defines the loss?",
        "What exact numeric value is reported?",
    ],
)
def test_graph_gate_uses_locator_for_exact_extraction_with_assets(question: str) -> None:
    decision = _classify_graph_need(question, asset_registry_available=True)

    assert decision.use_graph is True
    assert decision.role == "locator"
    assert decision.locator_only is True
    assert decision.final_graph_context_allowed is False
    assert "exact" in decision.reason.lower()


def test_graph_gate_skips_exact_table_value_when_no_locator_assets_exist() -> None:
    decision = _classify_graph_need(
        "What Params and FLOPs are reported in Table 1?",
        asset_registry_available=False,
    )

    assert decision.use_graph is False
    assert decision.role == "skip"


def test_graph_gate_skips_questions_without_graph_intent() -> None:
    decision = _classify_graph_need("What is Weak-Mamba-UNet?")

    assert decision.use_graph is False
    assert decision.role == "skip"


def test_graph_gate_marks_evolution_queries_for_planning_only() -> None:
    decision = _classify_graph_need("Explain the technical evolution across these papers")

    assert decision.use_graph is True
    assert decision.role == "planning"
    assert decision.locator_only is True
    assert decision.final_graph_context_allowed is False


def test_graph_gate_allows_manual_override_as_locator_only() -> None:
    decision = _classify_graph_need("What Params are reported?", manual_override=True)

    assert decision.use_graph is True
    assert decision.role == "locator"
    assert decision.locator_only is True
    assert decision.final_graph_context_allowed is False


@pytest.mark.parametrize(
    ("auto", "locator", "to_chunk", "expected"),
    [
        (False, False, False, "raw_legacy"),
        (False, False, True, "source_expand"),
        (False, True, False, "skip"),
        (False, True, True, "source_expand"),
        (True, False, False, "skip"),
        (True, False, True, "source_expand"),
        (True, True, False, "skip"),
        (True, True, True, "source_expand"),
    ],
)
def test_graph_execution_flag_matrix_never_routes_partial_locator_flags_to_raw_prompt(
    auto: bool, locator: bool, to_chunk: bool, expected: str
) -> None:
    flags = get_graph_feature_flags(
        {
            "graph_raw_current_enabled": True,
            "graph_auto_gate_enabled": auto,
            "graph_evidence_locator_enabled": locator,
            "graph_to_chunk_enabled": to_chunk,
        }
    )

    decision = _graph_execution_strategy(
        question="Compare the claim scope across papers",
        flags=flags,
        graph_evidence_mode="raw_current",
        manual_override=False,
        asset_registry_available=False,
    )

    assert decision.strategy == expected


def test_graph_execution_strategy_skips_planning_without_packing_or_prompting() -> None:
    strategy = _graph_execution_strategy(
        question="Explain the technical evolution across papers",
        flags=get_graph_feature_flags(
            {
                "graph_raw_current_enabled": True,
                "graph_auto_gate_enabled": True,
                "graph_to_chunk_enabled": True,
            }
        ),
        graph_evidence_mode="router_auto",
        manual_override=False,
        asset_registry_available=False,
    )

    assert strategy.strategy == "skip"
    assert strategy.gate_decision is not None
    assert strategy.gate_decision.role == "planning"


def test_graph_asset_gate_requires_feature_and_request_scoped_probe_result() -> None:
    flags = get_graph_feature_flags({"graph_asset_graph_enabled": True})

    _, unavailable = _graph_gate_inputs(
        {"asset_registry_available": True}, None, flags
    )
    _, available = _graph_gate_inputs(
        {"graph_asset_probe_result": True}, None, flags
    )
    _, disabled_feature = _graph_gate_inputs(
        {"graph_asset_probe_result": True},
        None,
        get_graph_feature_flags({"graph_asset_graph_enabled": False}),
    )

    assert unavailable is False
    assert available is True
    assert disabled_feature is False


def test_graph_evidence_lifecycle_tracks_only_finally_packed_item_ids() -> None:
    lifecycle = GraphEvidenceLifecycle(
        candidate_item_ids=["candidate", "resolved", "scored"],
        resolved_item_ids=["resolved", "scored"],
        scope_approved_item_ids=["scored"],
        scored_item_ids=["scored"],
        packed_item_ids=["scored"],
    )

    assert lifecycle.to_router_reason() == (
        "candidate_ids=candidate,resolved,scored; resolved_ids=resolved,scored; "
        "scope_approved_ids=scored; scored_ids=scored; packed_ids=scored"
    )


def test_graph_observability_marks_only_merged_lifecycle_items_as_packed() -> None:
    lifecycle = GraphEvidenceLifecycle(
        candidate_item_ids=["candidate", "packed"],
        resolved_item_ids=["candidate", "packed"],
        scope_approved_item_ids=["packed"],
        scored_item_ids=["packed"],
        packed_item_ids=["packed"],
    )
    evidence_items = _build_graph_evidence_items(
        graph_event_id="event-1",
        evidence_units=[
            GraphEvidence(
                evidence_id="candidate",
                evidence_type="local_node",
                text="Candidate",
                score=0.8,
                token_estimate=1,
                metadata={"doc_ids": ["doc-1"], "chunk_ids": ["chunk-1"]},
            ),
            GraphEvidence(
                evidence_id="packed",
                evidence_type="local_edge",
                text="Packed",
                score=0.9,
                token_estimate=1,
                metadata={"doc_ids": ["doc-2"], "chunk_ids": ["chunk-2"]},
            ),
        ],
        graph_evidence_mode="locator_to_chunk",
        created_at=datetime.now(timezone.utc),
        lifecycle=lifecycle,
    )

    assert {
        item.graph_evidence_item_id.rsplit(":", maxsplit=1)[1]: item.packed_in_context
        for item in evidence_items
    } == {"candidate": False, "packed": True}


def _vector_document() -> Document:
    return Document(
        page_content="Allowed vector source context.",
        metadata={"doc_id": "doc-allowed", "chunk_id": "vector-1"},
    )


def _graph_item() -> GraphEvidenceItem:
    anchor = EvidenceAnchor(
        doc_id="doc-outside",
        chunk_id="graph-outside",
        quote="Out-of-scope graph source.",
        quote_hash="quote-hash",
        chunk_hash="chunk-hash",
        confidence=0.9,
    )
    return GraphEvidenceItem.from_anchor(
        item_id="edge-outside",
        graph_mode="local",
        source="edge",
        edge_ids=["edge-outside"],
        node_ids=["node-a", "node-b"],
        relation_type="supports",
        summary="Inferred graph summary.",
        anchor=anchor,
        resolution_status="resolved",
        verification_status="quote_match",
    )


def _rag_patches() -> tuple[Mock, Mock, AsyncMock]:
    retriever = Mock()
    retriever.invoke.return_value = [_vector_document()]
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="final answer"))
    get_llm = Mock(return_value=llm)
    return retriever, get_llm, llm.ainvoke


@pytest.mark.asyncio
async def test_auto_gate_skip_does_not_call_graph_or_change_vector_context() -> None:
    retriever, get_llm, _ = _rag_patches()
    graph_context = AsyncMock(return_value="raw graph context")

    with (
        patch("data_base.RAG_QA_service.get_llm", get_llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch("data_base.RAG_QA_service.get_user_retriever", return_value=retriever),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-allowed": "allowed.pdf"}),
        ),
        patch("data_base.RAG_QA_service._get_graph_context", new=graph_context),
    ):
        result = await rag_answer_question(
            question="What is Weak-Mamba-UNet?",
            user_id="user-1",
            enable_graph_rag=True,
            graph_execution_hints={
                "graph_feature_flags": {"graph_auto_gate_enabled": True},
            },
            return_docs=True,
        )

    assert isinstance(result, RAGResult)
    assert [document.metadata["doc_id"] for document in result.documents] == ["doc-allowed"]
    assert "Allowed vector source context." in (result.thought_process or "")
    assert "raw graph context" not in (result.thought_process or "")
    graph_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_gate_locator_only_never_injects_raw_graph_context() -> None:
    retriever, get_llm, _ = _rag_patches()
    graph_context = AsyncMock(return_value="raw inferred graph text")

    with (
        patch("data_base.RAG_QA_service.get_llm", get_llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch("data_base.RAG_QA_service.get_user_retriever", return_value=retriever),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-allowed": "allowed.pdf"}),
        ),
        patch("data_base.RAG_QA_service._get_graph_context", new=graph_context),
    ):
        result = await rag_answer_question(
            question="What Params are reported?",
            user_id="user-1",
            enable_graph_rag=True,
            graph_execution_hints={
                "graph_manual_override": True,
                "graph_feature_flags": {"graph_auto_gate_enabled": True},
            },
            return_docs=True,
        )

    assert isinstance(result, RAGResult)
    assert "raw inferred graph text" not in (result.thought_process or "")
    graph_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_gate_planning_records_skip_without_bundling_merging_or_prompting() -> None:
    retriever, get_llm, _ = _rag_patches()
    graph_context = AsyncMock(return_value="raw inferred graph text")
    graph_bundle = AsyncMock()
    merge = Mock()
    record = AsyncMock()

    with (
        patch("data_base.RAG_QA_service.get_llm", get_llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch("data_base.RAG_QA_service.get_user_retriever", return_value=retriever),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-allowed": "allowed.pdf"}),
        ),
        patch("data_base.RAG_QA_service._get_graph_context", new=graph_context),
        patch("data_base.RAG_QA_service._get_graph_evidence_bundle", new=graph_bundle),
        patch("data_base.RAG_QA_service.merge_vector_and_graph_docs", merge),
        patch("data_base.RAG_QA_service._record_graph_observability", new=record),
    ):
        result = await rag_answer_question(
            question="Explain the technical evolution across papers",
            user_id="user-1",
            enable_graph_rag=True,
            graph_execution_hints={
                "graph_evidence_mode": "router_auto",
                "graph_feature_flags": {
                    "graph_auto_gate_enabled": True,
                    "graph_to_chunk_enabled": True,
                },
            },
            return_docs=True,
        )

    assert isinstance(result, RAGResult)
    graph_context.assert_not_awaited()
    graph_bundle.assert_not_awaited()
    merge.assert_not_called()
    record.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_located_chunks_outside_scoped_doc_ids_are_excluded() -> None:
    retriever, get_llm, _ = _rag_patches()
    item = _graph_item()
    bundle = GraphEvidenceBundle(
        query="Compare claim scope",
        route="local-first",
        evidence_items=[item],
        final_context_items=[item],
    )
    out_of_scope = Document(
        page_content="Out-of-scope graph source.",
        metadata={"doc_id": "doc-outside", "chunk_id": "graph-outside"},
    )

    with (
        patch("data_base.RAG_QA_service.get_llm", get_llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch("data_base.RAG_QA_service.get_user_retriever", return_value=retriever),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-allowed": "allowed.pdf"}),
        ),
        patch(
            "data_base.RAG_QA_service._get_graph_evidence_bundle",
            new=AsyncMock(return_value=bundle),
        ),
        patch(
            "data_base.RAG_QA_service.expand_graph_evidence_to_chunks",
            return_value=[GraphLocatedChunk(out_of_scope, item)],
        ),
        patch(
            "data_base.RAG_QA_service.score_graph_located_chunks",
            return_value=[out_of_scope],
        ),
    ):
        result = await rag_answer_question(
            question="Compare the claim scope across documents",
            user_id="user-1",
            doc_ids=["doc-allowed"],
            enable_graph_rag=True,
            graph_execution_hints={
                "graph_feature_flags": {
                    "graph_auto_gate_enabled": True,
                    "graph_evidence_locator_enabled": True,
                    "graph_to_chunk_enabled": True,
                },
            },
            return_docs=True,
        )

    assert isinstance(result, RAGResult)
    assert {document.metadata.get("doc_id") for document in result.documents} == {"doc-allowed"}
    assert "Out-of-scope graph source." not in (result.thought_process or "")
