from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from data_base.RAG_QA_service import (
    RAGResult,
    _classify_graph_need,
    rag_answer_question,
)
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
