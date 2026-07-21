from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from data_base.rag_graph_locator import locate_graph_sources
from graph_rag.schemas import EvidenceAnchor, GraphEvidenceBundle, GraphEvidenceItem


@pytest.mark.asyncio
async def test_locator_returns_only_resolved_source_documents_not_raw_graph_content() -> None:
    vector_document = Document(
        page_content="Vector-backed source evidence.",
        metadata={"doc_id": "doc-vector", "chunk_id": "chunk-vector"},
    )
    raw_graph_item = GraphEvidenceItem(
        item_id="edge:unresolved",
        graph_mode="local",
        source="edge",
        node_ids=["node-a", "node-b"],
        edge_ids=["edge:unresolved"],
        source_doc_ids=["doc-graph"],
        relation_type="supports",
        summary="Raw inferred graph relation that cannot support a claim.",
        confidence=0.9,
        provenance_status="missing",
        usable_as_context=False,
        use_reason="missing source provenance",
    )
    bundle_locator = AsyncMock(
        return_value=GraphEvidenceBundle(
            query="q",
            route="local-first",
            evidence_items=[raw_graph_item],
            final_context_items=[],
        )
    )

    result = await locate_graph_sources(
        question="q",
        user_id="user-1",
        vector_documents=[vector_document],
        requested_doc_ids=None,
        graph_execution_hints={"evaluation_setup": {"model_name": "configured"}},
        required_modalities=[],
        evidence_mode="locator_to_chunk",
        bundle_locator=bundle_locator,
    )

    assert result.path == "source_expand"
    assert result.route == "local-first"
    assert result.fallback is None
    assert result.documents == [vector_document]
    assert result.resolved_source_documents == []
    assert result.resolved_source_doc_ids == []
    assert all(
        "Raw inferred graph relation" not in document.page_content
        for document in result.documents
    )
    assert not hasattr(result, "raw_graph_context")
    bundle_locator.assert_awaited_once_with(
        question="q",
        user_id="user-1",
        search_mode="generic",
        graph_execution_hints={"evaluation_setup": {"model_name": "configured"}},
        chunk_lookup=result.chunk_lookup,
    )


@pytest.mark.asyncio
async def test_locator_retains_vector_documents_and_reports_fallback_on_lookup_failure() -> None:
    vector_document = Document(page_content="Vector evidence.", metadata={"doc_id": "doc-1"})

    result = await locate_graph_sources(
        question="q",
        user_id="user-1",
        vector_documents=[vector_document],
        requested_doc_ids=["doc-1"],
        graph_execution_hints=None,
        required_modalities=[],
        evidence_mode="locator_to_chunk",
        bundle_locator=AsyncMock(side_effect=OSError("graph index unavailable")),
    )

    assert result.path == "source_expand"
    assert result.route == "none"
    assert result.fallback == "source_expand_failed"
    assert result.documents == [vector_document]
    assert result.resolved_source_documents == []
    assert result.resolved_source_doc_ids == []


@pytest.mark.asyncio
async def test_locator_reports_resolved_source_identifiers_and_merges_only_source_chunks() -> None:
    source_document = Document(
        page_content="Verified source quote with surrounding source context.",
        metadata={
            "doc_id": "doc-graph",
            "chunk_id": "chunk-graph",
            "chunk_hash": "chunk-hash",
        },
    )
    item = GraphEvidenceItem.from_anchor(
        item_id="edge:verified",
        graph_mode="local",
        source="edge",
        edge_ids=["edge:verified"],
        node_ids=["node-a", "node-b"],
        relation_type="supports",
        summary="An inferred summary that must not become the evidence document.",
        anchor=EvidenceAnchor(
            doc_id="doc-graph",
            chunk_id="chunk-graph",
            quote="Verified source quote",
            quote_hash="quote-hash",
            chunk_hash="chunk-hash",
            confidence=0.9,
        ),
        resolution_status="resolved",
        verification_status="quote_match",
    )

    result = await locate_graph_sources(
        question="q",
        user_id="user-1",
        vector_documents=[],
        requested_doc_ids=["doc-graph"],
        graph_execution_hints=None,
        required_modalities=[],
        evidence_mode="locator_to_chunk",
        bundle_locator=AsyncMock(
            return_value=GraphEvidenceBundle(
                query="q", route="local-first", final_context_items=[item]
            )
        ),
        chunk_lookup=_Lookup({"chunk-graph": source_document}),
        graph_chunk_ratio=1.0,
    )

    assert result.resolved_source_documents == [source_document]
    assert result.resolved_source_doc_ids == ["doc-graph"]
    assert result.resolved_source_chunk_ids == ["chunk-graph"]
    assert result.resolved_item_ids == ["edge:verified"]
    assert result.scope_approved_item_ids == ["edge:verified"]
    assert result.scored_item_ids == ["edge:verified"]
    assert result.packed_item_ids == ["edge:verified"]
    assert [document.page_content for document in result.documents] == [
        source_document.page_content
    ]
    assert all("inferred summary" not in document.page_content.lower() for document in result.documents)


class _Lookup:
    def __init__(self, documents: dict[str, Document]) -> None:
        self.documents = documents

    def by_chunk_id(self, user_id: str, chunk_id: str) -> Document | None:
        return self.documents.get(chunk_id)

    def by_doc_and_index(
        self, user_id: str, doc_id: str, chunk_index: int
    ) -> Document | None:
        return None

    def by_chunk_hash(
        self, user_id: str, doc_id: str, chunk_hash: str
    ) -> Document | None:
        return None

    def fuzzy_by_quote(self, user_id: str, doc_id: str, quote: str) -> Document | None:
        return None
