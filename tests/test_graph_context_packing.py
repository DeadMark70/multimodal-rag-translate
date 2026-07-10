from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from data_base.RAG_QA_service import RAGResult, rag_answer_question
from data_base.context_packing import merge_vector_and_graph_docs
from graph_rag.schemas import GraphEvidenceBundle, GraphEvidenceItem


def _graph_item() -> GraphEvidenceItem:
    return GraphEvidenceItem(
        item_id="edge-1",
        graph_mode="local",
        source="edge",
        node_ids=["a", "b"],
        edge_ids=["edge-1"],
        source_chunk_ids=["graph-chunk"],
        source_doc_ids=["doc-1"],
        pages=[1],
        relation_type="extends",
        evidence_quote="Graph-backed source text.",
        summary="Inferred graph summary.",
        confidence=0.9,
        provenance_status="full",
        resolution_status="resolved",
        verification_status="quote_match",
        usable_as_context=True,
        use_reason="resolved provenance",
    )


def test_merge_vector_and_graph_docs_marks_overlap_and_interleaves_graph_only() -> None:
    vector_docs = [
        Document(
            page_content=f"Vector {index}",
            metadata={"doc_id": "doc-1", "chunk_id": f"v-{index}"},
        )
        for index in range(1, 6)
    ]
    graph_docs = [
        Document(
            page_content="Duplicate",
            metadata={"doc_id": "doc-1", "chunk_id": "v-2", "selected_by": "graph"},
        ),
        Document(
            page_content="Graph 1",
            metadata={"doc_id": "doc-2", "chunk_id": "g-1", "selected_by": "graph"},
        ),
        Document(
            page_content="Graph 2",
            metadata={"doc_id": "doc-2", "chunk_id": "g-2", "selected_by": "graph"},
        ),
    ]

    merged = merge_vector_and_graph_docs(
        vector_docs, graph_docs, graph_chunk_ratio=0.35, graph_every_n=2
    )

    assert [document.metadata.get("chunk_id") for document in merged] == [
        "v-1",
        "v-2",
        "g-1",
        "v-3",
        "v-4",
        "g-2",
        "v-5",
    ]
    assert merged[1].metadata["selected_by"] == "both"


def test_merge_vector_and_graph_docs_preserves_vector_documents_without_chunk_ids() -> None:
    vector_without_id = Document(
        page_content="Vector without ID",
        metadata={"doc_id": "doc-1", "chunk_index": 1, "rank": 1},
    )
    second_vector_without_id = Document(
        page_content="Second vector without ID",
        metadata={"doc_id": "doc-1", "chunk_index": 1, "rank": 2},
    )
    vector_with_id = Document(
        page_content="Vector with ID",
        metadata={"doc_id": "doc-2", "chunk_id": "vector-id", "rank": 2},
    )
    graph_overlap = Document(
        page_content="Graph copy",
        metadata={"doc_id": "doc-2", "chunk_id": "vector-id", "selected_by": "graph"},
    )

    merged = merge_vector_and_graph_docs(
        [vector_without_id, second_vector_without_id, vector_with_id],
        [graph_overlap],
        graph_chunk_ratio=0.35,
    )

    assert [document.page_content for document in merged] == [
        "Vector without ID",
        "Second vector without ID",
        "Vector with ID",
    ]
    assert merged[0].metadata["rank"] == 1
    assert merged[1].metadata["rank"] == 2
    assert merged[2].metadata["selected_by"] == "both"


def test_merge_vector_and_graph_docs_uses_document_and_chunk_identity() -> None:
    vector_doc = Document(
        page_content="Document one chunk.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    graph_doc = Document(
        page_content="Document two chunk.",
        metadata={
            "doc_id": "doc-2",
            "chunk_id": "chunk-1",
            "selected_by": "graph",
        },
    )

    merged = merge_vector_and_graph_docs(
        [vector_doc], [graph_doc], graph_chunk_ratio=1.0
    )

    assert [document.page_content for document in merged] == [
        "Document one chunk.",
        "Document two chunk.",
    ]


def test_merge_vector_and_graph_docs_enforces_strict_graph_only_ratio() -> None:
    vector_docs = [
        Document(page_content=f"Vector {index}", metadata={"doc_id": "doc-1", "chunk_id": f"v-{index}"})
        for index in range(2)
    ]
    graph_docs = [
        Document(
            page_content=f"Graph {index}",
            metadata={"doc_id": "doc-2", "chunk_id": f"g-{index}", "selected_by": "graph"},
        )
        for index in range(4)
    ]

    merged = merge_vector_and_graph_docs(
        vector_docs, graph_docs, graph_chunk_ratio=0.35
    )

    graph_only_count = sum(
        document.metadata.get("selected_by") == "graph" for document in merged
    )
    assert graph_only_count == 1
    assert graph_only_count / len(merged) <= 0.35


def test_merge_vector_and_graph_docs_does_not_force_one_graph_document() -> None:
    vector_doc = Document(
        page_content="Vector", metadata={"doc_id": "doc-1", "chunk_id": "v-1"}
    )
    graph_doc = Document(
        page_content="Graph",
        metadata={"doc_id": "doc-2", "chunk_id": "g-1", "selected_by": "graph"},
    )

    merged = merge_vector_and_graph_docs(
        [vector_doc], [graph_doc], graph_chunk_ratio=0.35
    )

    assert [document.page_content for document in merged] == ["Vector"]


@pytest.mark.asyncio
async def test_graph_to_chunk_flag_disabled_preserves_legacy_graph_wrapper() -> None:
    retriever = Mock()
    retriever.invoke.return_value = [
        Document(page_content="Vector source", metadata={"doc_id": "doc-1"})
    ]
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="answer"))
    legacy_context = AsyncMock(return_value=("=== Graph Evidence ===\nLegacy", [], None))

    with (
        patch("data_base.RAG_QA_service.get_llm", return_value=llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch(
            "data_base.RAG_QA_service.get_user_retriever",
            new=AsyncMock(return_value=retriever),
        ),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-1": "doc.pdf"}),
        ),
        patch("data_base.RAG_QA_service._get_graph_context", new=legacy_context),
        patch(
            "data_base.RAG_QA_service._get_graph_evidence_bundle",
            new=AsyncMock(),
        ) as bundle,
    ):
        result = await rag_answer_question(
            question="q",
            user_id="user-1",
            return_docs=True,
            enable_graph_rag=True,
        )

    assert isinstance(result, RAGResult)
    assert "Legacy" in result.thought_process
    legacy_context.assert_awaited_once()
    bundle.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_to_chunk_flag_uses_source_chunks_and_falls_back_on_lookup_failure() -> None:
    retriever = Mock()
    vector_document = Document(
        page_content="Vector source", metadata={"doc_id": "doc-1", "chunk_id": "vector"}
    )
    retriever.invoke.return_value = [vector_document]
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="answer"))
    bundle = GraphEvidenceBundle(
        query="q", route="local-first", final_context_items=[_graph_item()]
    )

    class FailingLookup:
        def by_chunk_id(self, user_id: str, chunk_id: str) -> Document | None:
            raise OSError("index unavailable")

        def by_doc_and_index(
            self, user_id: str, doc_id: str, chunk_index: int
        ) -> Document | None:
            return None

        def by_chunk_hash(
            self, user_id: str, doc_id: str, chunk_hash: str
        ) -> Document | None:
            return None

        def fuzzy_by_quote(
            self, user_id: str, doc_id: str, quote: str
        ) -> Document | None:
            return None

    lookup = FailingLookup()
    bundle_mock = AsyncMock(return_value=bundle)

    with (
        patch("data_base.RAG_QA_service.get_llm", return_value=llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch(
            "data_base.RAG_QA_service.get_user_retriever",
            new=AsyncMock(return_value=retriever),
        ),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-1": "doc.pdf"}),
        ),
        patch("data_base.RAG_QA_service._get_graph_evidence_bundle", new=bundle_mock),
        patch("data_base.RAG_QA_service.VectorStoreChunkLookup", return_value=lookup),
        patch("data_base.RAG_QA_service._get_graph_context", new=AsyncMock()) as legacy,
    ):
        result = await rag_answer_question(
            question="q",
            user_id="user-1",
            return_docs=True,
            enable_graph_rag=True,
            graph_execution_hints={"graph_to_chunk_enabled": True},
        )

    assert isinstance(result, RAGResult)
    assert result.documents == [vector_document]
    assert "Inferred graph summary" not in result.thought_process
    assert bundle_mock.await_args.kwargs["chunk_lookup"] is lookup
    legacy.assert_not_awaited()
