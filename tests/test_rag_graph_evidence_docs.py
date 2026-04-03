from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from data_base.RAG_QA_service import RAGResult, rag_answer_question
from graph_rag.generic_mode import GraphEvidence, estimate_token_count


@pytest.mark.asyncio
async def test_rag_return_docs_includes_graph_evidence_documents() -> None:
    retriever = Mock()
    retriever.invoke.return_value = [
        Document(page_content="Chunk from source doc", metadata={"doc_id": "doc-1"})
    ]
    graph_evidence = GraphEvidence(
        evidence_id="community-answer:1",
        evidence_type="community_answer",
        text="Graph evidence content",
        score=0.78,
        token_estimate=estimate_token_count("Graph evidence content"),
    )

    with (
        patch("data_base.RAG_QA_service.get_llm") as mock_get_llm,
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={"total_tokens": 11}),
        patch("data_base.RAG_QA_service.get_user_retriever", return_value=retriever),
        patch("data_base.RAG_QA_service.fetch_document_filenames", new=AsyncMock(return_value={"doc-1": "doc1.pdf"})),
        patch("data_base.RAG_QA_service._expand_short_chunks", side_effect=lambda docs, _uid: docs),
        patch(
            "data_base.RAG_QA_service._get_graph_context",
            new=AsyncMock(return_value=("Graph Evidence:\nGraph evidence content", [graph_evidence])),
        ),
    ):
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="final answer"))
        mock_get_llm.return_value = mock_llm

        result = await rag_answer_question(
            question="比較 benchmark 指標",
            user_id="user-1",
            enable_reranking=False,
            return_docs=True,
            enable_graph_rag=True,
        )

    assert isinstance(result, RAGResult)
    graph_docs = [doc for doc in result.documents if doc.metadata.get("source") == "graph_evidence"]
    assert len(graph_docs) == 1
    assert graph_docs[0].metadata["evidence_type"] == "community_answer"
    assert "Graph evidence content" in graph_docs[0].page_content
