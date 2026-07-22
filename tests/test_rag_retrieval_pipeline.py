"""Regression contracts for the extracted hybrid retrieval pipeline."""

from unittest.mock import AsyncMock, Mock, call

import pytest
from langchain_core.documents import Document

from data_base.rag_retrieval import retrieve_hybrid_documents


@pytest.mark.asyncio
async def test_multi_query_retrieval_preserves_expansion_origins_ranks_and_metadata() -> (
    None
):
    original = Document(
        page_content="Original evidence",
        metadata={"doc_id": "doc-original", "chunk": 7},
    )
    shared = Document(
        page_content="Shared evidence",
        metadata={"doc_id": "doc-shared", "chunk": 9},
    )
    variant = Document(
        page_content="Variant evidence",
        metadata={"doc_id": "doc-variant", "chunk": 3},
    )
    expand = AsyncMock(return_value=["question", "comparison variant"])
    execute = AsyncMock(return_value=[[original, shared], [variant, shared]])
    progress = AsyncMock()

    retriever = Mock()
    result = await retrieve_hybrid_documents(
        question="question",
        retriever=retriever,
        enable_multi_query=True,
        progress_callback=progress,
        multi_query_transformer=expand,
        query_executor=execute,
    )

    assert result.documents == [shared, original, variant]
    assert result.source_doc_ids == ["doc-shared", "doc-original", "doc-variant"]
    assert result.metadata["query_expansion"] == {
        "mode": "multi_query",
        "used": True,
        "queries": ["question", "comparison variant"],
    }
    assert result.metadata["query_origins"] == [
        {"query": "question", "origin": "original"},
        {"query": "comparison variant", "origin": "multi_query"},
    ]
    assert result.metadata["retrieval_batches"][0]["ranks"] == [
        {"rank": 1, "metadata": {"doc_id": "doc-original", "chunk": 7}},
        {"rank": 2, "metadata": {"doc_id": "doc-shared", "chunk": 9}},
    ]
    assert result.metadata["result_ranks"] == [
        {"rank": 1, "metadata": {"doc_id": "doc-shared", "chunk": 9}},
        {"rank": 2, "metadata": {"doc_id": "doc-original", "chunk": 7}},
        {"rank": 3, "metadata": {"doc_id": "doc-variant", "chunk": 3}},
    ]
    assert result.metadata["fusion"] == {
        "strategy": "reciprocal_rank_fusion",
        "used": True,
    }
    expand.assert_awaited_once_with("question", enabled=True, phase="query_expansion")
    execute.assert_awaited_once_with(retriever, ["question", "comparison variant"])
    progress.assert_has_awaits(
        [
            call("query_expansion", {"mode": "multi_query"}),
            call("retrieval", {"query_count": 2}),
        ]
    )


@pytest.mark.asyncio
async def test_hyde_retrieval_marks_the_transformed_query_origin_without_generation() -> (
    None
):
    document = Document(page_content="HyDE evidence", metadata={"doc_id": "doc-hyde"})
    transform = AsyncMock(return_value="hypothetical answer")
    execute = AsyncMock(return_value=[[document]])

    result = await retrieve_hybrid_documents(
        question="question",
        retriever=Mock(),
        enable_hyde=True,
        hyde_transformer=transform,
        query_executor=execute,
    )

    assert result.documents == [document]
    assert result.metadata["query_expansion"] == {
        "mode": "hyde",
        "used": True,
        "queries": ["hypothetical answer"],
    }
    assert result.metadata["query_origins"] == [
        {"query": "hypothetical answer", "origin": "hyde"}
    ]
    assert result.metadata["fusion"] == {"strategy": "single_query", "used": False}
    transform.assert_awaited_once_with(
        "question", enabled=True, phase="query_expansion"
    )
