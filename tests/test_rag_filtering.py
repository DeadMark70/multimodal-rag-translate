"""Contracts for retrieval-boundary filtering and reranking."""

from langchain_core.documents import Document

from data_base.rag_filtering import filter_and_rerank_retrieval
from data_base.rag_pipeline_schemas import RagRetrievalResult


def test_filtering_and_reranking_preserve_ranks_thresholds_and_rejections() -> None:
    first = Document(page_content="First", metadata={"doc_id": "kept", "chunk": 1})
    rejected = Document(
        page_content="Rejected", metadata={"doc_id": "excluded", "chunk": 2}
    )
    second = Document(page_content="Second", metadata={"doc_id": "kept", "chunk": 3})
    retrieval = RagRetrievalResult(
        documents=[first, rejected, second],
        source_doc_ids=["kept", "excluded"],
        metadata={"query_expansion": {"mode": "none", "used": False}},
    )

    result = filter_and_rerank_retrieval(
        "question",
        retrieval,
        doc_ids=["kept"],
        enable_reranking=True,
        reranker_available=True,
        target_k=2,
        max_candidates=2,
        rerank_with_scores=lambda _query, _documents, _top_k: [
            (second, 0.9),
            (first, 0.4),
        ],
    )

    assert result.documents == [second, first]
    assert result.source_doc_ids == ["kept"]
    assert result.metadata["query_expansion"] == {"mode": "none", "used": False}
    assert result.metadata["filtering"] == {
        "thresholds": {
            "document_ids": ["kept"],
            "rerank_candidate_limit": 2,
            "target_k": 2,
            "relevance_score": None,
        },
        "pre_filter_ranks": [
            {"rank": 1, "metadata": {"doc_id": "kept", "chunk": 1}, "score": None},
            {"rank": 2, "metadata": {"doc_id": "excluded", "chunk": 2}, "score": None},
            {"rank": 3, "metadata": {"doc_id": "kept", "chunk": 3}, "score": None},
        ],
        "post_filter_ranks": [
            {"rank": 1, "metadata": {"doc_id": "kept", "chunk": 1}, "score": None},
            {"rank": 2, "metadata": {"doc_id": "kept", "chunk": 3}, "score": None},
        ],
        "rejected_candidates": [
            {
                "rank": 2,
                "metadata": {"doc_id": "excluded", "chunk": 2},
                "score": None,
                "reason": "document_id_filter",
            }
        ],
    }
    assert result.metadata["reranking"] == {
        "enabled": True,
        "available": True,
        "candidate_count": 2,
        "pre_rerank_ranks": [
            {"rank": 1, "metadata": {"doc_id": "kept", "chunk": 1}, "score": None},
            {"rank": 2, "metadata": {"doc_id": "kept", "chunk": 3}, "score": None},
        ],
        "post_rerank_ranks": [
            {
                "rank": 1,
                "pre_rerank_rank": 2,
                "metadata": {"doc_id": "kept", "chunk": 3},
                "score": 0.9,
            },
            {
                "rank": 2,
                "pre_rerank_rank": 1,
                "metadata": {"doc_id": "kept", "chunk": 1},
                "score": 0.4,
            },
        ],
        "rejected_candidates": [],
    }


def test_unavailable_reranker_keeps_legacy_candidates_and_uses_none_scores() -> None:
    first = Document(page_content="First", metadata={"doc_id": "one"})
    second = Document(page_content="Second", metadata={"doc_id": "two"})
    retrieval = RagRetrievalResult(documents=[first, second])

    result = filter_and_rerank_retrieval(
        "question",
        retrieval,
        enable_reranking=True,
        reranker_available=False,
        target_k=1,
    )

    assert result.documents == [first, second]
    assert result.metadata["filtering"]["thresholds"]["relevance_score"] is None
    assert result.metadata["reranking"] == {
        "enabled": True,
        "available": False,
        "candidate_count": 2,
        "pre_rerank_ranks": [
            {"rank": 1, "metadata": {"doc_id": "one"}, "score": None},
            {"rank": 2, "metadata": {"doc_id": "two"}, "score": None},
        ],
        "post_rerank_ranks": [
            {
                "rank": 1,
                "pre_rerank_rank": 1,
                "metadata": {"doc_id": "one"},
                "score": None,
            },
            {
                "rank": 2,
                "pre_rerank_rank": 2,
                "metadata": {"doc_id": "two"},
                "score": None,
            },
        ],
        "rejected_candidates": [],
    }
