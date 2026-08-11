"""Contracts for retrieval-boundary filtering and reranking."""

from langchain_core.documents import Document
import pytest

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

    assert [document.page_content for document in result.documents] == [
        "Second",
        "First",
    ]
    assert [
        document.metadata["relevance_score"] for document in result.documents
    ] == [0.9, 0.4]
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
        "candidate_diversification": {
            "policy": "tail_source_diversity_r1",
            "enabled": False,
            "applied": False,
            "retrieved_doc_ids": ["kept"],
            "candidate_doc_ids": ["kept"],
            "represented_doc_ids_before_tail": [],
            "admitted_doc_ids": [],
        },
    }


def test_unavailable_reranker_preserves_original_top_k_with_none_scores() -> None:
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

    assert result.documents == [first]
    assert "relevance_score" not in result.documents[0].metadata
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
        ],
        "rejected_candidates": [
            {
                "rank": 2,
                "metadata": {"doc_id": "two"},
                "score": None,
                "reason": "selection_limit",
            }
        ],
        "candidate_diversification": {
            "policy": "tail_source_diversity_r1",
            "enabled": False,
            "applied": False,
            "retrieved_doc_ids": ["one", "two"],
            "candidate_doc_ids": ["one", "two"],
            "represented_doc_ids_before_tail": [],
            "admitted_doc_ids": [],
        },
    }


def test_unavailable_reranker_caps_candidates_to_requested_target() -> None:
    documents = [
        Document(page_content=f"Document {index}", metadata={"doc_id": str(index)})
        for index in range(8)
    ]

    result = filter_and_rerank_retrieval(
        "question",
        RagRetrievalResult(documents=documents),
        enable_reranking=True,
        reranker_available=False,
        target_k=4,
        max_candidates=8,
    )

    assert result.documents == documents[:4]
    assert result.metadata["reranking"]["candidate_count"] == 8
    assert [
        row["metadata"]["doc_id"]
        for row in result.metadata["reranking"]["pre_rerank_ranks"]
    ] == [
        str(index) for index in range(8)
    ]
    assert all(
        row["score"] is None
        for row in result.metadata["reranking"]["post_rerank_ranks"]
    )


def test_candidate_diversification_reserves_tail_candidates_for_other_documents() -> None:
    """A multi-source treatment keeps the high-ranked prefix before diversifying."""
    documents = [
        Document(page_content=f"primary-{index}", metadata={"doc_id": "primary"})
        for index in range(8)
    ] + [
        Document(page_content="secondary", metadata={"doc_id": "secondary"}),
        Document(page_content="tertiary", metadata={"doc_id": "tertiary"}),
    ]
    observed_candidates: list[Document] = []

    def preserve_candidate_order(_query, candidates, _top_k):
        observed_candidates[:] = candidates
        return [(document, 1.0) for document in candidates]

    result = filter_and_rerank_retrieval(
        "Compare the primary, secondary, and tertiary models.",
        RagRetrievalResult(documents=documents),
        enable_reranking=True,
        reranker_available=True,
        target_k=4,
        max_candidates=8,
        diversify_rerank_candidates=True,
        rerank_with_scores=preserve_candidate_order,
    )

    assert [document.page_content for document in observed_candidates] == [
        "primary-0",
        "primary-1",
        "primary-2",
        "primary-3",
        "primary-4",
        "primary-5",
        "secondary",
        "tertiary",
    ]
    assert result.metadata["reranking"]["candidate_diversification"] == {
        "policy": "tail_source_diversity_r1",
        "enabled": True,
        "applied": True,
        "retrieved_doc_ids": ["primary", "secondary", "tertiary"],
        "candidate_doc_ids": ["primary", "secondary", "tertiary"],
        "represented_doc_ids_before_tail": ["primary"],
        "admitted_doc_ids": ["secondary", "tertiary"],
    }


def test_reranking_records_candidate_stage_when_diversification_is_disabled() -> None:
    """Exact-structured routes must expose document loss at the candidate cap."""
    documents = [
        Document(page_content=f"primary-{index}", metadata={"doc_id": "primary"})
        for index in range(8)
    ] + [Document(page_content="secondary", metadata={"doc_id": "secondary"})]

    result = filter_and_rerank_retrieval(
        "What does the requested table report?",
        RagRetrievalResult(documents=documents),
        enable_reranking=True,
        reranker_available=True,
        target_k=4,
        max_candidates=8,
        diversify_rerank_candidates=False,
        rerank_with_scores=lambda _query, candidates, _top_k: [
            (document, 1.0) for document in candidates
        ],
    )

    assert result.metadata["reranking"]["candidate_diversification"] == {
        "policy": "tail_source_diversity_r1",
        "enabled": False,
        "applied": False,
        "retrieved_doc_ids": ["primary", "secondary"],
        "candidate_doc_ids": ["primary"],
        "represented_doc_ids_before_tail": [],
        "admitted_doc_ids": [],
    }


def test_strict_reranking_propagates_an_injected_scoring_failure() -> None:
    """A caller that needs fail-soft recovery can distinguish a scoring failure."""
    documents = [
        Document(page_content="First", metadata={"doc_id": "one"}),
        Document(page_content="Second", metadata={"doc_id": "two"}),
    ]

    def failing_reranker(_query, _documents, _top_k):
        raise RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        filter_and_rerank_retrieval(
            "question",
            RagRetrievalResult(documents=documents),
            enable_reranking=True,
            reranker_available=True,
            rerank_with_scores=failing_reranker,
            strict_reranking=True,
        )
