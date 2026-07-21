"""Filtering and reranking at the generic RAG retrieval boundary.

The module owns retrieval-candidate selection only.  It neither retrieves new
documents nor performs corrective retrieval or answer generation.  Its
metadata is deliberately evidence-oriented so callers can inspect every rank,
threshold, rejection, and score without inventing values that were not
measured.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from data_base.document_metadata import get_document_id
from data_base.rag_pipeline_schemas import RagRetrievalResult
from data_base.reranker import DocumentReranker

logger = logging.getLogger(__name__)

RERANK_TARGET_K = 8
RERANK_CANDIDATE_LIMIT = 12
RERANK_NOISE_KEYWORDS = (
    "SAM",
    "Segment Anything",
    "Interactive Segmentation",
    "SegVol",
)

RerankWithScores = Callable[[str, list[Document], int], list[tuple[Document, float]]]


def filter_and_rerank_retrieval(
    question: str,
    retrieval_result: RagRetrievalResult,
    *,
    doc_ids: list[str] | None = None,
    enable_reranking: bool = False,
    reranker_available: bool | None = None,
    target_k: int = RERANK_TARGET_K,
    max_candidates: int = RERANK_CANDIDATE_LIMIT,
    rerank_with_scores: RerankWithScores | None = None,
) -> RagRetrievalResult:
    """Filter and select retrieved evidence while preserving legacy ordering.

    Scores are ``None`` until a reranker actually returns a measured score.
    This makes unavailable scoring explicit rather than treating rank, a
    threshold, or a fallback value as a relevance score.
    """
    pre_filter_documents = list(retrieval_result.documents)
    filtered_documents, filter_rejections = _filter_document_ids(
        pre_filter_documents, doc_ids
    )
    availability = (
        DocumentReranker.is_initialized()
        if reranker_available is None
        else reranker_available
    )
    rerank_candidates = (
        _limit_rerank_candidates(filtered_documents, max_candidates)
        if enable_reranking
        else list(filtered_documents)
    )
    candidate_rejections = _candidate_limit_rejections(
        filtered_documents, rerank_candidates
    )

    selected_documents, rerank_rows, selection_rejections = _select_documents(
        question,
        rerank_candidates,
        requested_doc_ids=doc_ids,
        enable_reranking=enable_reranking,
        reranker_available=availability,
        target_k=target_k,
        rerank_with_scores=rerank_with_scores,
    )

    metadata = dict(retrieval_result.metadata)
    metadata["filtering"] = {
        "thresholds": {
            "document_ids": list(doc_ids) if doc_ids else None,
            "rerank_candidate_limit": max_candidates if enable_reranking else None,
            "target_k": target_k,
            "relevance_score": None,
        },
        "pre_filter_ranks": _ranked_rows(pre_filter_documents),
        "post_filter_ranks": _ranked_rows(filtered_documents),
        "rejected_candidates": filter_rejections,
    }
    metadata["reranking"] = {
        "enabled": enable_reranking,
        "available": availability if enable_reranking else None,
        "candidate_count": len(rerank_candidates),
        "pre_rerank_ranks": _ranked_rows(rerank_candidates),
        "post_rerank_ranks": rerank_rows,
        "rejected_candidates": candidate_rejections + selection_rejections,
    }
    return RagRetrievalResult(
        documents=selected_documents,
        source_doc_ids=_source_doc_ids(selected_documents),
        context=retrieval_result.context,
        metadata=metadata,
        images=list(retrieval_result.images),
    )


def rerank_documents_for_generation(
    question: str,
    documents: list[Document],
    target_k: int = RERANK_TARGET_K,
) -> list[Document]:
    """Compatibility projection for legacy callers that only need documents."""
    selected, _, _ = _rerank_documents(question, documents, target_k)
    return selected


def limit_rerank_candidates(
    documents: list[Document],
    max_candidates: int = RERANK_CANDIDATE_LIMIT,
) -> list[Document]:
    """Compatibility wrapper around the retrieval-boundary candidate cap."""
    return _limit_rerank_candidates(documents, max_candidates)


def _filter_document_ids(
    documents: list[Document], doc_ids: list[str] | None
) -> tuple[list[Document], list[dict[str, Any]]]:
    if not doc_ids:
        return list(documents), []

    allowed_ids = set(doc_ids)
    accepted: list[Document] = []
    rejected: list[dict[str, Any]] = []
    for rank, document in enumerate(documents, start=1):
        if get_document_id(document.metadata) in allowed_ids:
            accepted.append(document)
        else:
            rejected.append(
                {
                    **_candidate_row(rank, document),
                    "reason": "document_id_filter",
                }
            )
    return accepted, rejected


def _select_documents(
    question: str,
    documents: list[Document],
    *,
    requested_doc_ids: list[str] | None,
    enable_reranking: bool,
    reranker_available: bool,
    target_k: int,
    rerank_with_scores: RerankWithScores | None,
) -> tuple[list[Document], list[dict[str, Any]], list[dict[str, Any]]]:
    if enable_reranking and reranker_available and documents:
        selected, scores, rejections = _rerank_documents(
            question, documents, target_k, rerank_with_scores
        )
        return selected, _post_rerank_rows(selected, documents, scores), rejections

    if enable_reranking:
        return list(documents), _post_rerank_rows(documents, documents, {}), []

    if requested_doc_ids and len(requested_doc_ids) > 1:
        selected = _fair_multi_document_selection(
            documents, requested_doc_ids, target_k
        )
    else:
        selected = documents[:target_k]
    return selected, _post_rerank_rows(selected, documents, {}), _selection_rejections(
        documents, selected
    )


def _rerank_documents(
    question: str,
    documents: list[Document],
    target_k: int,
    rerank_with_scores: RerankWithScores | None = None,
) -> tuple[list[Document], dict[int, float], list[dict[str, Any]]]:
    if not documents or not DocumentReranker.is_initialized() and rerank_with_scores is None:
        selected = documents[:target_k]
        return selected, {}, _selection_rejections(documents, selected)

    if rerank_with_scores is None:
        reranker = DocumentReranker.get_instance()
        scored_documents = reranker.rerank_with_scores(question, documents, len(documents))
    else:
        scored_documents = rerank_with_scores(question, documents, len(documents))
    if not scored_documents:
        selected = documents[:target_k]
        return selected, {}, _selection_rejections(documents, selected)

    if _query_explicitly_requests_noise_topics(question):
        selected_pairs = scored_documents[:target_k]
    else:
        non_noise_documents = [
            entry for entry in scored_documents if not _is_noise_document(entry[0])
        ]
        noise_documents = [
            entry for entry in scored_documents if _is_noise_document(entry[0])
        ]
        selected_pairs = list(non_noise_documents[:target_k])
        if len(selected_pairs) < target_k:
            selected_pairs.extend(noise_documents[: target_k - len(selected_pairs)])

    selected = [document for document, _ in selected_pairs]
    score_by_identity = {id(document): score for document, score in scored_documents}
    logger.debug(
        "Reranker selection complete: total=%s selected=%s",
        len(scored_documents),
        len(selected),
    )
    return selected, score_by_identity, _selection_rejections(
        documents, selected, score_by_identity
    )


def _fair_multi_document_selection(
    documents: list[Document], doc_ids: list[str], target_k: int
) -> list[Document]:
    docs_per_source = max(2, target_k // len(doc_ids))
    docs_by_id: dict[str, list[Document]] = {}
    for document in documents:
        document_id = get_document_id(document.metadata) or "unknown"
        docs_by_id.setdefault(document_id, []).append(document)

    selected: list[Document] = []
    for document_id in doc_ids:
        selected.extend(docs_by_id.get(document_id, [])[:docs_per_source])
    return selected[:target_k]


def _limit_rerank_candidates(
    documents: list[Document], max_candidates: int
) -> list[Document]:
    if len(documents) <= max_candidates:
        return list(documents)
    logger.info(
        "Capping reranker candidates from %s to %s for memory stability",
        len(documents),
        max_candidates,
    )
    return documents[:max_candidates]


def _candidate_limit_rejections(
    documents: list[Document], candidates: list[Document]
) -> list[dict[str, Any]]:
    return _selection_rejections(
        documents, candidates, reason="rerank_candidate_limit"
    )


def _selection_rejections(
    documents: list[Document],
    selected: list[Document],
    scores: dict[int, float] | None = None,
    reason: str = "selection_limit",
) -> list[dict[str, Any]]:
    selected_ids = {id(document) for document in selected}
    return [
        {
            **_candidate_row(rank, document, (scores or {}).get(id(document))),
            "reason": reason,
        }
        for rank, document in enumerate(documents, start=1)
        if id(document) not in selected_ids
    ]


def _post_rerank_rows(
    selected: list[Document], candidates: list[Document], scores: dict[int, float]
) -> list[dict[str, Any]]:
    pre_ranks = {id(document): rank for rank, document in enumerate(candidates, start=1)}
    return [
        {
            "rank": rank,
            "pre_rerank_rank": pre_ranks[id(document)],
            "metadata": dict(document.metadata),
            "score": scores.get(id(document)),
        }
        for rank, document in enumerate(selected, start=1)
    ]


def _ranked_rows(documents: list[Document]) -> list[dict[str, Any]]:
    return [
        _candidate_row(rank, document)
        for rank, document in enumerate(documents, start=1)
    ]


def _candidate_row(
    rank: int, document: Document, score: float | None = None
) -> dict[str, Any]:
    return {"rank": rank, "metadata": dict(document.metadata), "score": score}


def _source_doc_ids(documents: list[Document]) -> list[str]:
    source_doc_ids: list[str] = []
    for document in documents:
        document_id = get_document_id(document.metadata)
        if document_id and document_id not in source_doc_ids:
            source_doc_ids.append(document_id)
    return source_doc_ids


def _query_explicitly_requests_noise_topics(question: str) -> bool:
    query_lower = question.lower()
    return any(keyword.lower() in query_lower for keyword in RERANK_NOISE_KEYWORDS)


def _is_noise_document(document: Document) -> bool:
    content_sample = document.page_content[:500]
    filename = document.metadata.get("file_name") or document.metadata.get("source_file") or ""
    return any(keyword in content_sample or keyword in filename for keyword in RERANK_NOISE_KEYWORDS)


__all__ = [
    "RERANK_CANDIDATE_LIMIT",
    "RERANK_TARGET_K",
    "filter_and_rerank_retrieval",
    "limit_rerank_candidates",
    "rerank_documents_for_generation",
]
