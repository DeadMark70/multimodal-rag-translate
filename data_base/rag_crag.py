"""Corrective retrieval guard for the legacy generic RAG pipeline.

The module owns only the CRAG decision and the corrective retrieval round.  It
does not generate an answer or create a provider itself: callers supply the
optional relevance judge and all rewrite/retrieval seams.  This keeps v8's
LLM-judge behavior intact while allowing deterministic callers to retain the
initial retrieval without a provider invocation.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document

from data_base.document_metadata import get_document_id
from data_base.query_transformer import (
    reciprocal_rank_fusion,
    transform_query_multi,
    transform_query_with_hyde,
)
from data_base.rag_filtering import (
    limit_rerank_candidates,
    rerank_documents_for_generation,
)
from data_base.vector_store_manager import invoke_retriever_queries_async

logger = logging.getLogger(__name__)

CragRewriteMode = Literal["hyde", "multi_query", "none"]
CragClassificationStatus = Literal["insufficient", "judge_required"]
CragStatus = Literal["accepted", "corrected", "insufficient"]
ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
CragJudge = Callable[..., Awaitable[bool]]
HydeTransformer = Callable[..., Awaitable[str]]
MultiQueryTransformer = Callable[..., Awaitable[list[str]]]
QueryExecutor = Callable[[Any, list[str]], Awaitable[list[list[Document]]]]
DocumentReranker = Callable[[str, list[Document], int], list[Document]]
DocumentLimiter = Callable[[list[Document]], list[Document]]
DocumentFusion = Callable[[list[list[Document]]], list[Document]]


@dataclass(frozen=True, slots=True)
class CragClassification:
    """Deterministic pre-judge classification of the retrieval state."""

    status: CragClassificationStatus
    reason: str


@dataclass(slots=True)
class CragRetrievalResult:
    """The selected documents and whether CRAG changed the initial retrieval."""

    documents: list[Document]
    status: CragStatus
    classification: CragClassification
    correction_applied: bool = False


def classify_crag_retrieval(documents: list[Document]) -> CragClassification:
    """Classify retrieval without inferring relevance or invoking a model."""
    if not documents:
        return CragClassification("insufficient", "no_documents")
    return CragClassification("judge_required", "documents_present")


async def build_crag_queries(
    question: str,
    rewrite_mode: CragRewriteMode,
    *,
    hyde_transformer: HydeTransformer = transform_query_with_hyde,
    multi_query_transformer: MultiQueryTransformer = transform_query_multi,
) -> list[str]:
    """Preserve the legacy corrective-rewrite policy without judging relevance."""
    if rewrite_mode == "multi_query":
        transformed = await multi_query_transformer(
            question,
            enabled=True,
            phase="retrieval_rewrite",
        )
        return list(transformed)
    if rewrite_mode == "hyde":
        rewritten = await hyde_transformer(
            question,
            enabled=True,
            phase="retrieval_rewrite",
        )
        return [rewritten or question]
    return [question]


async def judge_retrieved_documents(
    *, question: str, documents: list[Document]
) -> bool:
    """Use the legacy LLM judge only when a caller elects to supply one."""
    from agents.evaluator import RAGEvaluator

    return await RAGEvaluator().grade_documents(question=question, documents=documents)


async def run_corrective_retrieval(
    *,
    question: str,
    documents: list[Document],
    retriever: Any,
    judge: CragJudge | None,
    rewrite_mode: CragRewriteMode = "hyde",
    doc_ids: list[str] | None = None,
    enable_reranking: bool = False,
    reranker_available: bool = False,
    target_k: int = 8,
    progress_callback: ProgressCallback | None = None,
    hyde_transformer: HydeTransformer = transform_query_with_hyde,
    multi_query_transformer: MultiQueryTransformer = transform_query_multi,
    query_executor: QueryExecutor = invoke_retriever_queries_async,
    rerank_documents: DocumentReranker = rerank_documents_for_generation,
    limit_rerank_candidates: DocumentLimiter = limit_rerank_candidates,
    fuse_documents: DocumentFusion = reciprocal_rank_fusion,
) -> CragRetrievalResult:
    """Judge and, only when rejected, perform v8-compatible corrective retrieval.

    A missing judge is deliberately a no-op.  It gives deterministic callers a
    safe, provider-free decision point while the legacy caller passes
    :func:`judge_retrieved_documents` and therefore retains its prior behavior.
    """
    classification = classify_crag_retrieval(documents)
    if classification.status == "insufficient":
        return CragRetrievalResult([], "insufficient", classification)
    if judge is None:
        return CragRetrievalResult(list(documents), "accepted", classification)

    is_relevant = await judge(question=question, documents=documents)
    if is_relevant:
        return CragRetrievalResult(list(documents), "accepted", classification)

    logger.info(
        "CRAG guard detected low relevance for question '%s'; triggering corrective retrieval",
        question[:120],
    )
    await _emit_progress(progress_callback, "crag_correction", {"status": "rewriting_query"})
    corrected_queries = await build_crag_queries(
        question,
        rewrite_mode,
        hyde_transformer=hyde_transformer,
        multi_query_transformer=multi_query_transformer,
    )
    corrected_batches = await query_executor(retriever, corrected_queries)
    corrected_documents = (
        corrected_batches[0]
        if len(corrected_batches) == 1
        else fuse_documents(corrected_batches)
    )
    corrected_documents = _filter_document_ids(corrected_documents, doc_ids)
    if corrected_documents and enable_reranking and reranker_available:
        candidates = limit_rerank_candidates(corrected_documents)
        corrected_documents = await run_in_threadpool(
            rerank_documents, question, candidates, target_k
        )
    elif corrected_documents:
        corrected_documents = corrected_documents[:target_k]

    if not corrected_documents:
        return CragRetrievalResult([], "insufficient", classification, True)
    return CragRetrievalResult(
        list(corrected_documents), "corrected", classification, True
    )


async def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    details: dict[str, Any],
) -> None:
    if progress_callback is not None:
        await progress_callback(stage, details)


def _filter_document_ids(
    documents: list[Document], doc_ids: list[str] | None
) -> list[Document]:
    if not doc_ids:
        return list(documents)
    allowed_ids = set(doc_ids)
    return [
        document
        for document in documents
        if get_document_id(document.metadata) in allowed_ids
    ]


__all__ = [
    "CragClassification",
    "CragRetrievalResult",
    "CragRewriteMode",
    "build_crag_queries",
    "classify_crag_retrieval",
    "judge_retrieved_documents",
    "run_corrective_retrieval",
]
