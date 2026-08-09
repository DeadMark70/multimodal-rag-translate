"""Functional orchestration stages for the legacy RAG answer path."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document

from data_base.document_metadata import get_document_id
from data_base.query_transformer import (
    transform_query_multi,
    transform_query_with_hyde,
)
from data_base.rag_filtering import (
    RERANK_CANDIDATE_LIMIT,
    RERANK_TARGET_K,
    filter_and_rerank_retrieval,
)
from data_base.rag_pipeline_schemas import ProgressCallback, RAGResult
from data_base.rag_retrieval import retrieve_hybrid_documents
from data_base.reranker import DocumentReranker
from data_base.vector_store_manager import (
    get_user_retriever_async,
    invoke_retriever_queries_async,
)

logger = logging.getLogger(__name__)

LegacyRagResponse = Union[tuple[str, List[str]], RAGResult]


@dataclass(slots=True)
class RetrievalStageOutcome:
    """Evidence and runtime dependencies produced by retrieval selection."""

    documents: List[Document]
    retriever: Any = None
    reranker_available: bool = False
    target_k: int = 0
    terminal_result: Optional[LegacyRagResponse] = None


def _terminal_result(
    message: str,
    source_doc_ids: List[str],
    return_docs: bool,
) -> LegacyRagResponse:
    """Project a terminal pipeline state into the legacy public response."""
    if return_docs:
        return RAGResult(message, source_doc_ids, [])
    return (message, source_doc_ids)


async def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    stage: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a best-effort progress callback when provided."""
    if progress_callback is None:
        return
    await progress_callback(stage, details)


def _resolve_retrieval_policy(
    mode_hints: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    """Resolve bounded retrieval policy overrides from mode hints."""
    policy = (mode_hints or {}).get("retrieval_policy")
    if not isinstance(policy, dict):
        return {}

    resolved: Dict[str, int] = {}
    retrieval_k_raw = policy.get("retrieval_k")
    target_k_raw = policy.get("target_k")
    if retrieval_k_raw is not None:
        try:
            resolved["retrieval_k"] = max(2, min(40, int(retrieval_k_raw)))
        except (TypeError, ValueError):
            pass
    if target_k_raw is not None:
        try:
            resolved["target_k"] = max(2, min(20, int(target_k_raw)))
        except (TypeError, ValueError):
            pass
    return resolved


async def _run_retrieval_stage(
    *,
    question: str,
    user_id: str,
    doc_ids: Optional[List[str]],
    enable_reranking: bool,
    enable_hyde: bool,
    enable_multi_query: bool,
    plain_mode: bool,
    mode_hints: Optional[Dict[str, Any]],
    progress_callback: Optional[ProgressCallback],
    return_docs: bool,
) -> RetrievalStageOutcome:
    """Retrieve, filter, and optionally rerank evidence for generation."""
    retrieval_policy = _resolve_retrieval_policy(mode_hints)
    retrieval_k = int(
        retrieval_policy.get(
            "retrieval_k",
            RERANK_CANDIDATE_LIMIT if enable_reranking else (18 if doc_ids else 6),
        )
    )
    retriever = await get_user_retriever_async(
        user_id,
        retrieval_k,
        plain_mode=plain_mode,
    )
    if retriever is None:
        return RetrievalStageOutcome(
            documents=[],
            terminal_result=_terminal_result(
                "抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。",
                [],
                return_docs,
            ),
        )

    try:
        retrieval_result = await retrieve_hybrid_documents(
            question,
            retriever,
            enable_hyde=enable_hyde,
            enable_multi_query=enable_multi_query,
            progress_callback=progress_callback,
            hyde_transformer=transform_query_with_hyde,
            multi_query_transformer=transform_query_multi,
            query_executor=invoke_retriever_queries_async,
        )
    except (RuntimeError, ValueError) as error:
        logger.error("Retrieval error: %s", error, exc_info=True)
        return RetrievalStageOutcome(
            documents=[],
            terminal_result=_terminal_result(
                "抱歉，檢索知識庫時發生錯誤。",
                [],
                return_docs,
            ),
        )

    if not retrieval_result.documents:
        return RetrievalStageOutcome(
            documents=[],
            terminal_result=_terminal_result(
                "抱歉，在知識庫中找不到相關資訊。",
                [],
                return_docs,
            ),
        )

    target_k = int(retrieval_policy.get("target_k", RERANK_TARGET_K))
    reranker_available = DocumentReranker.is_initialized()
    selection_result = await run_in_threadpool(
        filter_and_rerank_retrieval,
        question,
        retrieval_result,
        doc_ids=doc_ids,
        enable_reranking=enable_reranking,
        reranker_available=reranker_available,
        target_k=target_k,
        max_candidates=RERANK_CANDIDATE_LIMIT,
    )
    documents = selection_result.documents
    reranking_metadata = selection_result.metadata["reranking"]

    if doc_ids and not documents:
        requested_ids = list(doc_ids)
        return RetrievalStageOutcome(
            documents=[],
            terminal_result=_terminal_result(
                "抱歉，在指定的文件中找不到相關資訊。",
                requested_ids,
                return_docs,
            ),
        )

    if doc_ids:
        document_chunk_count: Dict[str, int] = {}
        for document in documents:
            document_id = get_document_id(document.metadata) or "unknown"
            document_chunk_count[document_id] = (
                document_chunk_count.get(document_id, 0) + 1
            )
        logger.info("Multi-doc retrieval: %s", document_chunk_count)

    if enable_reranking:
        await _emit_progress(
            progress_callback,
            "reranking",
            {
                "reranker_available": reranker_available,
                "document_count": len(
                    selection_result.metadata["filtering"]["post_filter_ranks"]
                ),
                "candidate_count": reranking_metadata["candidate_count"],
            },
        )

    if enable_reranking and not reranker_available:
        logger.info(
            "Reranking requested but inactive: %s",
            DocumentReranker.runtime_metadata(reason="runtime_not_initialized"),
        )

    return RetrievalStageOutcome(
        documents=documents,
        retriever=retriever,
        reranker_available=reranker_available,
        target_k=target_k,
    )
