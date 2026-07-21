"""Hybrid document retrieval before filtering, reranking, or generation.

This module owns query expansion, execution against the caller-provided hybrid
retriever, and reciprocal-rank fusion.  It deliberately returns evidence and
observability only; later pipeline stages own filtering and answer generation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.documents import Document

from data_base.document_metadata import get_document_id
from data_base.query_transformer import (
    reciprocal_rank_fusion,
    transform_query_multi,
    transform_query_with_hyde,
)
from data_base.rag_pipeline_schemas import RagRetrievalResult
from data_base.vector_store_manager import invoke_retriever_queries_async

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
QueryTransformer = Callable[..., Awaitable[str | list[str]]]
QueryExecutor = Callable[[Any, list[str]], Awaitable[list[list[Document]]]]


async def retrieve_hybrid_documents(
    question: str,
    retriever: Any,
    *,
    enable_hyde: bool = False,
    enable_multi_query: bool = False,
    progress_callback: ProgressCallback | None = None,
    hyde_transformer: QueryTransformer = transform_query_with_hyde,
    multi_query_transformer: QueryTransformer = transform_query_multi,
    query_executor: QueryExecutor = invoke_retriever_queries_async,
) -> RagRetrievalResult:
    """Expand and retrieve evidence without filtering or generating an answer.

    ``retriever`` is intentionally supplied by the caller so its dense/BM25
    composition remains request-scoped and unchanged.  The returned metadata
    records the expansion path and one-based ranks without mutating documents.
    """
    search_queries = [question]
    expansion_mode = "none"

    if enable_hyde:
        expansion_mode = "hyde"
        await _emit_progress(
            progress_callback, "query_expansion", {"mode": expansion_mode}
        )
        transformed = await hyde_transformer(
            question,
            enabled=True,
            phase="query_expansion",
        )
        search_queries = [str(transformed)]
    elif enable_multi_query:
        expansion_mode = "multi_query"
        await _emit_progress(
            progress_callback, "query_expansion", {"mode": expansion_mode}
        )
        transformed_queries = await multi_query_transformer(
            question,
            enabled=True,
            phase="query_expansion",
        )
        search_queries = list(transformed_queries)

    await _emit_progress(
        progress_callback,
        "retrieval",
        {"query_count": len(search_queries)},
    )
    retrieved_batches = await query_executor(retriever, search_queries)
    if len(retrieved_batches) == 1:
        documents = retrieved_batches[0]
        fusion = {"strategy": "single_query", "used": False}
    else:
        documents = reciprocal_rank_fusion(retrieved_batches)
        fusion = {"strategy": "reciprocal_rank_fusion", "used": True}

    query_origins = [
        {
            "query": query,
            "origin": _query_origin(index, expansion_mode),
        }
        for index, query in enumerate(search_queries)
    ]
    return RagRetrievalResult(
        documents=documents,
        source_doc_ids=_source_doc_ids(documents),
        metadata={
            "original_query": question,
            "query_expansion": {
                "mode": expansion_mode,
                "used": expansion_mode != "none",
                "queries": search_queries,
            },
            "query_origins": query_origins,
            "retrieval_batches": [
                {
                    **query_origins[index],
                    "ranks": _ranked_metadata(batch),
                }
                for index, batch in enumerate(retrieved_batches)
            ],
            "fusion": fusion,
            "result_ranks": _ranked_metadata(documents),
        },
    )


async def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    details: dict[str, Any],
) -> None:
    if progress_callback is not None:
        await progress_callback(stage, details)


def _query_origin(index: int, expansion_mode: str) -> str:
    if expansion_mode == "multi_query":
        return "original" if index == 0 else "multi_query"
    return expansion_mode if expansion_mode != "none" else "original"


def _ranked_metadata(documents: list[Document]) -> list[dict[str, Any]]:
    return [
        {"rank": rank, "metadata": dict(document.metadata)}
        for rank, document in enumerate(documents, start=1)
    ]


def _source_doc_ids(documents: list[Document]) -> list[str]:
    source_doc_ids: list[str] = []
    for document in documents:
        document_id = get_document_id(document.metadata)
        if document_id and document_id not in source_doc_ids:
            source_doc_ids.append(document_id)
    return source_doc_ids


__all__ = ["retrieve_hybrid_documents"]
