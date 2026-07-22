"""Graph-to-source locator boundary for generic RAG retrieval.

This module deliberately turns graph output into *locations*, never answer
context.  A graph item may influence retrieval only after its provenance anchor
resolves to a persisted source chunk.  Raw graph summaries, relations, and
community hints therefore cannot become claim-supporting documents here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from langchain_core.documents import Document

from data_base.context_packing import (
    GraphLocatedChunk,
    expand_graph_evidence_to_chunks,
    merge_vector_and_graph_docs,
    score_graph_located_chunks,
)
from data_base.document_metadata import get_document_id
from graph_rag.anchor_resolver import ChunkAnchorResolver, ChunkLookup, VectorStoreChunkLookup
from graph_rag.schemas import GraphEvidenceBundle

logger = logging.getLogger(__name__)

GraphBundleLocator = Callable[..., Awaitable[GraphEvidenceBundle]]
ClaimScopeApprover = Callable[[str, GraphLocatedChunk], bool]


@dataclass(frozen=True, slots=True)
class GraphSourceLocatorResult:
    """Source documents and observability produced by graph location.

    ``documents`` contains vector documents plus only source-resolved graph
    documents.  It intentionally has no raw graph context field: graph text is
    not evidence and must not be passed on as claim support.
    """

    documents: list[Document]
    resolved_source_documents: list[Document]
    resolved_source_doc_ids: list[str]
    resolved_source_chunk_ids: list[str]
    candidate_item_ids: list[str]
    resolved_item_ids: list[str]
    scope_approved_item_ids: list[str]
    scored_item_ids: list[str]
    packed_item_ids: list[str]
    route: str
    path: str
    fallback: Optional[str]
    graph_latency_ms: int
    bundle: Optional[GraphEvidenceBundle]
    chunk_lookup: ChunkLookup
    resolved_chunks: list[GraphLocatedChunk]
    scoped_chunks: list[GraphLocatedChunk]
    graph_documents: list[Document]


async def locate_graph_sources(
    *,
    question: str,
    user_id: str,
    vector_documents: list[Document],
    requested_doc_ids: Optional[list[str]],
    graph_execution_hints: Optional[dict[str, Any]],
    required_modalities: list[str],
    evidence_mode: str,
    bundle_locator: GraphBundleLocator,
    search_mode: str = "generic",
    chunk_lookup: Optional[ChunkLookup] = None,
    claim_scope_approver: Optional[ClaimScopeApprover] = None,
    graph_chunk_ratio: float = 0.35,
) -> GraphSourceLocatorResult:
    """Locate graph anchors in persisted chunks and merge those chunks only.

    ``bundle_locator`` is injected by the compatibility layer so its existing
    graph routing and feature-flag seams remain authoritative.  This boundary
    does not instantiate an LLM, read evaluation setup, or render graph text.
    """
    started_at = time.perf_counter()
    lookup = chunk_lookup or VectorStoreChunkLookup()
    original_documents = list(vector_documents)
    bundle: GraphEvidenceBundle | None = None

    try:
        bundle = await bundle_locator(
            question=question,
            user_id=user_id,
            search_mode=search_mode,
            graph_execution_hints=graph_execution_hints,
            chunk_lookup=lookup,
        )
        resolved_chunks = expand_graph_evidence_to_chunks(
            user_id,
            bundle,
            ChunkAnchorResolver(lookup),
        )
        scoped_chunks = _scope_resolved_chunks(
            question=question,
            chunks=resolved_chunks,
            requested_doc_ids=requested_doc_ids,
            evidence_mode=evidence_mode,
            claim_scope_approver=claim_scope_approver,
        )
        graph_documents = score_graph_located_chunks(
            scoped_chunks,
            required_modalities=required_modalities,
        )
        documents = merge_vector_and_graph_docs(
            original_documents,
            graph_documents,
            graph_chunk_ratio=graph_chunk_ratio,
        )
        return _result(
            documents=documents,
            resolved_chunks=resolved_chunks,
            scoped_chunks=scoped_chunks,
            graph_documents=graph_documents,
            route=bundle.route,
            bundle=bundle,
            chunk_lookup=lookup,
            started_at=started_at,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Graph-to-chunk expansion failed; retaining vector retrieval: %s", exc
        )
        return _result(
            documents=original_documents,
            resolved_chunks=[],
            scoped_chunks=[],
            graph_documents=[],
            route=bundle.route if bundle is not None else "none",
            fallback="source_expand_failed",
            bundle=bundle,
            chunk_lookup=lookup,
            started_at=started_at,
        )


def _scope_resolved_chunks(
    *,
    question: str,
    chunks: list[GraphLocatedChunk],
    requested_doc_ids: Optional[list[str]],
    evidence_mode: str,
    claim_scope_approver: Optional[ClaimScopeApprover],
) -> list[GraphLocatedChunk]:
    scoped_chunks = list(chunks)
    if requested_doc_ids:
        allowed_doc_ids = set(requested_doc_ids)
        scoped_chunks = [
            chunk
            for chunk in scoped_chunks
            if get_document_id(chunk.document.metadata) in allowed_doc_ids
        ]
    if evidence_mode == "claim_gated" and claim_scope_approver is not None:
        scoped_chunks = [
            chunk
            for chunk in scoped_chunks
            if claim_scope_approver(question, chunk)
        ]
    return scoped_chunks


def _result(
    *,
    documents: list[Document],
    resolved_chunks: list[GraphLocatedChunk],
    scoped_chunks: list[GraphLocatedChunk],
    graph_documents: list[Document],
    route: str,
    bundle: Optional[GraphEvidenceBundle],
    chunk_lookup: ChunkLookup,
    started_at: float,
    fallback: Optional[str] = None,
) -> GraphSourceLocatorResult:
    resolved_source_documents = [chunk.document for chunk in resolved_chunks]
    return GraphSourceLocatorResult(
        documents=documents,
        resolved_source_documents=resolved_source_documents,
        resolved_source_doc_ids=_unique_nonempty(
            get_document_id(document.metadata) for document in resolved_source_documents
        ),
        resolved_source_chunk_ids=_unique_nonempty(
            str(document.metadata.get("chunk_id", ""))
            for document in resolved_source_documents
        ),
        candidate_item_ids=_unique_nonempty(
            item.item_id for item in (bundle.evidence_items if bundle else [])
        ),
        resolved_item_ids=_unique_nonempty(
            chunk.evidence_item.item_id for chunk in resolved_chunks
        ),
        scope_approved_item_ids=_unique_nonempty(
            chunk.evidence_item.item_id for chunk in scoped_chunks
        ),
        scored_item_ids=_unique_nonempty(
            str(document.metadata.get("graph_evidence_item_id", ""))
            for document in graph_documents
        ),
        packed_item_ids=_unique_nonempty(
            str(document.metadata.get("graph_evidence_item_id", ""))
            for document in documents
        ),
        route=route,
        path="source_expand",
        fallback=fallback,
        graph_latency_ms=max(int((time.perf_counter() - started_at) * 1000), 0),
        bundle=bundle,
        chunk_lookup=chunk_lookup,
        resolved_chunks=resolved_chunks,
        scoped_chunks=scoped_chunks,
        graph_documents=graph_documents,
    )


def _unique_nonempty(values: Any) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))
