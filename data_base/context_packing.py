"""Source-backed graph context expansion and deterministic context packing."""

from __future__ import annotations

from dataclasses import dataclass
from langchain_core.documents import Document

from data_base.document_metadata import matches_document_id
from graph_rag.anchor_resolver import ChunkAnchorResolver
from graph_rag.schemas import (
    EvidenceAnchor,
    GraphEvidenceBundle,
    GraphEvidenceItem,
    is_graph_evidence_item_eligible,
)

_GRAPH_BASE_BOOST = 0.08
_FULL_PROVENANCE_BOOST = 0.05
_REQUIRED_MODALITY_BOOST = 0.07
_LOW_CONFIDENCE_PENALTY = 0.10


@dataclass(frozen=True, slots=True)
class GraphLocatedChunk:
    """A source document re-located from a graph evidence item."""

    document: Document
    evidence_item: GraphEvidenceItem
    pre_boost_score: float = 0.0


def expand_graph_evidence_to_chunks(
    user_id: str,
    bundle: GraphEvidenceBundle,
    resolver: ChunkAnchorResolver,
) -> list[GraphLocatedChunk]:
    """Resolve eligible graph evidence back to their persisted source documents."""
    located_chunks: list[GraphLocatedChunk] = []
    seen_identities: set[str] = set()

    for item in bundle.final_context_items:
        if not is_graph_evidence_item_eligible(item):
            continue
        for chunk_id, doc_id in _source_chunk_pairs(item):
            anchor = EvidenceAnchor(
                doc_id=doc_id,
                chunk_id=chunk_id,
                quote=item.evidence_quote,
                confidence=item.confidence,
            )
            try:
                result = resolver.resolve(user_id, anchor)
            except (AttributeError, KeyError, OSError, RuntimeError, ValueError):
                continue
            document = result.document
            if (
                document is None
                or result.resolution_status not in {"resolved", "fuzzy_resolved"}
                or result.verification_status not in {"quote_match", "not_checked"}
                or not matches_document_id(document.metadata, doc_id)
            ):
                continue

            identity = _source_chunk_identity(chunk_id, doc_id)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            located_chunks.append(
                GraphLocatedChunk(
                    document=document,
                    evidence_item=item,
                    pre_boost_score=_document_score(document),
                )
            )

    return located_chunks


def score_graph_located_chunks(
    chunks: list[GraphLocatedChunk],
    *,
    required_modalities: list[str],
    max_graph_chunks: int = 5,
) -> list[Document]:
    """Apply bounded graph boosts while retaining source-document metadata."""
    scored: list[Document] = []
    required_modality_set = set(required_modalities)
    for chunk in chunks:
        item = chunk.evidence_item
        score = chunk.pre_boost_score + _GRAPH_BASE_BOOST
        if item.provenance_status == "full":
            score += _FULL_PROVENANCE_BOOST
        modality = str(chunk.document.metadata.get("modality", "text"))
        if modality in required_modality_set:
            score += _REQUIRED_MODALITY_BOOST
        if item.confidence < 0.6:
            score -= _LOW_CONFIDENCE_PENALTY
        score = round(score, 8)

        metadata = dict(chunk.document.metadata)
        metadata.update(
            {
                "selected_by": "graph",
                "graph_evidence_item_id": item.item_id,
                "graph_source_chunk_ids": list(item.source_chunk_ids),
                "graph_source_doc_ids": list(item.source_doc_ids),
                "graph_boost_applied": True,
                "graph_pre_boost_score": chunk.pre_boost_score,
                "graph_post_boost_score": score,
                "graph_drop_reason": None,
                "provenance_status": item.provenance_status,
                "resolution_status": item.resolution_status,
                "verification_status": item.verification_status,
            }
        )
        scored.append(Document(page_content=chunk.document.page_content, metadata=metadata))

    scored.sort(
        key=lambda document: float(document.metadata["graph_post_boost_score"]),
        reverse=True,
    )
    return scored[:max(0, max_graph_chunks)]


def merge_vector_and_graph_docs(
    vector_docs: list[Document],
    graph_docs: list[Document],
    *,
    graph_chunk_ratio: float,
    graph_every_n: int = 3,
) -> list[Document]:
    """Deduplicate and interleave graph-located chunks within a hard ratio cap."""
    if not graph_docs or graph_chunk_ratio <= 0:
        return list(vector_docs)

    vector_after_dedup: list[Document] = []
    vector_by_identity: dict[str, int] = {}
    for position, document in enumerate(vector_docs):
        identity = _canonical_chunk_identity(
            document, fallback_identity=f"vector-fallback:{position}"
        )
        if identity in vector_by_identity:
            continue
        vector_by_identity[identity] = len(vector_after_dedup)
        vector_after_dedup.append(document)

    unique_graph_docs: list[Document] = []
    seen_graph_identities: set[str] = set()
    for position, document in enumerate(graph_docs):
        identity = _canonical_chunk_identity(
            document, fallback_identity=f"graph-fallback:{position}"
        )
        if identity in seen_graph_identities:
            continue
        seen_graph_identities.add(identity)
        unique_graph_docs.append(document)

    capped_ratio = min(graph_chunk_ratio, 1.0)
    max_graph_docs = max(
        1, int((len(vector_after_dedup) + len(unique_graph_docs)) * capped_ratio)
    )
    selected_graph_docs = unique_graph_docs[:max_graph_docs]
    graph_only: list[Document] = []
    for position, document in enumerate(selected_graph_docs):
        identity = _canonical_chunk_identity(
            document, fallback_identity=f"graph-fallback:{position}"
        )
        vector_index = vector_by_identity.get(identity)
        if vector_index is None:
            graph_only.append(document)
            continue

        vector_document = vector_after_dedup[vector_index]
        metadata = {
            **vector_document.metadata,
            **document.metadata,
            "selected_by": "both",
        }
        vector_after_dedup[vector_index] = Document(
            page_content=vector_document.page_content,
            metadata=metadata,
        )

    output: list[Document] = []
    graph_index = 0
    interval = max(1, graph_every_n)
    for vector_index, document in enumerate(vector_after_dedup, start=1):
        output.append(document)
        if vector_index % interval == 0 and graph_index < len(graph_only):
            output.append(graph_only[graph_index])
            graph_index += 1
    output.extend(graph_only[graph_index:])
    return output


def _source_chunk_pairs(item: GraphEvidenceItem) -> list[tuple[str, str]]:
    """Return source chunk/doc pairs only when the document identity is unambiguous."""
    chunk_ids = [chunk_id for chunk_id in item.source_chunk_ids if chunk_id]
    doc_ids = [doc_id for doc_id in item.source_doc_ids if doc_id]
    if len(doc_ids) == 1:
        return [(chunk_id, doc_ids[0]) for chunk_id in chunk_ids]
    if len(chunk_ids) == len(doc_ids):
        return list(zip(chunk_ids, doc_ids))
    return []


def _source_chunk_identity(chunk_id: str, doc_id: str) -> str:
    return f"{doc_id}:{chunk_id}"


def _canonical_chunk_identity(document: Document, *, fallback_identity: str) -> str:
    chunk_id = str(document.metadata.get("chunk_id", "")).strip()
    if chunk_id:
        return f"chunk:{chunk_id}"
    return fallback_identity


def _document_score(document: Document) -> float:
    for key in ("score", "relevance_score", "reranker_score"):
        value = document.metadata.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0
