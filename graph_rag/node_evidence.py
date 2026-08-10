from __future__ import annotations

from core.errors import AppError, ErrorCode
from graph_rag.schemas import (
    EvidenceAnchor,
    GraphNodeEvidenceResponse,
    GraphNodeSourceDocument,
    GraphNodeSourceEvidence,
)
from graph_rag.store import GraphStore
from pdfserviceMD.repository import get_owned_documents_by_ids

_MAX_EVIDENCE = 20


def _safe_bbox(anchor: EvidenceAnchor):
    if anchor.bbox is None or len(anchor.bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in anchor.bbox)
    if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _anchor_key(anchor: EvidenceAnchor):
    return (
        anchor.doc_id,
        anchor.chunk_id or "",
        anchor.page,
        anchor.quote_hash or anchor.quote or "",
    )


async def build_node_evidence_response(
    *, user_id: str, node_key: str
) -> GraphNodeEvidenceResponse:
    store = GraphStore(user_id)
    node = store.get_node(node_key)
    if node is None:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Graph node not found",
            status_code=404,
        )

    anchors = []
    for edge in store.get_edges_for_node(node_key):
        edge_id = store.edge_id(edge.source_id, edge.target_id, edge.relation)
        anchors.extend(store.get_edge_provenance(edge_id))

    candidate_doc_ids = sorted(
        set(node.doc_ids).union(anchor.doc_id for anchor in anchors)
    )
    rows = await get_owned_documents_by_ids(
        doc_ids=candidate_doc_ids,
        user_id=user_id,
        columns="id,file_name",
    )
    document_rows = {
        doc_id: row
        for row in rows
        if isinstance((doc_id := row.get("id")), str) and doc_id
    }

    unique = {}
    for anchor in anchors:
        if (
            anchor.quote
            and anchor.verification_status == "quote_match"
            and anchor.provenance_status in {"full", "partial"}
            and anchor.doc_id in document_rows
        ):
            unique.setdefault(_anchor_key(anchor), anchor)

    ordered = sorted(
        unique.values(),
        key=lambda anchor: (
            0 if anchor.provenance_status == "full" else 1,
            anchor.doc_id,
            anchor.page if anchor.page is not None else 10**9,
        ),
    )[:_MAX_EVIDENCE]

    return GraphNodeEvidenceResponse(
        node_key=node_key,
        label=node.label,
        evidence=[
            GraphNodeSourceEvidence(
                doc_id=anchor.doc_id,
                filename=(document_rows.get(anchor.doc_id) or {}).get("file_name"),
                page=anchor.page,
                quote=anchor.quote,
                bbox=_safe_bbox(anchor),
                provenance_status=anchor.provenance_status,
            )
            for anchor in ordered
        ],
        source_documents=[
            GraphNodeSourceDocument(
                doc_id=doc_id,
                filename=(document_rows.get(doc_id) or {}).get("file_name"),
            )
            for doc_id in sorted(document_rows)
        ],
    )
