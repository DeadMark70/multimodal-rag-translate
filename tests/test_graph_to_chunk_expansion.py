from langchain_core.documents import Document

from data_base.context_packing import (
    GraphLocatedChunk,
    expand_graph_evidence_to_chunks,
    score_graph_located_chunks,
)
from graph_rag.anchor_resolver import AnchorResolutionResult, ChunkAnchorResolver
from graph_rag.schemas import EvidenceAnchor, GraphEvidenceBundle, GraphEvidenceItem


def _item(
    item_id: str,
    *,
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-1",
    quote: str | None = "Verified source text.",
    confidence: float = 0.9,
    resolution_status: str = "resolved",
    verification_status: str = "quote_match",
) -> GraphEvidenceItem:
    return GraphEvidenceItem(
        item_id=item_id,
        graph_mode="local",
        source="edge",
        node_ids=["a", "b"],
        edge_ids=[item_id],
        source_chunk_ids=[chunk_id],
        source_doc_ids=[doc_id],
        pages=[3],
        relation_type="extends",
        evidence_quote=quote,
        summary="Inferred summary that must not become a document.",
        confidence=confidence,
        provenance_status="full",
        resolution_status=resolution_status,
        verification_status=verification_status,
        usable_as_context=True,
        use_reason="resolved provenance",
    )


class _Lookup:
    def __init__(self, documents: dict[str, Document]) -> None:
        self.documents = documents

    def by_chunk_id(self, user_id: str, chunk_id: str) -> Document | None:
        return self.documents.get(chunk_id)

    def by_doc_and_index(
        self, user_id: str, doc_id: str, chunk_index: int
    ) -> Document | None:
        return None

    def by_chunk_hash(
        self, user_id: str, doc_id: str, chunk_hash: str
    ) -> Document | None:
        return None

    def fuzzy_by_quote(self, user_id: str, doc_id: str, quote: str) -> Document | None:
        return None


def test_expand_graph_evidence_resolves_real_source_document() -> None:
    source = Document(
        page_content="Verified source text. More source context.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1", "modality": "text"},
    )
    bundle = GraphEvidenceBundle(
        query="q", route="local-first", final_context_items=[_item("edge-1")]
    )

    located = expand_graph_evidence_to_chunks(
        "user-1", bundle, ChunkAnchorResolver(_Lookup({"chunk-1": source}))
    )

    assert len(located) == 1
    assert located[0].document is source
    assert located[0].document.page_content != located[0].evidence_item.summary
    assert located[0].evidence_item.item_id == "edge-1"


def test_expand_graph_evidence_excludes_ineligible_mismatched_and_stale_items() -> None:
    valid = _item("valid")
    ineligible = _item("ineligible", resolution_status="unresolved")
    mismatch = _item("mismatch", chunk_id="mismatch")
    stale = _item("stale", chunk_id="stale")
    bundle = GraphEvidenceBundle.model_construct(
        query="q",
        route="local-first",
        hints=[],
        evidence_items=[],
        final_context_items=[valid, ineligible, mismatch, stale, valid],
        token_estimate=0,
    )
    source = Document(
        page_content="Verified source text.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )

    class Resolver:
        def resolve(
            self, user_id: str, anchor: EvidenceAnchor
        ) -> AnchorResolutionResult:
            if anchor.chunk_id == "mismatch":
                return AnchorResolutionResult(
                    anchor=anchor,
                    document=source,
                    resolution_status="resolved",
                    verification_status="quote_mismatch",
                    reason="chunk_id",
                )
            if anchor.chunk_id == "stale":
                return AnchorResolutionResult(
                    anchor=anchor,
                    document=source,
                    resolution_status="stale",
                    verification_status="hash_mismatch",
                    reason="chunk_id_hash_mismatch",
                )
            return AnchorResolutionResult(
                anchor=anchor,
                document=source,
                resolution_status="resolved",
                verification_status="quote_match",
                reason="chunk_id",
            )

    located = expand_graph_evidence_to_chunks("user-1", bundle, Resolver())

    assert [chunk.evidence_item.item_id for chunk in located] == ["valid"]


def test_score_graph_located_chunks_adds_bounded_boost_metadata_and_stable_cap() -> None:
    text_item = _item("text", confidence=0.9)
    low_confidence_item = _item("low-confidence", confidence=0.5)
    chunks = [
        GraphLocatedChunk(
            Document(
                page_content="Text source.",
                metadata={"chunk_id": "text", "doc_id": "doc-1", "modality": "text"},
            ),
            text_item,
            pre_boost_score=0.4,
        ),
        GraphLocatedChunk(
            Document(
                page_content="Low confidence source.",
                metadata={"chunk_id": "low", "doc_id": "doc-1", "modality": "image"},
            ),
            low_confidence_item,
            pre_boost_score=0.4,
        ),
        GraphLocatedChunk(
            Document(
                page_content="Equal source.",
                metadata={"chunk_id": "equal", "doc_id": "doc-1", "modality": "image"},
            ),
            _item("equal", confidence=0.9),
            pre_boost_score=0.4,
        ),
    ]

    scored = score_graph_located_chunks(
        chunks, required_modalities=["text"], max_graph_chunks=2
    )

    assert [document.metadata["graph_evidence_item_id"] for document in scored] == [
        "text",
        "equal",
    ]
    assert scored[0].metadata["selected_by"] == "graph"
    assert scored[0].metadata["graph_post_boost_score"] == 0.6
    assert scored[0].metadata["graph_drop_reason"] is None
    assert scored[0].metadata["provenance_status"] == "full"
    assert scored[0].metadata["resolution_status"] == "resolved"
    assert scored[0].metadata["verification_status"] == "quote_match"


def test_score_graph_located_chunks_uses_item_id_for_equal_score_ties() -> None:
    beta = GraphLocatedChunk(
        Document(page_content="Beta", metadata={"doc_id": "doc-2", "chunk_id": "chunk-2"}),
        _item("beta", doc_id="doc-2", chunk_id="chunk-2"),
        pre_boost_score=0.4,
    )
    alpha = GraphLocatedChunk(
        Document(page_content="Alpha", metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"}),
        _item("alpha", doc_id="doc-1", chunk_id="chunk-1"),
        pre_boost_score=0.4,
    )

    scored = score_graph_located_chunks([beta, alpha], required_modalities=[])

    assert [document.metadata["graph_evidence_item_id"] for document in scored] == [
        "alpha",
        "beta",
    ]
