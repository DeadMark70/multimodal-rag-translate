from pathlib import Path

from langchain_core.documents import Document

from graph_rag.anchor_resolver import ChunkAnchorResolver
from graph_rag.generic_mode import merge_graph_evidence_bundle
from graph_rag.local_search import build_local_evidence_items
from graph_rag.schemas import Community, EntityType, EvidenceAnchor, GraphHint
from graph_rag.store import GraphStore


class FakeChunkLookup:
    def __init__(self, documents: dict[str, Document]) -> None:
        self._documents = documents

    def by_chunk_id(self, user_id: str, chunk_id: str) -> Document | None:
        return self._documents.get(chunk_id)

    def by_doc_and_index(self, user_id: str, doc_id: str, chunk_index: int) -> Document | None:
        return None

    def by_chunk_hash(self, user_id: str, doc_id: str, chunk_hash: str) -> Document | None:
        return None

    def fuzzy_by_quote(self, user_id: str, doc_id: str, quote: str) -> Document | None:
        return None


def _add_edge_with_anchor(
    store: GraphStore,
    source_id: str,
    *,
    relation: str,
    anchor: EvidenceAnchor,
) -> str:
    target_id = store.add_node_from_extraction(
        label=f"target-{relation}",
        entity_type=EntityType.METHOD,
        doc_id="doc-1",
        pending_resolution=False,
    )
    store.add_edge_from_extraction(source_id, target_id, relation, "doc-1")
    edge_id = store.edge_id(source_id, target_id, relation)
    store.record_edge_provenance(edge_id, [anchor])
    return target_id


def test_real_store_filters_unusable_anchors_before_bundle_selection() -> None:
    store = GraphStore(
        "graph-evidence-integration-user",
        storage_dir=Path("tests") / ".graph-evidence-integration",
    )
    source_id = store.add_node_from_extraction(
        label="source",
        entity_type=EntityType.METHOD,
        doc_id="doc-1",
        pending_resolution=False,
    )
    resolved = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-resolved",
        quote="Verified relation.",
        quote_hash="quote-resolved",
        chunk_hash="hash-resolved",
        confidence=0.9,
    )
    stale = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-stale",
        quote="Stale relation.",
        quote_hash="quote-stale",
        chunk_hash="hash-before-update",
        confidence=0.8,
    )
    unresolved = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-unresolved",
        quote="Unresolved relation.",
        quote_hash="quote-unresolved",
        chunk_hash="hash-unresolved",
        confidence=0.7,
    )
    partial = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-partial",
        confidence=0.6,
    )
    mismatched = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-mismatched",
        quote="Quoted relation that is absent.",
        quote_hash="quote-mismatched",
        chunk_hash="hash-mismatched",
        confidence=0.5,
    )
    node_ids = [source_id]
    for relation, anchor in (
        ("resolved", resolved),
        ("stale", stale),
        ("unresolved", unresolved),
        ("partial", partial),
        ("mismatched", mismatched),
    ):
        node_ids.append(
            _add_edge_with_anchor(store, source_id, relation=relation, anchor=anchor)
        )

    lookup = FakeChunkLookup(
        {
            "chunk-resolved": Document(
                page_content="Verified relation.",
                metadata={
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-resolved",
                    "chunk_hash": "hash-resolved",
                },
            ),
            "chunk-stale": Document(
                page_content="Stale relation.",
                metadata={
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-stale",
                    "chunk_hash": "hash-after-update",
                },
            ),
            "chunk-mismatched": Document(
                page_content="Different source wording.",
                metadata={
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-mismatched",
                    "chunk_hash": "hash-mismatched",
                },
            ),
        }
    )

    items = build_local_evidence_items(
        store,
        node_ids,
        user_id="user-1",
        anchor_resolver=ChunkAnchorResolver(lookup),
    )
    bundle = merge_graph_evidence_bundle(
        hints=[],
        evidence_items=items,
        token_budget=800,
        query="q",
        route="local-first",
    )

    assert len(items) == 1
    assert items[0].evidence_quote == "Verified relation."
    assert [item.item_id for item in bundle.final_context_items] == [items[0].item_id]


def test_global_community_material_stays_in_hints_with_real_store() -> None:
    store = GraphStore(
        "graph-evidence-community-user",
        storage_dir=Path("tests") / ".graph-evidence-community",
    )
    node_id = store.add_node_from_extraction(
        label="community-node",
        entity_type=EntityType.METHOD,
        doc_id="doc-1",
        pending_resolution=False,
    )
    store.communities = [
        Community(
            id=1,
            node_ids=[node_id],
            title="Community",
            summary="Inferred community summary.",
        )
    ]
    hint = GraphHint(
        hint_id="community-summary:1",
        hint_type="community_summary",
        text=f"{store.get_communities()[0].title}: {store.get_communities()[0].summary}",
        confidence=0.9,
        source_ids=["community:1"],
    )

    bundle = merge_graph_evidence_bundle(
        hints=[hint],
        evidence_items=[],
        token_budget=800,
        query="q",
        route="global-first",
    )

    assert bundle.hints == [hint]
    assert bundle.final_context_items == []
