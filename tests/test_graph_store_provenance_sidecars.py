from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from graph_rag.schemas import EvidenceAnchor
from graph_rag.store import GraphStore


def test_graph_store_persists_edge_provenance_sidecar() -> None:
    store_dir = Path(tempfile.gettempdir()) / f"graph-store-{uuid4().hex}"
    store_dir.mkdir(parents=True, exist_ok=False)
    try:
        store = GraphStore("user-1", storage_dir=store_dir)

        anchor = EvidenceAnchor(
            doc_id="doc-1",
            chunk_id="chunk-1",
            chunk_index=0,
            page=1,
            quote="A source-backed relation.",
            quote_hash="quote-hash",
            chunk_hash="chunk-hash",
            confidence=0.9,
        )

        edge_id = store.edge_id("node_a", "node_b", "supports")
        store.record_edge_provenance(edge_id, [anchor])
        store.save_sidecars()

        reloaded = GraphStore("user-1", storage_dir=store_dir)
        anchors = reloaded.get_edge_provenance(edge_id)

        assert len(anchors) == 1
        assert anchors[0].provenance_status == "full"
        assert reloaded.get_edge_provenance_status(edge_id) == "full"
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)


def test_graph_store_reports_partial_provenance_when_no_full_anchor() -> None:
    store_dir = Path(tempfile.gettempdir()) / f"graph-store-{uuid4().hex}"
    store_dir.mkdir(parents=True, exist_ok=False)
    try:
        store = GraphStore("user-1", storage_dir=store_dir)

        edge_id = store.edge_id("node_a", "node_b", "supports")
        store.record_edge_provenance(
            edge_id,
            [
                EvidenceAnchor(
                    doc_id="doc-1",
                    chunk_id="chunk-1",
                    chunk_index=0,
                    confidence=0.7,
                )
            ],
        )

        assert store.get_edge_provenance_status(edge_id) == "partial"
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)
