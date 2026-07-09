import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import pytest

from graph_rag.schemas import EntityType
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


def test_remove_document_cleans_provenance_for_edges_removed_with_nodes() -> None:
    store_dir = Path(tempfile.gettempdir()) / f"graph-store-{uuid4().hex}"
    store_dir.mkdir(parents=True, exist_ok=False)
    try:
        store = GraphStore("user-1", storage_dir=store_dir)

        source_id = store.add_node_from_extraction("MedSAM", EntityType.METHOD, "doc-1")
        target_id = store.add_node_from_extraction("SAM", EntityType.METHOD, "doc-1")
        store.add_edge_from_extraction(source_id, target_id, "extends", "doc-1")

        edge_id = store.graph.edges[source_id, target_id]["edge_id"]
        store.record_edge_provenance(
            edge_id,
            [
                EvidenceAnchor(
                    doc_id="doc-1",
                    chunk_id="chunk-1",
                    chunk_index=0,
                    page=2,
                    quote="MedSAM extends SAM.",
                    quote_hash="quote-hash",
                    chunk_hash="chunk-hash",
                    confidence=0.95,
                )
            ],
        )
        store.save()

        assert store.get_edge_provenance(edge_id)

        removed_nodes = store.remove_document("doc-1")
        store.save_sidecars()

        reloaded = GraphStore("user-1", storage_dir=store_dir)

        assert removed_nodes == 2
        assert store.graph.number_of_nodes() == 0
        assert store.graph.number_of_edges() == 0
        assert reloaded.get_edge_provenance(edge_id) == []
        assert reloaded.get_edge_provenance_status(edge_id) == "missing"
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)


def test_atomic_write_json_preserves_existing_file_until_replace_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_dir = Path(tempfile.gettempdir()) / f"graph-store-{uuid4().hex}"
    store_dir.mkdir(parents=True, exist_ok=False)
    try:
        store = GraphStore("user-1", storage_dir=store_dir)
        target_path = store_dir / "graph.meta.json"
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        original_payload = {"status": "old"}
        updated_payload = {"status": "new"}

        target_path.write_text(json.dumps(original_payload), encoding="utf-8")

        original_replace = Path.replace

        def fail_replace(self: Path, target: Path) -> Path:
            if self == temp_path and target == target_path:
                raise OSError("replace failed")
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", fail_replace)

        with pytest.raises(OSError, match="replace failed"):
            store._atomic_write_json(target_path, updated_payload)

        assert json.loads(target_path.read_text(encoding="utf-8")) == original_payload
        assert temp_path.exists()

        monkeypatch.setattr(Path, "replace", original_replace)

        store._atomic_write_json(target_path, updated_payload)

        assert json.loads(target_path.read_text(encoding="utf-8")) == updated_payload
        assert not temp_path.exists()
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)
