from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from graph_rag.schemas import EntityType
from graph_rag.store import GraphStore


def test_legacy_store_without_provenance_loads_as_missing() -> None:
    store_dir = Path(tempfile.gettempdir()) / f"graph-store-{uuid4().hex}"
    store_dir.mkdir(parents=True, exist_ok=False)
    try:
        store = GraphStore("user-1", storage_dir=store_dir)
        source_id = store.add_node_from_extraction("MedSAM", EntityType.METHOD, "doc-1")
        target_id = store.add_node_from_extraction("SAM", EntityType.METHOD, "doc-1")
        store.add_edge_from_extraction(source_id, target_id, "extends", "doc-1")
        store.save()

        provenance_path = store_dir / "graph.provenance.json"
        if provenance_path.exists():
            provenance_path.unlink()

        reloaded = GraphStore("user-1", storage_dir=store_dir)
        edge_id = reloaded.edge_id(source_id, target_id, "extends")

        assert reloaded.get_edge_provenance(edge_id) == []
        assert reloaded.get_edge_provenance_status(edge_id) == "missing"
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)


def test_edge_id_is_deterministic_hash_not_raw_relation_string() -> None:
    store_dir = Path(tempfile.gettempdir()) / f"graph-store-{uuid4().hex}"
    store_dir.mkdir(parents=True, exist_ok=False)
    try:
        store = GraphStore("user-1", storage_dir=store_dir)

        first = store.edge_id("source/node", "target/node", "method reports metric")
        second = store.edge_id("source/node", "target/node", "method reports metric")

        assert first == second
        assert first.startswith("edge:")
        assert " " not in first
        assert "/" not in first
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)
