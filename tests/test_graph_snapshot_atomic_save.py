from graph_rag.schemas import EntityType
from graph_rag.store import GraphStore


def test_snapshot_save_updates_current_pointer_after_complete_write(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)
    store.add_node_from_extraction("MedSAM", EntityType.METHOD, "doc-1")

    version = store.save_snapshot()

    assert version.startswith("v")
    pointer = store.load_current_pointer()
    assert pointer["current_version"] == version
    assert "graph.pkl" in pointer["sidecar_hashes"]
    assert (tmp_path / "versions" / version / "graph.pkl").exists()


def test_new_store_reads_current_snapshot_before_legacy_root(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)
    store.add_node_from_extraction("MedSAM", EntityType.METHOD, "doc-1")
    store.save_snapshot()

    reloaded = GraphStore("user-1", storage_dir=tmp_path)

    assert reloaded.get_status().node_count == 1
