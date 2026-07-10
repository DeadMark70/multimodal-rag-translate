from graph_rag.schemas import EvidenceAnchor, EntityType, GraphAssetLink
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


def test_snapshot_keeps_evidence_sidecars_readable(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)
    source_id = store.add_node_from_extraction("MedSAM", EntityType.METHOD, "doc-1")
    target_id = store.add_node_from_extraction("memory", EntityType.CONCEPT, "doc-1")
    store.add_edge_from_extraction(
        source_id,
        target_id,
        "uses",
        "doc-1",
    )
    edge_id = store.edge_id(source_id, target_id, "uses")
    store.record_edge_provenance(
        edge_id,
        [
            EvidenceAnchor(
                doc_id="doc-1",
                chunk_id="chunk-1",
                quote="MedSAM uses memory.",
                quote_hash="quote-hash",
                chunk_hash="chunk-hash",
                confidence=0.9,
            )
        ],
    )
    store.record_asset_link(
        GraphAssetLink(
            asset_id="asset-1",
            doc_id="doc-1",
            asset_type="table",
            asset_parse_status="parsed",
        )
    )

    version = store.save_snapshot()
    reloaded = GraphStore("user-1", storage_dir=tmp_path)

    assert edge_id in reloaded.edge_provenance
    assert "asset-1" in reloaded.asset_links
    sidecars = reloaded.load_current_pointer()["sidecar_hashes"]
    assert "graph.provenance.json" in sidecars
    assert "graph.asset_links.json" in sidecars
    assert (tmp_path / "versions" / version / "graph.aliases.json").exists()
