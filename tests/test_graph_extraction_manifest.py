from graph_rag.schemas import GraphExtractionRunManifest
from graph_rag.store import GraphStore


def _manifest(doc_id: str = "doc-1") -> GraphExtractionRunManifest:
    return GraphExtractionRunManifest(
        extraction_run_id="run-1",
        graph_extraction_version="schema-v1",
        extractor_provider="google",
        extractor_model="gemini-3.1-flash-lite",
        thinking_level="medium",
        extraction_profile="standard",
        prompt_version="graph-schema-v1",
        schema_version="v1",
        doc_id=doc_id,
        chunk_hashes=["chunk-hash"],
        validated=True,
    )


def test_manifest_survives_snapshot_promotion_and_reload(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)
    store.record_extraction_manifest(_manifest())

    version = store.save_snapshot()
    reloaded = GraphStore("user-1", storage_dir=tmp_path)
    manifest = reloaded.get_latest_extraction_manifest("doc-1")

    assert manifest is not None
    assert manifest.thinking_level == "medium"
    assert manifest.graph_snapshot_version == version
    assert "graph.extraction_runs.json" in reloaded.load_current_pointer()["sidecar_hashes"]


def test_missing_manifest_sidecar_is_legacy_compatible(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)

    assert store.get_latest_extraction_manifest("missing") is None
