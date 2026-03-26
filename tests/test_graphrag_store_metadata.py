from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from graph_rag.schemas import Community, EntityType, GraphDocumentStatus
from graph_rag.store import GraphStore


def _workspace_upload_root(test_name: str) -> Path:
    root = Path("output") / "test_tmp" / f"{test_name}_{uuid4().hex}" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_graph_store_persists_metadata_sidecar() -> None:
    upload_root = _workspace_upload_root("graphrag_store_meta")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("test-user")
        node_id = store.add_node_from_extraction(
            label="Transformer",
            entity_type=EntityType.METHOD,
            doc_id="doc-1",
            pending_resolution=False,
        )
        store.communities = [
            Community(
                id=1,
                node_ids=[node_id],
                title="Leaf",
                summary="Leaf summary",
                level=0,
                parent_id=2,
                ranking_text="Leaf summary",
                summary_version=1,
            ),
            Community(
                id=2,
                node_ids=[node_id],
                title="Parent",
                summary="Parent summary",
                level=1,
                child_ids=[1],
                ranking_text="Parent summary",
                summary_version=1,
            ),
        ]
        store.mark_optimized()
        store.save()

        reloaded = GraphStore("test-user")
        status = reloaded.get_status()

    assert status.index_version == 2
    assert status.community_level_counts == {"0": 1, "1": 1}
    assert status.last_optimized_at is not None
    assert len(reloaded.get_communities(level=1)) == 1
    assert reloaded.get_communities(level=1)[0].child_ids == [1]


def test_graph_store_marks_graph_dirty_after_mutation() -> None:
    upload_root = _workspace_upload_root("graphrag_store_dirty")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("dirty-user")
        store.add_node_from_extraction(
            label="BERT",
            entity_type=EntityType.METHOD,
            doc_id="doc-2",
            pending_resolution=False,
        )
        status = store.get_status()

    assert status.needs_optimization is True


def test_graph_store_falls_back_to_legacy_metadata_when_sidecar_is_corrupted() -> None:
    upload_root = _workspace_upload_root("graphrag_store_corrupt_meta")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("legacy-user")
        node_id = store.add_node_from_extraction(
            label="GraphSAGE",
            entity_type=EntityType.METHOD,
            doc_id="doc-3",
            pending_resolution=False,
        )
        store.communities = [
            Community(
                id=7,
                node_ids=[node_id],
                title="Legacy community",
                summary="Loaded from legacy pickle metadata",
                level=0,
                ranking_text="Legacy community Loaded from legacy pickle metadata",
                summary_version=1,
            )
        ]
        store.mark_optimized()
        store.last_updated = store.last_optimized_at
        graph_path = store._get_graph_path()
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_path, "wb") as f:
            import pickle
            pickle.dump(
                {
                    "graph": store.graph,
                    "communities": [community.model_dump() for community in store.communities],
                    "last_updated": store.last_updated,
                    "pending_count": 0,
                    "last_optimized_at": store.last_optimized_at,
                    "index_version": 1,
                },
                f,
            )
        metadata_path = store._get_metadata_path()
        metadata_path.write_text("{not-valid-json", encoding="utf-8")

        reloaded = GraphStore("legacy-user")

    assert len(reloaded.communities) == 1
    assert reloaded.communities[0].title == "Legacy community"


def test_graph_store_persists_document_status_sidecar_without_graph_pickle() -> None:
    upload_root = _workspace_upload_root("graphrag_store_doc_status")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        artifact_dir = upload_root / "status-user" / "doc-1"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "extracted.md").write_text("demo", encoding="utf-8")

        store = GraphStore("status-user")
        store.upsert_document_status(
            GraphDocumentStatus(
                doc_id="doc-1",
                status="failed",
                chunk_count=2,
                chunks_succeeded=1,
                chunks_failed=1,
                entities_added=3,
                edges_added=2,
                last_error="chunk 1: quota exceeded",
            )
        )
        store.set_active_job_state("rebuild_full")
        store.save_sidecars()

        reloaded = GraphStore("status-user")
        status = reloaded.get_status()

    assert reloaded.get_document_status("doc-1") is not None
    assert reloaded.get_document_status("doc-1").status == "failed"
    assert status.eligible_document_count == 1
    assert status.failed_document_count == 1
    assert status.active_job_state == "rebuild_full"
