"""Durable GraphRAG full-rebuild job persistence tests."""

from pathlib import Path

from graph_rag.rebuild_jobs import GraphRebuildJobStore


SOURCES = [
    {
        "doc_id": "doc-1",
        "file_name": "first.pdf",
        "original_path": "uploads/user-1/doc-1/first.pdf",
    },
    {
        "doc_id": "doc-2",
        "file_name": "second.pdf",
        "original_path": "uploads/user-1/doc-2/second.pdf",
    },
]


def test_create_job_freezes_sources_and_round_trips(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)

    manifest = store.create_job(SOURCES)
    restored = store.load_current()

    assert restored is not None
    assert restored.job_id == manifest.job_id
    assert [document.doc_id for document in restored.documents] == ["doc-1", "doc-2"]
    assert restored.source_snapshot_hash == manifest.source_snapshot_hash
    assert restored.state == "pending"
    assert (tmp_path / manifest.job_id / "manifest.json").is_file()


def test_status_aggregates_terminal_document_counts(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = store.create_job(SOURCES)
    manifest.documents[0].state = "indexed"
    manifest.documents[1].state = "failed"
    store.save(manifest)

    status = store.to_status(manifest)

    assert status.total == 2
    assert status.processed == 2
    assert status.succeeded == 1
    assert status.failed == 1
    assert status.progress_percent == 100
    assert status.live_graph_unchanged is True
