"""Durable GraphRAG full-rebuild job persistence tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from graph_rag.rebuild_jobs import GraphRebuildJobStore
from graph_rag.schemas import GraphRebuildLease


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


def test_only_one_store_can_acquire_runner_lease(tmp_path: Path) -> None:
    first = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = first.create_job(SOURCES)
    second = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)

    token = first.acquire_lease(manifest.job_id)

    assert token is not None
    assert second.acquire_lease(manifest.job_id) is None
    assert first.load(manifest.job_id).lease is not None


def test_stale_running_job_becomes_interrupted_without_starting_work(tmp_path: Path) -> None:
    store = GraphRebuildJobStore(
        "user-1",
        rebuild_root=tmp_path,
        lease_ttl=timedelta(seconds=30),
    )
    manifest = store.create_job(SOURCES)
    old_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manifest.state = "running"
    manifest.lease = GraphRebuildLease(
        owner_token="dead-runner",
        acquired_at=old_time,
        heartbeat_at=old_time,
    )
    store.save(manifest)
    (tmp_path / manifest.job_id / "runner.lock").write_text("dead-runner", encoding="utf-8")

    reconciled = store.reconcile_status(store.load_current())

    assert reconciled.state == "interrupted"
    assert reconciled.lease is None
    assert not (tmp_path / manifest.job_id / "runner.lock").exists()


def test_only_lease_owner_can_release_runner_lock(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = store.create_job(SOURCES)
    owner_token = store.acquire_lease(manifest.job_id)

    assert owner_token is not None
    assert store.release_lease(manifest.job_id, "wrong-owner") is False
    assert store.release_lease(manifest.job_id, owner_token) is True
    assert store.load(manifest.job_id).lease is None
