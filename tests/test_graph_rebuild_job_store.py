"""Durable GraphRAG full-rebuild job persistence tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from graph_rag.rebuild_jobs import GraphRebuildJobStore
from graph_rag.schemas import GraphRebuildLease
from graph_rag.store import GraphStore


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
    assert restored.documents[0].original_path == "uploads/user-1/doc-1/first.pdf"
    assert restored.source_snapshot_hash == manifest.source_snapshot_hash
    assert restored.state == "pending"
    assert (tmp_path / manifest.job_id / "manifest.json").is_file()


def test_default_job_root_survives_live_snapshot_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    graph_root = tmp_path / "live_graph"
    monkeypatch.setattr(
        "graph_rag.store.upload_paths.get_rag_index_dir_path",
        lambda _user_id: graph_root,
    )
    GraphStore("user-1").save_snapshot()
    jobs = GraphRebuildJobStore("user-1")
    manifest = jobs.create_job(SOURCES)

    GraphStore("user-1").save_snapshot()
    reopened = GraphRebuildJobStore("user-1")

    assert reopened.load_current() is not None
    assert reopened.load_current().job_id == manifest.job_id


def test_job_persists_and_verifies_frozen_source_markdown(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    source_markdown = {"doc-1": "frozen first", "doc-2": "frozen second"}

    manifest = store.create_job(SOURCES, source_markdown=source_markdown)

    assert manifest.documents[0].source_markdown_sha256 is not None
    assert store.load_source_markdown(manifest.job_id, "doc-1") == "frozen first"


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


def test_status_reports_live_graph_changed_only_after_publication(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    running = store.create_job(SOURCES)
    running.state = "running"
    completed = running.model_copy(update={"state": "completed"})

    assert store.to_status(running).live_graph_unchanged is True
    assert store.to_status(completed).live_graph_unchanged is False


def test_status_exposes_manifest_retry_limit(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = store.create_job(SOURCES)
    manifest.max_attempts = 5

    assert store.to_status(manifest).max_attempts == 5


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


def test_dead_process_job_becomes_interrupted_without_waiting_for_ttl(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = store.create_job(SOURCES)
    now = datetime.now(timezone.utc)
    manifest.state = "running"
    manifest.lease = GraphRebuildLease(
        owner_token="dead-process",
        acquired_at=now,
        heartbeat_at=now,
        process_id=999_999_999,
    )
    store.save(manifest)
    (tmp_path / manifest.job_id / "runner.lock").write_text(
        '{"owner_token":"dead-process","process_id":999999999}', encoding="utf-8"
    )

    reconciled = store.reconcile_status(store.load_current())

    assert reconciled is not None
    assert reconciled.state == "interrupted"
    assert reconciled.lease is None


def test_only_lease_owner_can_release_runner_lock(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = store.create_job(SOURCES)
    owner_token = store.acquire_lease(manifest.job_id)

    assert owner_token is not None
    assert store.release_lease(manifest.job_id, "wrong-owner") is False
    assert store.release_lease(manifest.job_id, owner_token) is True
    assert store.load(manifest.job_id).lease is None


def test_saving_stale_manifest_does_not_erase_newer_lease_heartbeat(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = store.create_job(SOURCES)
    owner_token = store.acquire_lease(manifest.job_id)
    assert owner_token is not None
    stale_manifest = store.load(manifest.job_id)
    assert stale_manifest.lease is not None
    stale_manifest.lease.heartbeat_at = datetime(2020, 1, 1, tzinfo=timezone.utc)

    assert store.heartbeat(manifest.job_id, owner_token) is True
    store.save(stale_manifest)

    restored = store.load(manifest.job_id)
    assert restored.lease is not None
    assert restored.lease.heartbeat_at > datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_reset_failed_documents_keeps_successful_checkpoints(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = store.create_job(SOURCES)
    manifest.state = "completed_with_failures"
    manifest.documents[0].state = "indexed"
    manifest.documents[1].state = "failed"
    manifest.documents[1].attempt = 3
    manifest.documents[1].last_error = "quota exceeded"

    reset = store.reset_failed_documents(manifest)

    assert reset.state == "pending"
    assert reset.phase == "extracting"
    assert reset.documents[0].state == "indexed"
    assert reset.documents[1].state == "pending"
    assert reset.documents[1].attempt == 0
    assert reset.documents[1].last_error is None
