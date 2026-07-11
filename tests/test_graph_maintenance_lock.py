"""Durable mutual exclusion tests for graph-maintenance operations."""

from pathlib import Path

from graph_rag.maintenance_lock import GraphMaintenanceLock


def test_only_one_maintenance_operation_can_hold_stable_lock(tmp_path: Path) -> None:
    first = GraphMaintenanceLock("user-1", lock_root=tmp_path)
    second = GraphMaintenanceLock("user-1", lock_root=tmp_path)

    owner_token = first.acquire("rebuild_full")

    assert owner_token is not None
    assert second.acquire("optimize") is None
    assert first.release(owner_token) is True
    assert second.acquire("optimize") is not None


def test_non_owner_cannot_release_maintenance_lock(tmp_path: Path) -> None:
    lock = GraphMaintenanceLock("user-1", lock_root=tmp_path)
    owner_token = lock.acquire("retry:doc-1")

    assert owner_token is not None
    assert lock.release("wrong-owner") is False
    assert lock.current_activity() == "retry:doc-1"
