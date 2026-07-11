"""Stable, per-user mutual exclusion for graph-mutating maintenance work."""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
from typing import Optional

from graph_rag.store import GraphStore


class GraphMaintenanceLock:
    """Own the single graph-maintenance slot outside immutable graph snapshots."""

    def __init__(self, user_id: str, *, lock_root: Path | None = None) -> None:
        self.user_id = user_id
        self.root = (lock_root or GraphStore(user_id).root_storage_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "maintenance.lock"

    def acquire(self, activity: str) -> Optional[str]:
        """Atomically acquire the maintenance slot, reclaiming a dead process lock."""
        owner_token = secrets.token_urlsafe(32)
        payload = {
            "owner_token": owner_token,
            "activity": activity,
            "process_id": os.getpid(),
        }
        for _ in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not self._reclaim_dead_owner():
                    return None
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            return owner_token
        return None

    def release(self, owner_token: str) -> bool:
        """Release only a lock owned by this caller."""
        payload = self._read()
        if payload is None or not secrets.compare_digest(
            str(payload.get("owner_token", "")), owner_token
        ):
            return False
        self.path.unlink(missing_ok=True)
        return True

    def current_activity(self) -> Optional[str]:
        """Return the current activity label, reclaiming locks from dead processes."""
        payload = self._read()
        if payload is None:
            return None
        process_id = payload.get("process_id")
        if isinstance(process_id, int) and not self._process_is_alive(process_id):
            self.path.unlink(missing_ok=True)
            return None
        activity = payload.get("activity")
        return activity if isinstance(activity, str) else None

    def _reclaim_dead_owner(self) -> bool:
        payload = self._read()
        if payload is None:
            self.path.unlink(missing_ok=True)
            return True
        process_id = payload.get("process_id")
        if isinstance(process_id, int) and not self._process_is_alive(process_id):
            self.path.unlink(missing_ok=True)
            return True
        return False

    def _read(self) -> Optional[dict[str, object]]:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _process_is_alive(process_id: int) -> bool:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
