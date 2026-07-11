"""Durable persistence primitives for GraphRAG full-rebuild jobs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any, Optional
from uuid import uuid4

from graph_rag.schemas import (
    GraphRebuildDocument,
    GraphRebuildLease,
    GraphRebuildManifest,
    GraphRebuildStatusResponse,
)
from graph_rag.store import GraphStore


class GraphRebuildJobStore:
    """Persist one user's full-rebuild manifests under a constrained root."""

    def __init__(
        self,
        user_id: str,
        rebuild_root: Path | None = None,
        lease_ttl: timedelta = timedelta(minutes=2),
    ) -> None:
        self.user_id = user_id
        if rebuild_root is None:
            rebuild_root = GraphStore(user_id).storage_dir / "rebuild_jobs"
        self.root = rebuild_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lease_ttl = lease_ttl

    def create_job(self, sources: list[dict[str, str | None]]) -> GraphRebuildManifest:
        """Freeze eligible sources into a newly created pending job."""
        if not sources:
            raise ValueError("A full rebuild requires at least one source document")

        normalized_sources = sorted(sources, key=lambda source: str(source["doc_id"]))
        snapshot = json.dumps(
            normalized_sources,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = self._now()
        manifest = GraphRebuildManifest(
            job_id=str(uuid4()),
            user_id=self.user_id,
            created_at=now,
            updated_at=now,
            source_snapshot_hash=hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
            documents=[GraphRebuildDocument(**source) for source in normalized_sources],
        )
        self._job_dir(manifest.job_id).mkdir(parents=True, exist_ok=False)
        self.save(manifest)
        return manifest

    def load_current(self) -> Optional[GraphRebuildManifest]:
        """Load the current job pointer, if this user has one."""
        pointer_path = self._current_path()
        if not pointer_path.exists():
            return None
        pointer = self._read_json(pointer_path)
        job_id = pointer.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError("Graph rebuild current-job pointer is invalid")
        return self.load(job_id)

    def load(self, job_id: str) -> GraphRebuildManifest:
        """Load and validate one user-owned manifest."""
        manifest = GraphRebuildManifest.model_validate(
            self._read_json(self._manifest_path(job_id))
        )
        if manifest.user_id != self.user_id:
            raise ValueError("Graph rebuild job does not belong to this user")
        return manifest

    def save(self, manifest: GraphRebuildManifest) -> None:
        """Atomically persist a manifest and mark it as the current job."""
        if manifest.user_id != self.user_id:
            raise ValueError("Cannot save a graph rebuild job for another user")
        manifest.updated_at = self._now()
        self._atomic_write_json(
            self._manifest_path(manifest.job_id), manifest.model_dump(mode="json")
        )
        self._atomic_write_json(self._current_path(), {"job_id": manifest.job_id})

    def to_status(self, manifest: GraphRebuildManifest) -> GraphRebuildStatusResponse:
        """Project a manifest to frontend-safe aggregate progress."""
        counts = {state: 0 for state in ("indexed", "empty", "failed", "partial", "pending")}
        for document in manifest.documents:
            if document.state in counts:
                counts[document.state] += 1

        total = len(manifest.documents)
        processed = counts["indexed"] + counts["empty"] + counts["failed"] + counts["partial"]
        current_document = next(
            (document for document in manifest.documents if document.doc_id == manifest.current_doc_id),
            None,
        )
        public_documents = [self._public_document(document) for document in manifest.documents]
        return GraphRebuildStatusResponse(
            job_id=manifest.job_id,
            state=manifest.state,
            phase=manifest.phase,
            total=total,
            processed=processed,
            succeeded=counts["indexed"] + counts["empty"],
            empty=counts["empty"],
            failed=counts["failed"],
            partial=counts["partial"],
            pending=counts["pending"],
            progress_percent=round(processed * 100 / total) if total else 0,
            current_document=(
                self._public_document(current_document)
                if current_document is not None
                else None
            ),
            documents=public_documents,
            can_resume=manifest.state == "interrupted",
            can_retry_failed=manifest.state == "completed_with_failures",
            last_error=manifest.last_error,
        )

    def acquire_lease(self, job_id: str) -> Optional[str]:
        """Atomically claim the one runner slot for a persistent job."""
        lock_path = self._lock_path(job_id)
        owner_token = secrets.token_urlsafe(32)
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None

        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(owner_token)
            handle.flush()
            os.fsync(handle.fileno())

        manifest = self.load(job_id)
        now = self._now()
        manifest.lease = GraphRebuildLease(
            owner_token=owner_token,
            acquired_at=now,
            heartbeat_at=now,
        )
        self.save(manifest)
        return owner_token

    def heartbeat(self, job_id: str, owner_token: str) -> bool:
        """Refresh a lease heartbeat only when the caller still owns it."""
        manifest = self.load(job_id)
        if not self._has_owner(manifest, owner_token):
            return False
        manifest.lease.heartbeat_at = self._now()
        self.save(manifest)
        return True

    def release_lease(self, job_id: str, owner_token: str) -> bool:
        """Release a runner slot without allowing a non-owner to unlock it."""
        manifest = self.load(job_id)
        if not self._has_owner(manifest, owner_token):
            return False
        lock_path = self._lock_path(job_id)
        if lock_path.exists() and lock_path.read_text(encoding="utf-8") != owner_token:
            return False
        if lock_path.exists():
            lock_path.unlink()
        manifest.lease = None
        self.save(manifest)
        return True

    def reconcile_status(self, manifest: Optional[GraphRebuildManifest]) -> Optional[GraphRebuildManifest]:
        """Turn a stale running job into manual-resume state without doing work."""
        if manifest is None or manifest.state != "running" or manifest.lease is None:
            return manifest
        if manifest.lease.heartbeat_at + self.lease_ttl >= self._now():
            return manifest

        lock_path = self._lock_path(manifest.job_id)
        if lock_path.exists() and lock_path.read_text(encoding="utf-8") == manifest.lease.owner_token:
            lock_path.unlink()
        manifest.state = "interrupted"
        manifest.lease = None
        manifest.current_doc_id = None
        self.save(manifest)
        return manifest

    def staging_dir(self, job_id: str) -> Path:
        """Return the user-contained staging GraphStore directory for a job."""
        path = self._job_dir(job_id) / "staging_graph"
        self._assert_owned_path(path)
        return path

    def _job_dir(self, job_id: str) -> Path:
        if not job_id or Path(job_id).name != job_id:
            raise ValueError("Invalid graph rebuild job id")
        path = self.root / job_id
        self._assert_owned_path(path)
        return path

    def _manifest_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "manifest.json"

    def _lock_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "runner.lock"

    def _current_path(self) -> Path:
        return self.root / "current.json"

    def _assert_owned_path(self, path: Path) -> None:
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Graph rebuild path escapes the user's rebuild root")

    @staticmethod
    def _has_owner(manifest: GraphRebuildManifest, owner_token: str) -> bool:
        return manifest.lease is not None and secrets.compare_digest(
            manifest.lease.owner_token, owner_token
        )

    @staticmethod
    def _public_document(document: GraphRebuildDocument) -> GraphRebuildDocument:
        return document.model_copy(update={"original_path": None})

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Graph rebuild JSON payload must be an object")
        return payload

    @staticmethod
    def _atomic_write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
        temporary_path.replace(path)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
