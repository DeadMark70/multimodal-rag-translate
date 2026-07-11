"""Durable persistence primitives for GraphRAG full-rebuild jobs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
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
        graph_store = GraphStore(user_id)
        if rebuild_root is None:
            rebuild_root = graph_store.root_storage_dir / "rebuild_jobs"
        self.root = rebuild_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._legacy_root = (graph_store.storage_dir / "rebuild_jobs").resolve()
        self._migrate_legacy_current_job()
        self.lease_ttl = lease_ttl

    def create_job(
        self,
        sources: list[dict[str, str | None]],
        *,
        source_markdown: dict[str, str] | None = None,
    ) -> GraphRebuildManifest:
        """Freeze eligible sources into a newly created pending job."""
        if not sources:
            raise ValueError("A full rebuild requires at least one source document")

        normalized_sources = sorted(sources, key=lambda source: str(source["doc_id"]))
        documents = [GraphRebuildDocument(**source) for source in normalized_sources]
        if source_markdown is not None:
            expected_ids = {document.doc_id for document in documents}
            if set(source_markdown) != expected_ids:
                raise ValueError("Frozen GraphRAG sources do not match the rebuild documents")
            for document in documents:
                document.source_markdown_sha256 = self._content_hash(
                    source_markdown[document.doc_id]
                )
        snapshot = json.dumps(
            [document.model_dump(mode="json") for document in documents],
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
            documents=documents,
        )
        self._job_dir(manifest.job_id).mkdir(parents=True, exist_ok=False)
        if source_markdown is not None:
            for document in documents:
                self._write_source_markdown(
                    manifest.job_id, document.doc_id, source_markdown[document.doc_id]
                )
        self.save(manifest)
        return manifest

    def create_or_load_active(
        self,
        sources: list[dict[str, str | None]],
        *,
        source_markdown: dict[str, str] | None = None,
    ) -> tuple[GraphRebuildManifest, bool]:
        """Atomically create a job, unless another active rebuild already exists."""
        start_lock = self.root / "start.lock"
        try:
            descriptor = os.open(start_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = self.load_current()
            if current is None:
                raise RuntimeError("Another graph rebuild is being created")
            return current, False
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
                handle.flush()
                os.fsync(handle.fileno())
            current = self.load_current()
            if current is not None:
                current = self.reconcile_status(current)
            if current is not None and current.state not in {"completed", "failed"}:
                return current, False
            return self.create_job(sources, source_markdown=source_markdown), True
        finally:
            start_lock.unlink(missing_ok=True)

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

    def save(self, manifest: GraphRebuildManifest, *, preserve_lease: bool = True) -> None:
        """Atomically persist a manifest and mark it as the current job."""
        if manifest.user_id != self.user_id:
            raise ValueError("Cannot save a graph rebuild job for another user")
        existing = self._load_existing_manifest(manifest.job_id)
        if preserve_lease and existing is not None and existing.lease is not None:
            if manifest.lease is None or (
                manifest.lease.owner_token == existing.lease.owner_token
                and manifest.lease.heartbeat_at < existing.lease.heartbeat_at
            ):
                manifest.lease = existing.lease
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
            max_attempts=manifest.max_attempts,
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
            json.dump({"owner_token": owner_token, "process_id": os.getpid()}, handle)
            handle.flush()
            os.fsync(handle.fileno())

        manifest = self.load(job_id)
        now = self._now()
        manifest.lease = GraphRebuildLease(
            owner_token=owner_token,
            acquired_at=now,
            heartbeat_at=now,
            process_id=os.getpid(),
        )
        self.save(manifest, preserve_lease=False)
        return owner_token

    def heartbeat(self, job_id: str, owner_token: str) -> bool:
        """Refresh a lease heartbeat only when the caller still owns it."""
        manifest = self.load(job_id)
        if not self._has_owner(manifest, owner_token):
            return False
        manifest.lease.heartbeat_at = self._now()
        self.save(manifest, preserve_lease=False)
        return True

    def release_lease(self, job_id: str, owner_token: str) -> bool:
        """Release a runner slot without allowing a non-owner to unlock it."""
        manifest = self.load(job_id)
        if not self._has_owner(manifest, owner_token):
            return False
        lock_path = self._lock_path(job_id)
        if lock_path.exists() and self._lock_owner_token(lock_path) != owner_token:
            return False
        if lock_path.exists():
            lock_path.unlink()
        manifest.lease = None
        self.save(manifest, preserve_lease=False)
        return True

    def reconcile_status(self, manifest: Optional[GraphRebuildManifest]) -> Optional[GraphRebuildManifest]:
        """Turn a stale running job into manual-resume state without doing work."""
        if manifest is None or manifest.state != "running" or manifest.lease is None:
            return manifest
        lock_path = self._lock_path(manifest.job_id)
        lock_process_id = self._lock_process_id(lock_path) if lock_path.exists() else None
        if lock_process_id is not None and not self._process_is_alive(lock_process_id):
            stale = True
        else:
            stale = manifest.lease.heartbeat_at + self.lease_ttl < self._now()
        if not stale:
            return manifest

        if lock_path.exists() and self._lock_owner_token(lock_path) == manifest.lease.owner_token:
            lock_path.unlink()
        manifest.state = "interrupted"
        manifest.lease = None
        manifest.current_doc_id = None
        self.save(manifest, preserve_lease=False)
        return manifest

    def reset_failed_documents(self, manifest: GraphRebuildManifest) -> GraphRebuildManifest:
        """Prepare only failed/partial documents for a new retry cycle."""
        for document in manifest.documents:
            if document.state in {"failed", "partial"}:
                document.state = "pending"
                document.attempt = 0
                document.last_error = None
        manifest.state = "pending"
        manifest.phase = "extracting"
        manifest.current_doc_id = None
        manifest.last_error = None
        return manifest

    def staging_dir(self, job_id: str) -> Path:
        """Return the user-contained staging GraphStore directory for a job."""
        path = self._job_dir(job_id) / "staging_graph"
        self._assert_owned_path(path)
        return path

    def load_source_markdown(self, job_id: str, doc_id: str) -> str:
        """Load one frozen OCR markdown input and verify its recorded checksum."""
        manifest = self.load(job_id)
        document = next(
            (item for item in manifest.documents if item.doc_id == doc_id), None
        )
        if document is None or not document.source_markdown_sha256:
            raise ValueError("Graph rebuild source markdown is unavailable")
        source_path = self._source_path(job_id, doc_id)
        markdown = source_path.read_text(encoding="utf-8")
        if self._content_hash(markdown) != document.source_markdown_sha256:
            raise ValueError("Graph rebuild source markdown checksum does not match")
        return markdown

    def _job_dir(self, job_id: str) -> Path:
        if not job_id or Path(job_id).name != job_id:
            raise ValueError("Invalid graph rebuild job id")
        path = self.root / job_id
        self._assert_owned_path(path)
        return path

    def _manifest_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "manifest.json"

    def _load_existing_manifest(self, job_id: str) -> Optional[GraphRebuildManifest]:
        path = self._manifest_path(job_id)
        if not path.exists():
            return None
        return GraphRebuildManifest.model_validate(self._read_json(path))

    def _lock_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "runner.lock"

    def _current_path(self) -> Path:
        return self.root / "current.json"

    def _source_path(self, job_id: str, doc_id: str) -> Path:
        file_name = f"{hashlib.sha256(doc_id.encode('utf-8')).hexdigest()}.md"
        path = self._job_dir(job_id) / "sources" / file_name
        self._assert_owned_path(path)
        return path

    def _write_source_markdown(self, job_id: str, doc_id: str, markdown: str) -> None:
        path = self._source_path(job_id, doc_id)
        self._atomic_write_text(path, markdown)

    def _migrate_legacy_current_job(self) -> None:
        """Move a pre-release current job out of its mutable snapshot directory."""
        legacy_current = self._legacy_root / "current.json"
        if self._legacy_root == self.root or self._current_path().exists() or not legacy_current.exists():
            return
        pointer = self._read_json(legacy_current)
        job_id = pointer.get("job_id")
        if not isinstance(job_id, str):
            return
        legacy_job_dir = self._legacy_root / job_id
        if not legacy_job_dir.is_dir():
            return
        shutil.move(str(legacy_job_dir), str(self._job_dir(job_id)))
        self._atomic_write_json(self._current_path(), {"job_id": job_id})

    @staticmethod
    def _content_hash(markdown: str) -> str:
        return f"sha256:{hashlib.sha256(markdown.encode('utf-8')).hexdigest()}"

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
    def _lock_owner_token(path: Path) -> Optional[str]:
        """Read both current JSON locks and pre-release plain-token locks."""
        raw = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return payload.get("owner_token") if isinstance(payload, dict) else None

    @staticmethod
    def _lock_process_id(path: Path) -> Optional[int]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        process_id = payload.get("process_id") if isinstance(payload, dict) else None
        return process_id if isinstance(process_id, int) and process_id > 0 else None

    @staticmethod
    def _process_is_alive(process_id: int) -> bool:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            # Windows reports an invalid/nonexistent PID as WinError 87.
            return False
        return True

    @staticmethod
    def _atomic_write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
        temporary_path.replace(path)

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
