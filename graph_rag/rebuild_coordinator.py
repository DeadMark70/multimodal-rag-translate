"""Sequential document checkpointing for durable GraphRAG rebuild jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from graph_rag.rebuild_jobs import GraphRebuildJobStore
from graph_rag.schemas import (
    GraphExtractionRunResult,
    GraphRebuildDocument,
    GraphRebuildManifest,
)
from graph_rag.service import run_graph_extraction
from graph_rag.store import GraphStore
from pdfserviceMD.service import load_ocr_artifacts


ExtractionRunner = Callable[..., Awaitable[GraphExtractionRunResult]]
ArtifactLoader = Callable[..., tuple[str, object]]
StoreFactory = Callable[..., GraphStore]
Sleep = Callable[[float], Awaitable[None]]


class GraphRebuildCoordinator:
    """Run only the extract/checkpoint phase of a full graph rebuild."""

    retry_delays = (5.0, 20.0)

    def __init__(
        self,
        jobs: GraphRebuildJobStore,
        *,
        store_factory: StoreFactory = GraphStore,
        run_extraction: ExtractionRunner = run_graph_extraction,
        load_artifacts: ArtifactLoader = load_ocr_artifacts,
        sleep: Sleep = asyncio.sleep,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        self.jobs = jobs
        self.store_factory = store_factory
        self.run_extraction = run_extraction
        self.load_artifacts = load_artifacts
        self.sleep = sleep
        self.jitter = jitter or (lambda: 0.0)

    async def run(self, user_id: str, job_id: str, owner_token: str) -> None:
        """Extract all incomplete documents and persist a checkpoint after each one."""
        manifest = self.jobs.load(job_id)
        self._require_runner(manifest, owner_token)
        staging = self.store_factory(user_id, storage_dir=self.jobs.staging_dir(job_id))
        self._reset_interrupted_document(manifest, staging)
        manifest.state = "running"
        manifest.phase = "extracting"
        self.jobs.save(manifest)

        for document in manifest.documents:
            if document.state in {"indexed", "empty", "failed", "partial"}:
                continue
            await self._process_document(manifest, document, staging, owner_token)

        self._complete_extraction_phase(manifest)

    async def _process_document(
        self,
        manifest: GraphRebuildManifest,
        document: GraphRebuildDocument,
        staging: GraphStore,
        owner_token: str,
    ) -> None:
        while document.attempt < manifest.max_attempts:
            self._require_runner(manifest, owner_token)
            self.jobs.heartbeat(manifest.job_id, owner_token)
            document.attempt += 1
            document.cumulative_attempts += 1
            document.state = "running"
            manifest.current_doc_id = document.doc_id
            self.jobs.save(manifest)
            try:
                markdown_text, _ = await asyncio.to_thread(
                    self.load_artifacts,
                    user_folder=str(self._document_folder(document)),
                )
                result = await self.run_extraction(
                    user_id=manifest.user_id,
                    doc_id=document.doc_id,
                    markdown_text=markdown_text,
                    store=staging,
                    autosync=False,
                    extraction_profile="standard",
                )
            except Exception as exc:  # noqa: BLE001
                if self._is_retryable(exc) and document.attempt < manifest.max_attempts:
                    document.state = "retry_wait"
                    document.last_error = self._safe_error(exc)
                    manifest.phase = "retry_wait"
                    self.jobs.save(manifest)
                    delay = self.retry_delays[document.attempt - 1] + self.jitter()
                    await self.sleep(max(delay, 0.0))
                    manifest.phase = "extracting"
                    continue
                document.state = "failed"
                document.last_error = self._safe_error(exc)
                self.jobs.save(manifest)
                return

            self._apply_result(document, result)
            staging.save()
            if document.state in {"indexed", "empty"}:
                self._validate_document_checkpoint(staging, document.doc_id)
            self.jobs.save(manifest)
            return

    def _complete_extraction_phase(self, manifest: GraphRebuildManifest) -> None:
        manifest.current_doc_id = None
        if any(document.state in {"failed", "partial"} for document in manifest.documents):
            manifest.state = "completed_with_failures"
            manifest.phase = "done"
        else:
            manifest.state = "running"
            manifest.phase = "optimizing"
        self.jobs.save(manifest)

    @staticmethod
    def _document_folder(document: GraphRebuildDocument) -> Path:
        if not document.original_path:
            raise FileNotFoundError(f"Missing OCR artifact path for document {document.doc_id}")
        return Path(document.original_path).resolve().parent

    @staticmethod
    def _apply_result(document: GraphRebuildDocument, result: GraphExtractionRunResult) -> None:
        document.state = result.status
        document.chunk_count = result.chunk_count
        document.chunks_succeeded = result.chunks_succeeded
        document.chunks_failed = result.chunks_failed
        document.entities_added = result.entities_added
        document.edges_added = result.edges_added
        document.last_error = result.last_error

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        return isinstance(exc, (TimeoutError, httpx.TransportError)) or getattr(exc, "status_code", None) == 429

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        return str(exc).replace("\n", " ")[:500]

    @staticmethod
    def _validate_document_checkpoint(staging: GraphStore, doc_id: str) -> None:
        if doc_id not in staging.get_documents() or staging.get_latest_extraction_manifest(doc_id) is None:
            raise RuntimeError(f"Staging graph checkpoint is incomplete for document {doc_id}")

    @staticmethod
    def _reset_interrupted_document(manifest: GraphRebuildManifest, staging: GraphStore) -> None:
        for document in manifest.documents:
            if document.state == "running":
                staging.remove_document(document.doc_id)
                staging.remove_document_status(document.doc_id)
                document.state = "pending"
        staging.save()

    @staticmethod
    def _require_runner(manifest: GraphRebuildManifest, owner_token: str) -> None:
        if manifest.lease is None or manifest.lease.owner_token != owner_token:
            raise RuntimeError("Graph rebuild runner lease is no longer owned by this task")
