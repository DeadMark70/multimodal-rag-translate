"""GraphRAG maintenance jobs shared by routers and document services."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import shutil
from pathlib import Path
from uuid import uuid4

from core import uploads as upload_paths
from graph_rag.node_vector_index import (
    node_vector_autosync_enabled,
    sync_node_vector_index,
)
from graph_rag.schemas import GraphDocumentStatus
from graph_rag.service import run_graph_extraction
from graph_rag.store import GraphStore
from pdfserviceMD.repository import get_document, list_documents as list_pdf_documents
from pdfserviceMD.service import load_ocr_artifacts

logger = logging.getLogger(__name__)


def _copy_graph_sidecars(src: GraphStore, dest: GraphStore) -> None:
    """Copy graph pickle and sidecars from one store location to another."""
    for source_path, target_path in (
        (src._get_graph_path(), dest._get_graph_path()),
        (src._get_metadata_path(), dest._get_metadata_path()),
        (src._get_document_status_path(), dest._get_document_status_path()),
        (src._get_extraction_runs_path(), dest._get_extraction_runs_path()),
        (src._get_provenance_path(), dest._get_provenance_path()),
        (src._get_raw_candidates_path(), dest._get_raw_candidates_path()),
        (src._get_asset_links_path(), dest._get_asset_links_path()),
        (src._get_aliases_path(), dest._get_aliases_path()),
        (src._get_type_index_path(), dest._get_type_index_path()),
        (src._get_doc_index_path(), dest._get_doc_index_path()),
        (src._get_node_vector_faiss_path(), dest._get_node_vector_faiss_path()),
        (src._get_node_vector_pickle_path(), dest._get_node_vector_pickle_path()),
        (src._get_node_vector_map_path(), dest._get_node_vector_map_path()),
        (src._get_node_vector_meta_path(), dest._get_node_vector_meta_path()),
    ):
        if source_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _replace_live_graph_files(temp_store: GraphStore, live_store: GraphStore) -> None:
    """Promote a completed temp rebuild through the live immutable snapshot pointer."""
    live_store.graph = temp_store.graph
    live_store.communities = temp_store.communities
    live_store.document_statuses = temp_store.document_statuses
    live_store.extraction_manifests = temp_store.extraction_manifests
    live_store.edge_provenance = temp_store.edge_provenance
    live_store.raw_candidates = temp_store.raw_candidates
    live_store.asset_links = temp_store.asset_links
    live_store.canonical_entities = temp_store.canonical_entities
    live_store.alias_index = temp_store.alias_index
    live_store.type_index = temp_store.type_index
    live_store.doc_index = temp_store.doc_index
    live_store.last_optimized_at = temp_store.last_optimized_at
    live_store.index_version = temp_store.index_version
    live_store.graph_dirty = temp_store.graph_dirty
    live_store.node_vector_dirty = temp_store.node_vector_dirty
    live_store.node_vector_sync = temp_store.node_vector_sync
    live_store.save_snapshot(node_vector_source=temp_store)


def _make_graph_work_dir(base_dir: Path, prefix: str) -> Path:
    """Create a writable unique workspace directory for graph maintenance jobs."""
    work_dir = base_dir / f"{prefix}{uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=False)
    return work_dir


def _parse_optional_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


async def list_graph_source_documents(user_id: str) -> list[dict[str, str | None]]:
    """List OCR-complete documents that can act as GraphRAG rebuild sources."""
    store = GraphStore(user_id)
    eligible_ids = store.list_eligible_document_ids()
    rows = await list_pdf_documents(
        user_id=user_id, limit=max(len(eligible_ids) + 50, 200)
    )
    row_map = {row["id"]: row for row in rows}
    sources: list[dict[str, str | None]] = []
    for doc_id in eligible_ids:
        row = row_map.get(doc_id, {})
        sources.append(
            {
                "doc_id": doc_id,
                "file_name": row.get("file_name"),
                "original_path": row.get("original_path"),
            }
        )
    return sources


async def node_vector_sync_task(user_id: str) -> None:
    """Run manual node-vector sync in background and persist progress."""
    logger.info("Starting node-vector sync for user %s", user_id)

    async def _progress_callback(payload: dict[str, object]) -> None:
        store = GraphStore(user_id)
        raw_state = str(payload.get("state", "running")).lower()
        state = (
            raw_state
            if raw_state in {"idle", "running", "completed", "failed"}
            else "running"
        )
        update_fields: dict[str, object] = {
            "state": state,
            "processed": int(payload.get("processed", 0) or 0),
            "total": int(payload.get("total", 0) or 0),
            "changed": int(payload.get("changed", 0) or 0),
            "reused": int(payload.get("reused", 0) or 0),
            "removed": int(payload.get("removed", 0) or 0),
        }
        if "index_state" in payload:
            update_fields["index_state"] = (
                str(payload.get("index_state"))
                if payload.get("index_state") is not None
                else None
            )
        if "autosync_duration_ms" in payload:
            update_fields["autosync_duration_ms"] = (
                int(payload.get("autosync_duration_ms", 0) or 0)
                if payload.get("autosync_duration_ms") is not None
                else None
            )
        if "last_error" in payload:
            update_fields["last_error"] = (
                str(payload.get("last_error"))
                if payload.get("last_error") is not None
                else None
            )
        if "started_at" in payload:
            update_fields["started_at"] = _parse_optional_datetime(
                payload.get("started_at")
            )
        if "updated_at" in payload:
            update_fields["updated_at"] = _parse_optional_datetime(
                payload.get("updated_at")
            )
        if "finished_at" in payload:
            update_fields["finished_at"] = _parse_optional_datetime(
                payload.get("finished_at")
            )

        store.set_node_vector_sync_status(
            state=state,
            **{k: v for k, v in update_fields.items() if k != "state"},
        )
        store.save_sidecars()

    try:
        await sync_node_vector_index(
            user_id=user_id,
            progress_callback=_progress_callback,
        )
        logger.info("Node-vector sync complete for user %s", user_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Node-vector sync failed for user %s: %s", user_id, exc, exc_info=True
        )
        store = GraphStore(user_id)
        now = datetime.now(timezone.utc)
        store.set_node_vector_sync_status(
            state="failed",
            last_error=str(exc),
            updated_at=now,
            finished_at=now,
            index_state="failed",
        )
        store.save_sidecars()
    finally:
        store = GraphStore(user_id)
        store.set_active_job_state(None)
        store.save_sidecars()


async def optimize_existing_graph(
    store: GraphStore,
    *,
    regenerate_communities: bool = True,
) -> tuple[int, int]:
    """Run entity resolution and optionally rebuild communities on an existing graph."""
    from graph_rag.entity_resolver import resolve_entities

    merges = await resolve_entities(store)
    communities_count = 0

    if regenerate_communities:
        from graph_rag.community_builder import build_communities

        communities = await build_communities(store, generate_summaries=True)
        communities_count = len(communities)

    store.save_snapshot()
    if node_vector_autosync_enabled() and isinstance(store, GraphStore):
        sync_result = await sync_node_vector_index(
            user_id=store.user_id,
            store=store,
        )
        logger.info(
            "Graph maintenance node-vector sync result | user_id=%s | index_state=%s | autosync_duration_ms=%s",
            store.user_id,
            sync_result.get("index_state"),
            sync_result.get("autosync_duration_ms"),
        )
    return merges, communities_count


async def rebuild_graph_task(user_id: str) -> None:
    """
    Background task to safely rebuild graph metadata and communities.

    This does not re-extract entities/relations from source documents.
    """
    logger.info(f"Starting graph rebuild for user {user_id}")

    try:
        store = GraphStore(user_id)

        if store.get_status().node_count == 0:
            logger.info(
                "Skipping graph rebuild for user %s because graph is empty", user_id
            )
            return

        await optimize_existing_graph(store, regenerate_communities=True)

        logger.info(f"Graph rebuild complete for user {user_id}")

    except Exception as e:
        logger.error(f"Graph rebuild failed for user {user_id}: {e}")


async def rebuild_full_graph_task(user_id: str) -> None:
    """Build a brand-new graph from all OCR-complete document artifacts."""
    logger.info("Starting full graph rebuild for user %s", user_id)
    live_store = GraphStore(user_id)
    sources = await list_graph_source_documents(user_id)
    temp_dir = _make_graph_work_dir(live_store.storage_dir.parent, "graph-rebuild-")

    try:
        temp_store = GraphStore(user_id, storage_dir=temp_dir)
        temp_store.clear()
        temp_store.save()

        for source in sources:
            doc_id = str(source["doc_id"])
            original_path = source.get("original_path")
            user_folder = (
                Path(original_path).resolve().parent
                if original_path
                else Path(upload_paths.get_document_upload_dir(user_id, doc_id))
            )

            try:
                markdown_text, _ = await asyncio.to_thread(
                    load_ocr_artifacts,
                    user_folder=str(user_folder),
                )
            except Exception as exc:  # noqa: BLE001
                temp_store.upsert_document_status(
                    GraphDocumentStatus(
                        doc_id=doc_id,
                        status="failed",
                        last_error=str(exc),
                    )
                )
                temp_store.save_sidecars()
                logger.warning(
                    "Failed to load OCR artifacts for %s during full rebuild: %s",
                    doc_id,
                    exc,
                )
                continue

            await run_graph_extraction(
                user_id=user_id,
                doc_id=doc_id,
                markdown_text=markdown_text,
                store=temp_store,
                autosync=False,
            )

        blocking_failures = [
            status
            for status in temp_store.get_all_document_statuses()
            if status.status in {"failed", "partial"}
        ]

        if not blocking_failures:
            await optimize_existing_graph(temp_store, regenerate_communities=True)
            temp_store.set_active_job_state(None)
            temp_store.save_sidecars()
            _replace_live_graph_files(temp_store, live_store)
            logger.info("Full graph rebuild complete for user %s", user_id)
        else:
            live_store.document_statuses = {
                status.doc_id: status
                for status in temp_store.get_all_document_statuses()
            }
            live_store.save_sidecars()
            logger.warning(
                "Full graph rebuild for user %s kept old graph because %s document(s) failed or were partial",
                user_id,
                len(blocking_failures),
            )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Full graph rebuild failed for user %s: %s", user_id, exc, exc_info=True
        )
        if "temp_store" in locals():
            live_store.document_statuses = {
                status.doc_id: status
                for status in temp_store.get_all_document_statuses()
            }
            live_store.save_sidecars()
    finally:
        live_store = GraphStore(user_id)
        live_store.set_active_job_state(None)
        live_store.save_sidecars()
        shutil.rmtree(temp_dir, ignore_errors=True)


async def retry_graph_document_task(user_id: str, doc_id: str) -> None:
    """Retry GraphRAG extraction for one document using a temp copy of the live graph."""
    logger.info("Starting graph retry for user %s doc %s", user_id, doc_id)
    live_store = GraphStore(user_id)
    temp_dir = _make_graph_work_dir(
        live_store.storage_dir.parent, f"graph-retry-{doc_id}-"
    )

    try:
        _copy_graph_sidecars(live_store, GraphStore(user_id, storage_dir=temp_dir))
        temp_store = GraphStore(user_id, storage_dir=temp_dir)
        temp_store.remove_document(doc_id)
        temp_store.remove_document_status(doc_id)
        temp_store.save()

        row = await get_document(
            doc_id=doc_id,
            user_id=user_id,
            columns="original_path",
        )
        original_path = row.get("original_path") if row else None
        user_folder = (
            Path(original_path).resolve().parent
            if original_path
            else Path(upload_paths.get_document_upload_dir(user_id, doc_id))
        )
        markdown_text, _ = await asyncio.to_thread(
            load_ocr_artifacts,
            user_folder=str(user_folder),
        )

        result = await run_graph_extraction(
            user_id=user_id,
            doc_id=doc_id,
            markdown_text=markdown_text,
            store=temp_store,
            autosync=False,
        )

        if result.status in {"failed", "partial"}:
            live_store.upsert_document_status(
                GraphDocumentStatus(
                    doc_id=doc_id,
                    status=result.status,
                    chunk_count=result.chunk_count,
                    chunks_succeeded=result.chunks_succeeded,
                    chunks_failed=result.chunks_failed,
                    entities_added=result.entities_added,
                    edges_added=result.edges_added,
                    last_error=result.last_error,
                )
            )
            live_store.save_sidecars()
            logger.warning(
                "Graph retry for %s failed or was partial; preserving live graph",
                doc_id,
            )
            return

        await optimize_existing_graph(temp_store, regenerate_communities=True)
        temp_store.set_active_job_state(None)
        temp_store.save_sidecars()
        _replace_live_graph_files(temp_store, live_store)
        logger.info("Graph retry complete for user %s doc %s", user_id, doc_id)

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Graph retry failed for user %s doc %s: %s",
            user_id,
            doc_id,
            exc,
            exc_info=True,
        )
        live_store.upsert_document_status(
            GraphDocumentStatus(
                doc_id=doc_id,
                status="failed",
                last_error=str(exc),
            )
        )
        live_store.save_sidecars()
    finally:
        live_store = GraphStore(user_id)
        live_store.set_active_job_state(None)
        live_store.save_sidecars()
        shutil.rmtree(temp_dir, ignore_errors=True)


async def purge_graph_document_task(user_id: str, doc_id: str) -> None:
    """Safely purge one document's remaining contribution from the live graph."""
    logger.info("Starting graph purge for user %s doc %s", user_id, doc_id)
    live_store = GraphStore(user_id)
    temp_dir = _make_graph_work_dir(
        live_store.storage_dir.parent, f"graph-purge-{doc_id}-"
    )

    try:
        _copy_graph_sidecars(live_store, GraphStore(user_id, storage_dir=temp_dir))
        temp_store = GraphStore(user_id, storage_dir=temp_dir)
        temp_store.remove_document(doc_id)
        temp_store.remove_document_status(doc_id)

        if temp_store.get_status().has_graph:
            await optimize_existing_graph(temp_store, regenerate_communities=True)
        else:
            temp_store.communities.clear()
            temp_store.graph_dirty = False
            temp_store.last_optimized_at = None
            temp_store.save()

        temp_store.set_active_job_state(None)
        temp_store.save_sidecars()
        _replace_live_graph_files(temp_store, live_store)
        logger.info("Graph purge complete for user %s doc %s", user_id, doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Graph purge failed for user %s doc %s: %s",
            user_id,
            doc_id,
            exc,
            exc_info=True,
        )
        live_store.upsert_document_status(
            GraphDocumentStatus(
                doc_id=doc_id,
                status="failed",
                last_error=f"Graph purge failed: {exc}",
            )
        )
        live_store.save_sidecars()
    finally:
        live_store = GraphStore(user_id)
        live_store.set_active_job_state(None)
        live_store.save_sidecars()
        shutil.rmtree(temp_dir, ignore_errors=True)
