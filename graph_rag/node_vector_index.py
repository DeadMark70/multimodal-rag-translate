"""GraphRAG node-vector indexing and search utilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from langchain_community.vectorstores import FAISS

from data_base.vector_store_manager import get_embeddings, initialize_embeddings
from graph_rag.schemas import GraphNode
from graph_rag.store import GraphStore, NODE_VECTOR_INDEX_NAME

logger = logging.getLogger(__name__)

_NODE_VECTOR_MAP_VERSION = 1
_VECTOR_SEARCH_ENABLED_ENV = "GRAPH_NODE_VECTOR_SEARCH_ENABLED"
_VECTOR_AUTOSYNC_ENABLED_ENV = "GRAPH_NODE_VECTOR_AUTOSYNC"
_VECTOR_TOP_K_ENV = "GRAPH_NODE_VECTOR_TOP_K"
_VECTOR_MIN_SCORE_ENV = "GRAPH_NODE_VECTOR_MIN_SCORE"
_VECTOR_BATCH_SIZE_ENV = "GRAPH_NODE_VECTOR_BATCH_SIZE"
_NODE_VECTOR_SYNC_LOCKS: dict[str, asyncio.Lock] = {}
_NODE_VECTOR_SYNC_LOCKS_GUARD = asyncio.Lock()


@dataclass(slots=True)
class NodeVectorSearchResult:
    """Result for node-vector local-search seed retrieval."""

    node_ids: list[str]
    vector_hit_count: int
    index_state: str
    fallback_reason: Optional[str] = None
    top_score: Optional[float] = None


def _is_true(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, 1)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.0, min(1.0, value))


def node_vector_search_enabled() -> bool:
    """Return whether node-vector retrieval is enabled."""
    return _is_true(os.getenv(_VECTOR_SEARCH_ENABLED_ENV), default=True)


def node_vector_autosync_enabled() -> bool:
    """Return whether node-vector autosync is enabled."""
    return _is_true(os.getenv(_VECTOR_AUTOSYNC_ENABLED_ENV), default=True)


def node_vector_top_k() -> int:
    """Return configured top-k for node-vector seed retrieval."""
    return _int_env(_VECTOR_TOP_K_ENV, default=12)


def node_vector_min_score() -> float:
    """Return configured min-score threshold for node-vector matches."""
    return _float_env(_VECTOR_MIN_SCORE_ENV, default=0.35)


def node_vector_batch_size() -> int:
    """Return configured embedding batch size for autosync."""
    return _int_env(_VECTOR_BATCH_SIZE_ENV, default=64)


def mark_node_vector_dirty(store: GraphStore) -> None:
    """Mark the node-vector sidecar index as stale."""
    store.mark_node_vector_dirty()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_node_embedding_text(node: GraphNode) -> str:
    description = (node.description or "").strip()
    if description:
        return f"{node.label} [{node.entity_type.value}] {description}"
    return f"{node.label} [{node.entity_type.value}]"


def _node_signature(node: GraphNode) -> str:
    payload = "|".join(
        [
            node.label.strip().lower(),
            node.entity_type.value.strip().lower(),
            (node.description or "").strip().lower(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Failed to load node-vector JSON sidecar %s: %s", path, exc)
        return {}


def _remove_node_vector_files(store: GraphStore) -> None:
    for path in (
        store._get_node_vector_faiss_path(),
        store._get_node_vector_pickle_path(),
        store._get_node_vector_map_path(),
        store._get_node_vector_meta_path(),
    ):
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning("Failed to remove stale node-vector file %s: %s", path, exc)


async def _get_node_vector_sync_lock(user_id: str) -> asyncio.Lock:
    async with _NODE_VECTOR_SYNC_LOCKS_GUARD:
        return _NODE_VECTOR_SYNC_LOCKS.setdefault(user_id, asyncio.Lock())


async def _ensure_embeddings_model():
    model = get_embeddings()
    if model is not None:
        return model
    try:
        await initialize_embeddings()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Node-vector embedding init failed: %s", exc)
        return None
    return get_embeddings()


async def _embed_texts(model, texts: list[str], batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        batch_vectors: Optional[list[list[float]]] = None
        if hasattr(model, "aembed_documents"):
            try:
                batch_vectors = await model.aembed_documents(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Batch node-vector embedding failed; falling back to per-query: %s", exc)
        if batch_vectors is None:
            batch_vectors = []
            for text in chunk:
                batch_vectors.append(await model.aembed_query(text))
        vectors.extend([list(map(float, vector)) for vector in batch_vectors])
    return vectors


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


async def sync_node_vector_index(
    *,
    user_id: str,
    store: GraphStore | None = None,
) -> dict[str, Any]:
    """Build or refresh local node-vector sidecars for one user graph."""
    lock = await _get_node_vector_sync_lock(user_id)
    async with lock:
        started = time.perf_counter()
        active_store = store or GraphStore(user_id)

        try:
            nodes = active_store.get_all_nodes()
            if not nodes:
                _remove_node_vector_files(active_store)
                active_store.clear_node_vector_dirty()
                active_store.save_sidecars()
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                logger.info(
                    "Node-vector autosync complete for user %s | index_state=%s | autosync_duration_ms=%s",
                    user_id,
                    "empty",
                    elapsed_ms,
                )
                return {
                    "index_state": "empty",
                    "node_count": 0,
                    "changed_count": 0,
                    "reused_count": 0,
                    "removed_count": 0,
                    "autosync_duration_ms": elapsed_ms,
                }

            embeddings_model = await _ensure_embeddings_model()
            if embeddings_model is None:
                raise RuntimeError("embedding model unavailable")

            map_payload = _load_json(active_store._get_node_vector_map_path())
            previous_nodes = map_payload.get("nodes", {})
            if not isinstance(previous_nodes, dict):
                previous_nodes = {}

            node_records: dict[str, dict[str, Any]] = {}
            changed_node_ids: list[str] = []
            changed_texts: list[str] = []
            reused_count = 0

            for node in nodes:
                signature = _node_signature(node)
                text = _build_node_embedding_text(node)
                previous = previous_nodes.get(node.id)
                if (
                    isinstance(previous, dict)
                    and previous.get("signature") == signature
                    and isinstance(previous.get("embedding"), list)
                    and previous.get("embedding")
                ):
                    node_records[node.id] = {
                        "signature": signature,
                        "text": text,
                        "embedding": [float(value) for value in previous.get("embedding", [])],
                    }
                    reused_count += 1
                else:
                    changed_node_ids.append(node.id)
                    changed_texts.append(text)

            if changed_texts:
                batch_size = node_vector_batch_size()
                new_vectors = await _embed_texts(embeddings_model, changed_texts, batch_size)
                if len(new_vectors) != len(changed_node_ids):
                    raise RuntimeError("embedding count mismatch while syncing node vector index")
                for node_id, text, vector in zip(changed_node_ids, changed_texts, new_vectors):
                    current_node = active_store.get_node(node_id)
                    if current_node is None:
                        continue
                    node_records[node_id] = {
                        "signature": _node_signature(current_node),
                        "text": text,
                        "embedding": vector,
                    }

            removed_count = len([node_id for node_id in previous_nodes.keys() if node_id not in node_records])

            ordered_node_ids = sorted(node_records.keys())
            text_embeddings = [
                (node_records[node_id]["text"], node_records[node_id]["embedding"])
                for node_id in ordered_node_ids
            ]
            metadatas = [
                {
                    "node_id": node_id,
                    "signature": node_records[node_id]["signature"],
                }
                for node_id in ordered_node_ids
            ]

            faiss_store = await asyncio.to_thread(
                FAISS.from_embeddings,
                text_embeddings,
                embeddings_model,
                metadatas=metadatas,
            )
            await asyncio.to_thread(
                faiss_store.save_local,
                str(active_store.storage_dir),
                NODE_VECTOR_INDEX_NAME,
            )

            map_payload = {
                "version": _NODE_VECTOR_MAP_VERSION,
                "index_name": NODE_VECTOR_INDEX_NAME,
                "updated_at": _utc_now_iso(),
                "nodes": {node_id: node_records[node_id] for node_id in ordered_node_ids},
            }
            _write_json(active_store._get_node_vector_map_path(), map_payload)

            vector_dim = len(node_records[ordered_node_ids[0]]["embedding"]) if ordered_node_ids else 0
            meta_payload = {
                "index_name": NODE_VECTOR_INDEX_NAME,
                "index_state": "ready",
                "updated_at": _utc_now_iso(),
                "node_count": len(ordered_node_ids),
                "changed_count": len(changed_node_ids),
                "reused_count": reused_count,
                "removed_count": removed_count,
                "vector_dim": vector_dim,
                "embedding_backend": type(embeddings_model).__name__,
                "last_error": None,
            }
            _write_json(active_store._get_node_vector_meta_path(), meta_payload)

            active_store.clear_node_vector_dirty()
            active_store.save_sidecars()

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "Node-vector autosync complete for user %s | index_state=%s | changed=%s | reused=%s | removed=%s | autosync_duration_ms=%s",
                user_id,
                "ready",
                len(changed_node_ids),
                reused_count,
                removed_count,
                elapsed_ms,
            )
            meta_payload["autosync_duration_ms"] = elapsed_ms
            return meta_payload

        except Exception as exc:  # noqa: BLE001
            active_store.mark_node_vector_dirty()
            active_store.save_sidecars()
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            meta_payload = {
                "index_name": NODE_VECTOR_INDEX_NAME,
                "index_state": "failed",
                "updated_at": _utc_now_iso(),
                "last_error": str(exc),
                "autosync_duration_ms": elapsed_ms,
            }
            _write_json(active_store._get_node_vector_meta_path(), meta_payload)
            logger.warning(
                "Node-vector autosync failed for user %s | index_state=%s | autosync_duration_ms=%s | error=%s",
                user_id,
                "failed",
                elapsed_ms,
                exc,
            )
            return meta_payload


def schedule_node_vector_autosync(
    *,
    user_id: str,
    store: GraphStore | None = None,
    reason: str = "",
) -> bool:
    """Schedule non-blocking node-vector autosync on the current event loop."""
    if not node_vector_autosync_enabled():
        return False
    if store is not None and not isinstance(store, GraphStore):
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    task = loop.create_task(sync_node_vector_index(user_id=user_id, store=store))

    def _done_callback(done_task: asyncio.Task) -> None:
        try:
            result = done_task.result()
            logger.debug(
                "Scheduled node-vector autosync finished for user %s | reason=%s | index_state=%s",
                user_id,
                reason or "unspecified",
                result.get("index_state"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Scheduled node-vector autosync crashed for user %s | reason=%s | error=%s",
                user_id,
                reason or "unspecified",
                exc,
            )

    task.add_done_callback(_done_callback)
    return True


async def search_nodes_by_vector(
    *,
    store: GraphStore,
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> NodeVectorSearchResult:
    """Retrieve candidate graph node ids from local node-vector sidecars."""
    if not node_vector_search_enabled():
        return NodeVectorSearchResult(
            node_ids=[],
            vector_hit_count=0,
            index_state="disabled",
            fallback_reason="vector_search_disabled",
        )

    normalized_query = query.strip()
    if not normalized_query:
        return NodeVectorSearchResult(
            node_ids=[],
            vector_hit_count=0,
            index_state="skipped",
            fallback_reason="empty_query",
        )

    faiss_path = store._get_node_vector_faiss_path()
    pickle_path = store._get_node_vector_pickle_path()
    map_path = store._get_node_vector_map_path()
    if not faiss_path.exists() or not pickle_path.exists() or not map_path.exists():
        schedule_node_vector_autosync(user_id=store.user_id, store=store, reason="index_missing")
        return NodeVectorSearchResult(
            node_ids=[],
            vector_hit_count=0,
            index_state="missing",
            fallback_reason="node_vector_index_missing",
        )

    embeddings_model = await _ensure_embeddings_model()
    if embeddings_model is None:
        return NodeVectorSearchResult(
            node_ids=[],
            vector_hit_count=0,
            index_state="unavailable",
            fallback_reason="embedding_unavailable",
        )

    map_payload = _load_json(map_path)
    node_map = map_payload.get("nodes", {})
    if not isinstance(node_map, dict) or not node_map:
        schedule_node_vector_autosync(user_id=store.user_id, store=store, reason="empty_map")
        return NodeVectorSearchResult(
            node_ids=[],
            vector_hit_count=0,
            index_state="empty",
            fallback_reason="node_vector_map_empty",
        )

    try:
        faiss_store = await asyncio.to_thread(
            lambda: FAISS.load_local(
                str(store.storage_dir),
                embeddings_model,
                index_name=NODE_VECTOR_INDEX_NAME,
                allow_dangerous_deserialization=True,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        schedule_node_vector_autosync(user_id=store.user_id, store=store, reason="index_load_failed")
        logger.warning("Node-vector index load failed for user %s: %s", store.user_id, exc)
        return NodeVectorSearchResult(
            node_ids=[],
            vector_hit_count=0,
            index_state="corrupted",
            fallback_reason="node_vector_index_load_failed",
        )

    try:
        query_vector = await embeddings_model.aembed_query(normalized_query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Node-vector query embedding failed for user %s: %s", store.user_id, exc)
        return NodeVectorSearchResult(
            node_ids=[],
            vector_hit_count=0,
            index_state="ready",
            fallback_reason="query_embedding_failed",
        )

    effective_top_k = min(top_k or node_vector_top_k(), len(node_map))
    fetch_k = min(max(effective_top_k * 3, effective_top_k), len(node_map))
    if fetch_k <= 0:
        return NodeVectorSearchResult(
            node_ids=[],
            vector_hit_count=0,
            index_state="empty",
            fallback_reason="no_indexed_nodes",
        )

    if hasattr(faiss_store, "similarity_search_with_score_by_vector"):
        raw_results = await asyncio.to_thread(
            faiss_store.similarity_search_with_score_by_vector,
            query_vector,
            fetch_k,
        )
    else:
        raw_results = await asyncio.to_thread(
            faiss_store.similarity_search_with_score,
            normalized_query,
            fetch_k,
        )

    ranked: list[tuple[float, str]] = []
    for doc, _distance in raw_results:
        node_id = str(doc.metadata.get("node_id", "")).strip()
        if not node_id:
            continue
        entry = node_map.get(node_id)
        embedding = entry.get("embedding") if isinstance(entry, dict) else None
        if not isinstance(embedding, list) or not embedding:
            continue
        score = _cosine_similarity(query_vector, [float(value) for value in embedding])
        ranked.append((score, node_id))

    ranked.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    deduped: list[tuple[float, str]] = []
    for score, node_id in ranked:
        if node_id in seen:
            continue
        seen.add(node_id)
        deduped.append((score, node_id))

    threshold = min_score if min_score is not None else node_vector_min_score()
    filtered = [(score, node_id) for score, node_id in deduped if score >= threshold]
    node_ids = [node_id for _, node_id in filtered[:effective_top_k]]
    top_score = filtered[0][0] if filtered else (deduped[0][0] if deduped else None)

    if not node_ids:
        return NodeVectorSearchResult(
            node_ids=[],
            vector_hit_count=0,
            index_state="ready",
            fallback_reason="vector_score_below_threshold",
            top_score=top_score,
        )

    return NodeVectorSearchResult(
        node_ids=node_ids,
        vector_hit_count=len(node_ids),
        index_state="ready",
        top_score=top_score,
    )
