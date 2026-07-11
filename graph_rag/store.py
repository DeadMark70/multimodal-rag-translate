"""
GraphRAG Store Module

Provides NetworkX-based graph storage with serialization support.
Manages per-user knowledge graphs with CRUD operations.
"""

# Standard library
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

# Third-party
import networkx as nx

# Local application
from core import uploads as upload_paths
from graph_rag.schemas import (
    Community,
    CanonicalEntity,
    ClaimIdentity,
    EvidenceAnchor,
    GraphAssetLink,
    GraphDocumentStatus,
    GraphEdgeProvenance,
    GraphExtractionRunManifest,
    RawGraphCandidate,
    EntityType,
    GraphEdge,
    GraphNode,
    NodeVectorSyncState,
    NodeVectorSyncStatusResponse,
    GraphStatusResponse,
)

# Configure logging
logger = logging.getLogger(__name__)

GRAPH_INDEX_VERSION = 2
GRAPH_DOCUMENTS_FILENAME = "graph.documents.json"
GRAPH_EXTRACTION_RUNS_FILENAME = "graph.extraction_runs.json"
GRAPH_PROVENANCE_FILENAME = "graph.provenance.json"
GRAPH_RAW_CANDIDATES_FILENAME = "graph.raw_candidates.json"
GRAPH_ALIASES_FILENAME = "graph.aliases.json"
GRAPH_TYPE_INDEX_FILENAME = "graph.type_index.json"
GRAPH_DOC_INDEX_FILENAME = "graph.doc_index.json"
GRAPH_ASSET_LINKS_FILENAME = "graph.asset_links.json"
GRAPH_SOURCE_MARKDOWN_FILENAME = "extracted.md"
NODE_VECTOR_INDEX_NAME = "node_index"
NODE_VECTOR_MAP_FILENAME = "node_index_map.json"
NODE_VECTOR_META_FILENAME = "node_index.meta.json"
_UNSET = object()

AUTO_MERGE_ENTITY_TYPES = {
    EntityType.PAPER,
    EntityType.DATASET,
    EntityType.METRIC,
    EntityType.METHOD,
}
REVIEW_REQUIRED_ENTITY_TYPES = {
    EntityType.MODEL,
    EntityType.ARCHITECTURE_COMPONENT,
    EntityType.TRAINING_SETTING,
}
NEVER_AUTO_MERGE_ENTITY_TYPES = {
    EntityType.CLAIM,
    EntityType.RESULT,
    EntityType.TABLE,
    EntityType.FORMULA,
}


def _generate_node_id(label: str, entity_type: EntityType) -> str:
    """
    Generate a unique node ID from label and type.
    
    Args:
        label: Entity label.
        entity_type: Entity type.
        
    Returns:
        Unique node ID string.
    """
    content = f"{entity_type.value}:{label.lower().strip()}"
    hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"node_{entity_type.value}_{hash_suffix}"


def _generate_scoped_node_id(
    label: str,
    entity_type: EntityType,
    source_doc_ids: List[str],
) -> str:
    """Generate a stable node id scoped to the source-document set."""
    scope = "|".join(sorted(source_doc_ids)) or "no-source"
    content = f"{entity_type.value}:{label.lower().strip()}:{scope}"
    hash_suffix = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"node_{entity_type.value}_{hash_suffix}"


def _next_unmerged_node_id(
    graph: nx.DiGraph,
    label: str,
    entity_type: EntityType,
    source_doc_ids: List[str],
) -> str:
    """Allocate a distinct identity for types that must never auto-merge."""
    base = _generate_scoped_node_id(label, entity_type, source_doc_ids)
    node_id = base
    suffix = 2
    while graph.has_node(node_id):
        node_id = f"{base}_{suffix}"
        suffix += 1
    return node_id


def _normalize_alias(value: str) -> str:
    """Normalize explicit aliases without applying fuzzy similarity."""
    return " ".join(value.strip().lower().replace("-", " ").split())


def _unique_aliases(values: List[str]) -> List[str]:
    """Return normalized alias keys in caller-provided order."""
    aliases: List[str] = []
    for value in values:
        alias = _normalize_alias(value)
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


class GraphStore:
    """
    Per-user knowledge graph storage using NetworkX.
    
    Provides CRUD operations for nodes and edges, serialization to pickle,
    and graph analysis utilities.
    
    Attributes:
        user_id: Owner user's ID.
        graph: NetworkX DiGraph instance.
        communities: List of detected communities.
        last_updated: Timestamp of last modification.
    """
    
    def __init__(self, user_id: str, storage_dir: str | Path | None = None) -> None:
        """
        Initialize GraphStore for a user.
        
        Loads existing graph from disk or creates a new empty graph.
        
        Args:
            user_id: User's ID.
        """
        self.user_id = user_id
        self._root_storage_dir = (
            Path(storage_dir)
            if storage_dir is not None
            else upload_paths.get_rag_index_dir_path(self.user_id)
        )
        self._storage_dir = self._current_snapshot_dir(self._root_storage_dir) or self._root_storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self.graph: nx.DiGraph = nx.DiGraph()
        self.communities: List[Community] = []
        self.document_statuses: Dict[str, GraphDocumentStatus] = {}
        self.extraction_manifests: List[GraphExtractionRunManifest] = []
        self.edge_provenance: Dict[str, GraphEdgeProvenance] = {}
        self.raw_candidates: Dict[str, RawGraphCandidate] = {}
        self.asset_links: Dict[str, GraphAssetLink] = {}
        self.canonical_entities: Dict[str, CanonicalEntity] = {}
        self.alias_index: Dict[str, List[str]] = {}
        self.type_index: Dict[str, List[str]] = {}
        self.doc_index: Dict[str, List[str]] = {}
        self.last_updated: Optional[datetime] = None
        self.last_optimized_at: Optional[datetime] = None
        self.index_version: int = GRAPH_INDEX_VERSION
        self.graph_dirty: bool = False
        self.node_vector_dirty: bool = False
        self.node_vector_sync: NodeVectorSyncStatusResponse = NodeVectorSyncStatusResponse()
        self._pending_count: int = 0
        self.active_job_state: Optional[str] = None
        
        # Try to load existing graph; if only sidecars exist, still recover metadata/status.
        if self._graph_exists():
            self.load()
        else:
            self._load_metadata()
            self._load_document_statuses()
            self._load_extraction_manifests()
            self._load_edge_provenance()
            self._load_raw_candidates()
            self._load_asset_links()
            self._load_alias_indexes()
    
    def _get_graph_path(self) -> Path:
        """
        Get path to graph pickle file.
        
        Returns:
            Path to graph.pkl file.
        """
        return self._storage_dir / "graph.pkl"

    @staticmethod
    def _current_snapshot_dir(root: Path) -> Path | None:
        pointer_path = root / "current.json"
        if not pointer_path.exists():
            return None
        try:
            payload = json.loads(pointer_path.read_text(encoding="utf-8"))
            version = payload.get("current_version")
            candidate = root / "versions" / str(version)
            return candidate if candidate.is_dir() else None
        except (OSError, ValueError, TypeError):
            return None

    def load_current_pointer(self) -> Dict[str, Any]:
        """Return the immutable snapshot pointer, or an empty mapping for legacy stores."""
        path = self._root_storage_dir / "current.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _next_snapshot_version(self) -> str:
        versions_dir = self._root_storage_dir / "versions"
        existing = [
            int(item.name[1:])
            for item in versions_dir.glob("v*")
            if item.is_dir() and item.name[1:].isdigit()
        ] if versions_dir.exists() else []
        return f"v{(max(existing, default=0) + 1):03d}"

    @staticmethod
    def _hash_file(path: Path) -> str:
        return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

    def save_snapshot(
        self,
        *,
        node_vector_source: "GraphStore | None" = None,
    ) -> str:
        """Write a complete immutable graph version before atomically swapping current."""
        version = self._next_snapshot_version()
        versions_dir = self._root_storage_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        target = versions_dir / version
        temp = versions_dir / f".{version}.tmp"
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        original_dir = self._storage_dir
        vector_store = node_vector_source or self
        vector_paths = (
            vector_store._get_node_vector_faiss_path(),
            vector_store._get_node_vector_pickle_path(),
            vector_store._get_node_vector_map_path(),
            vector_store._get_node_vector_meta_path(),
        )
        newly_versioned_manifests = [
            manifest
            for manifest in self.extraction_manifests
            if manifest.graph_snapshot_version is None
        ]
        try:
            for manifest in newly_versioned_manifests:
                manifest.graph_snapshot_version = version
            self._storage_dir = temp
            self.save()
            if not self.node_vector_dirty:
                for source_path in vector_paths:
                    if source_path.exists():
                        shutil.copy2(source_path, temp / source_path.name)
            required = [
                temp / "graph.pkl",
                temp / "graph.meta.json",
                temp / GRAPH_EXTRACTION_RUNS_FILENAME,
            ]
            if any(not path.exists() for path in required):
                raise RuntimeError("snapshot validation failed: graph files missing")
            temp.rename(target)
            hashes = {
                path.name: self._hash_file(path)
                for path in sorted(target.iterdir())
                if path.is_file()
            }
            pointer = {
                "current_version": version,
                "created_at": datetime.now().isoformat(),
                "schema_version": "graph-schema-v1",
                "graph_hash": hashes["graph.pkl"],
                "sidecar_hashes": hashes,
            }
            pointer_path = self._root_storage_dir / "current.json"
            pointer_temp = pointer_path.with_suffix(".json.tmp")
            pointer_temp.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
            pointer_temp.replace(pointer_path)
            self._storage_dir = target
            return version
        except Exception:
            for manifest in newly_versioned_manifests:
                manifest.graph_snapshot_version = None
            if temp.exists():
                shutil.rmtree(temp)
            raise
        finally:
            if self._storage_dir == temp:
                self._storage_dir = original_dir
    
    def _graph_exists(self) -> bool:
        """
        Check if graph file exists on disk.
        
        Returns:
            True if graph file exists.
        """
        return self._get_graph_path().exists()

    def _get_metadata_path(self) -> Path:
        """Get path to sidecar metadata JSON file."""
        return self._get_graph_path().with_name("graph.meta.json")

    def _get_document_status_path(self) -> Path:
        """Get path to GraphRAG per-document status sidecar."""
        return self._get_graph_path().with_name(GRAPH_DOCUMENTS_FILENAME)

    def _get_extraction_runs_path(self) -> Path:
        """Get path to GraphRAG extraction run manifest sidecar."""
        return self._get_graph_path().with_name(GRAPH_EXTRACTION_RUNS_FILENAME)

    def _get_provenance_path(self) -> Path:
        """Get path to graph edge provenance sidecar."""
        return self._get_graph_path().with_name(GRAPH_PROVENANCE_FILENAME)

    def _get_raw_candidates_path(self) -> Path:
        """Get path to the unverified extraction candidate sidecar."""
        return self._get_graph_path().with_name(GRAPH_RAW_CANDIDATES_FILENAME)

    def _get_asset_links_path(self) -> Path:
        """Get the graph asset registry sidecar path."""
        return self._get_graph_path().with_name(GRAPH_ASSET_LINKS_FILENAME)

    def _get_aliases_path(self) -> Path:
        return self._get_graph_path().with_name(GRAPH_ALIASES_FILENAME)

    def _get_type_index_path(self) -> Path:
        return self._get_graph_path().with_name(GRAPH_TYPE_INDEX_FILENAME)

    def _get_doc_index_path(self) -> Path:
        return self._get_graph_path().with_name(GRAPH_DOC_INDEX_FILENAME)

    def _get_node_vector_faiss_path(self) -> Path:
        """Get path to GraphRAG node-vector FAISS index."""
        return self._storage_dir / f"{NODE_VECTOR_INDEX_NAME}.faiss"

    def _get_node_vector_pickle_path(self) -> Path:
        """Get path to GraphRAG node-vector FAISS sidecar pickle."""
        return self._storage_dir / f"{NODE_VECTOR_INDEX_NAME}.pkl"

    def _get_node_vector_map_path(self) -> Path:
        """Get path to GraphRAG node-vector map sidecar JSON."""
        return self._storage_dir / NODE_VECTOR_MAP_FILENAME

    def _get_node_vector_meta_path(self) -> Path:
        """Get path to GraphRAG node-vector metadata sidecar JSON."""
        return self._storage_dir / NODE_VECTOR_META_FILENAME

    def _build_metadata_payload(self) -> Dict[str, Any]:
        """Build sidecar metadata payload."""
        return {
            "index_version": self.index_version,
            "communities": [c.model_dump() for c in self.communities],
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "last_optimized_at": (
                self.last_optimized_at.isoformat() if self.last_optimized_at else None
            ),
            "graph_dirty": self.graph_dirty,
            "node_vector_dirty": self.node_vector_dirty,
            "node_vector_sync": self.node_vector_sync.model_dump(mode="json"),
            "pending_count": self._pending_count,
            "community_level_counts": self.get_community_level_counts(),
            "active_job_state": self.active_job_state,
        }

    def _atomic_write_json(self, path: Path, payload: object) -> None:
        """Persist JSON using a temporary file and atomic replace."""
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)

    def _write_metadata(self) -> None:
        """Persist sidecar metadata JSON."""
        path = self._get_metadata_path()
        self._atomic_write_json(path, self._build_metadata_payload())

    def _write_document_statuses(self) -> None:
        """Persist GraphRAG per-document status JSON."""
        path = self._get_document_status_path()
        payload = {
            "documents": [
                status.model_dump(mode="json")
                for status in sorted(self.document_statuses.values(), key=lambda item: item.doc_id)
            ]
        }
        self._atomic_write_json(path, payload)

    def _write_extraction_manifests(self) -> None:
        """Persist successful GraphRAG extraction run metadata."""
        self._atomic_write_json(
            self._get_extraction_runs_path(),
            {"runs": [manifest.model_dump(mode="json") for manifest in self.extraction_manifests]},
        )

    def _write_edge_provenance(self) -> None:
        """Persist graph edge provenance sidecar JSON."""
        payload = {
            "edges": [
                provenance.model_dump(mode="json")
                for provenance in sorted(
                    self.edge_provenance.values(),
                    key=lambda item: item.edge_id,
                )
            ]
        }
        self._atomic_write_json(self._get_provenance_path(), payload)

    def _write_raw_candidates(self) -> None:
        """Persist unverified extraction output separately from graph evidence."""
        payload = {
            "candidates": [
                candidate.model_dump(mode="json")
                for candidate in sorted(
                    self.raw_candidates.values(),
                    key=lambda item: item.candidate_id,
                )
            ]
        }
        self._atomic_write_json(self._get_raw_candidates_path(), payload)

    def _write_asset_links(self) -> None:
        """Persist parsed document assets separately from answer evidence."""
        self._atomic_write_json(
            self._get_asset_links_path(),
            {
                "assets": [
                    link.model_dump(mode="json")
                    for link in sorted(
                        self.asset_links.values(), key=lambda item: item.asset_id
                    )
                ]
            },
        )

    def _write_alias_indexes(self) -> None:
        """Persist canonical entity and lookup sidecars independently of graph pickle."""
        self._atomic_write_json(
            self._get_aliases_path(),
            {
                "entities": [
                    entity.model_dump(mode="json")
                    for entity in sorted(
                        self.canonical_entities.values(),
                        key=lambda item: item.canonical_id,
                    )
                ],
                "aliases": self.alias_index,
            },
        )
        self._atomic_write_json(self._get_type_index_path(), self.type_index)
        self._atomic_write_json(self._get_doc_index_path(), self.doc_index)

    def _load_legacy_metadata(self, legacy_data: Dict[str, Any]) -> None:
        """Load metadata fields embedded in older pickle payloads."""
        self.communities = [
            Community(**c) for c in legacy_data.get("communities", [])
        ]
        self.last_updated = legacy_data.get("last_updated")
        self._pending_count = legacy_data.get("pending_count", 0)
        self.last_optimized_at = legacy_data.get("last_optimized_at")
        self.index_version = int(legacy_data.get("index_version", 1))
        self.graph_dirty = bool(self._pending_count) or self.index_version < GRAPH_INDEX_VERSION
        self.node_vector_dirty = self.graph_dirty
        self.node_vector_sync = NodeVectorSyncStatusResponse()
        self.active_job_state = legacy_data.get("active_job_state")

    def _load_metadata(self, legacy_data: Optional[Dict[str, Any]] = None) -> None:
        """Load sidecar metadata or fall back to legacy pickle metadata."""
        metadata_path = self._get_metadata_path()

        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.index_version = int(data.get("index_version", GRAPH_INDEX_VERSION))
                self.communities = [
                    Community(**c) for c in data.get("communities", [])
                ]
                last_updated = data.get("last_updated")
                self.last_updated = datetime.fromisoformat(last_updated) if last_updated else None
                last_optimized = data.get("last_optimized_at")
                self.last_optimized_at = (
                    datetime.fromisoformat(last_optimized) if last_optimized else None
                )
                self.graph_dirty = bool(data.get("graph_dirty", False))
                self.node_vector_dirty = bool(data.get("node_vector_dirty", self.graph_dirty))
                node_vector_sync = data.get("node_vector_sync")
                if isinstance(node_vector_sync, dict):
                    self.node_vector_sync = NodeVectorSyncStatusResponse.model_validate(
                        node_vector_sync
                    )
                else:
                    self.node_vector_sync = NodeVectorSyncStatusResponse()
                self._pending_count = int(data.get("pending_count", 0))
                self.active_job_state = data.get("active_job_state")
                return
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                logger.warning("Failed to load graph metadata for user %s: %s", self.user_id, e)
                if legacy_data:
                    self._load_legacy_metadata(legacy_data)
                    return

        if legacy_data:
            self._load_legacy_metadata(legacy_data)

    def _load_document_statuses(self) -> None:
        """Load per-document GraphRAG statuses from sidecar JSON."""
        self.document_statuses = {}
        path = self._get_document_status_path()
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for item in payload.get("documents", []):
                status = GraphDocumentStatus.model_validate(item)
                self.document_statuses[status.doc_id] = status
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.warning("Failed to load graph document statuses for user %s: %s", self.user_id, e)

    def _load_extraction_manifests(self) -> None:
        """Load extraction manifests while retaining malformed legacy-store readability."""
        self.extraction_manifests = []
        path = self._get_extraction_runs_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to load graph extraction manifests for user %s: %s", self.user_id, exc)
            return
        for item in payload.get("runs", []):
            try:
                self.extraction_manifests.append(GraphExtractionRunManifest.model_validate(item))
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed graph extraction manifest for user %s: %s", self.user_id, exc)

    def _load_edge_provenance(self) -> None:
        """Load graph edge provenance sidecar JSON."""
        self.edge_provenance = {}
        path = self._get_provenance_path()
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            items: List[GraphEdgeProvenance] = []
            if isinstance(payload, dict) and isinstance(payload.get("edges"), list):
                items = [GraphEdgeProvenance.model_validate(item) for item in payload["edges"]]
            elif isinstance(payload, dict):
                for edge_id, anchors in payload.items():
                    if isinstance(anchors, list):
                        items.append(
                            GraphEdgeProvenance(
                                edge_id=edge_id,
                                anchors=[EvidenceAnchor.model_validate(anchor) for anchor in anchors],
                            )
                        )

            self.edge_provenance = {
                provenance.edge_id: provenance
                for provenance in items
            }
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.warning("Failed to load graph provenance for user %s: %s", self.user_id, e)

    def _load_raw_candidates(self) -> None:
        """Load unverified extraction output without treating it as graph evidence."""
        self.raw_candidates = {}
        path = self._get_raw_candidates_path()
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            candidates = [
                RawGraphCandidate.model_validate(item)
                for item in payload.get("candidates", [])
            ]
            self.raw_candidates = {
                candidate.candidate_id: candidate for candidate in candidates
            }
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to load graph raw candidates for user %s: %s",
                self.user_id,
                exc,
            )

    def _load_asset_links(self) -> None:
        """Load a best-effort asset registry while retaining legacy graph readability."""
        self.asset_links = {}
        path = self._get_asset_links_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            links = [
                GraphAssetLink.model_validate(item)
                for item in payload.get("assets", [])
            ]
            self.asset_links = {link.asset_id: link for link in links}
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to load graph asset links for user %s: %s", self.user_id, exc)

    def _load_alias_indexes(self) -> None:
        """Load optional alias indexes while preserving legacy graph readability."""
        self.canonical_entities = {}
        self.alias_index = {}
        self.type_index = {}
        self.doc_index = {}
        try:
            aliases_path = self._get_aliases_path()
            if aliases_path.exists():
                with open(aliases_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                entities = [
                    CanonicalEntity.model_validate(item)
                    for item in payload.get("entities", [])
                ]
                self.canonical_entities = {
                    entity.canonical_id: entity for entity in entities
                }
                raw_aliases = payload.get("aliases", {})
                if isinstance(raw_aliases, dict):
                    self.alias_index = {
                        str(alias): [str(node_id) for node_id in node_ids]
                        for alias, node_ids in raw_aliases.items()
                        if isinstance(node_ids, list)
                    }

            for path, target in (
                (self._get_type_index_path(), "type_index"),
                (self._get_doc_index_path(), "doc_index"),
            ):
                if not path.exists():
                    continue
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    setattr(
                        self,
                        target,
                        {
                            str(key): [str(node_id) for node_id in value]
                            for key, value in payload.items()
                            if isinstance(value, list)
                        },
                    )
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to load graph alias indexes for user %s: %s", self.user_id, exc)

    def save_sidecars(self) -> None:
        """Persist metadata-only sidecars without rewriting the graph pickle."""
        self._write_metadata()
        self._write_document_statuses()
        self._write_extraction_manifests()
        self._write_edge_provenance()
        self._write_raw_candidates()
        self._write_asset_links()
        self._write_alias_indexes()

    def save(self) -> None:
        """
        Serialize graph to pickle file.
        
        Saves graph, communities, and metadata to disk.
        """
        self.last_updated = datetime.now()
        
        data = {
            "graph": self.graph,
        }

        path = self._get_graph_path()
        with open(path, "wb") as f:
            pickle.dump(data, f)
        self.save_sidecars()
        
        logger.info(
            f"Saved graph for user {self.user_id}: "
            f"{self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
    
    def load(self) -> bool:
        """
        Load graph from pickle file.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        path = self._get_graph_path()
        
        if not path.exists():
            logger.info(f"No graph file found for user {self.user_id}")
            return False
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            self.graph = data.get("graph", nx.DiGraph())
            self._load_metadata(legacy_data=data)
            self._load_document_statuses()
            self._load_extraction_manifests()
            self._load_edge_provenance()
            self._load_raw_candidates()
            self._load_asset_links()
            self._load_alias_indexes()
            
            logger.info(
                f"Loaded graph for user {self.user_id}: "
                f"{self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load graph for user {self.user_id}: {e}")
            self.graph = nx.DiGraph()
            return False
    
    def clear(self) -> None:
        """
        Clear all graph data.
        
        Removes all nodes, edges, and communities.
        """
        self.graph.clear()
        self.communities.clear()
        self.document_statuses.clear()
        self.extraction_manifests.clear()
        self.edge_provenance.clear()
        self.raw_candidates.clear()
        self.asset_links.clear()
        self.canonical_entities.clear()
        self.alias_index.clear()
        self.type_index.clear()
        self.doc_index.clear()
        self._pending_count = 0
        self.last_optimized_at = None
        self.graph_dirty = False
        self.node_vector_dirty = False
        self.node_vector_sync = NodeVectorSyncStatusResponse()
        self.active_job_state = None
        logger.info(f"Cleared graph for user {self.user_id}")

    @property
    def storage_dir(self) -> Path:
        """Return the on-disk storage directory used by this store."""
        return self._storage_dir

    def mark_dirty(self) -> None:
        """Mark graph metadata as needing optimization or rebuild."""
        self.graph_dirty = True
        self.node_vector_dirty = True

    def mark_node_vector_dirty(self) -> None:
        """Mark node-vector sidecars as stale."""
        self.node_vector_dirty = True

    def clear_node_vector_dirty(self) -> None:
        """Mark node-vector sidecars as fresh."""
        self.node_vector_dirty = False

    def get_node_vector_sync_status(self) -> NodeVectorSyncStatusResponse:
        """Return node-vector manual sync status snapshot."""
        return self.node_vector_sync

    def set_node_vector_sync_status(
        self,
        *,
        state: NodeVectorSyncState,
        processed: int | object = _UNSET,
        total: int | object = _UNSET,
        changed: int | object = _UNSET,
        reused: int | object = _UNSET,
        removed: int | object = _UNSET,
        index_state: Optional[str] | object = _UNSET,
        autosync_duration_ms: Optional[int] | object = _UNSET,
        last_error: Optional[str] | object = _UNSET,
        started_at: Optional[datetime] | object = _UNSET,
        updated_at: Optional[datetime] | object = _UNSET,
        finished_at: Optional[datetime] | object = _UNSET,
    ) -> None:
        """Update node-vector sync status sidecar fields."""
        current = self.node_vector_sync.model_copy(deep=True)
        current.state = state
        if processed is not _UNSET:
            current.processed = processed
        if total is not _UNSET:
            current.total = total
        if changed is not _UNSET:
            current.changed = changed
        if reused is not _UNSET:
            current.reused = reused
        if removed is not _UNSET:
            current.removed = removed
        if index_state is not _UNSET:
            current.index_state = index_state
        if autosync_duration_ms is not _UNSET:
            current.autosync_duration_ms = autosync_duration_ms
        if last_error is not _UNSET:
            current.last_error = last_error
        if started_at is not _UNSET:
            current.started_at = started_at
        if updated_at is not _UNSET:
            current.updated_at = updated_at
        if finished_at is not _UNSET:
            current.finished_at = finished_at
        self.node_vector_sync = current

    def get_communities(self, level: Optional[int] = None) -> List[Community]:
        """Return communities, optionally filtered by hierarchy level."""
        if level is None:
            return list(self.communities)
        return [community for community in self.communities if community.level == level]
    
    # ===== Node Operations =====
    
    def add_node(self, node: GraphNode) -> str:
        """
        Add a node to the graph.
        
        If node with same ID exists, merges doc_ids.
        
        Args:
            node: GraphNode to add.
            
        Returns:
            Node ID.
        """
        node_id = node.id
        
        if self.graph.has_node(node_id):
            # Merge doc_ids
            existing_docs = set(self.graph.nodes[node_id].get("doc_ids", []))
            existing_docs.update(node.doc_ids)
            self.graph.nodes[node_id]["doc_ids"] = list(existing_docs)
            logger.debug(f"Merged node {node_id}, now has {len(existing_docs)} docs")
        else:
            # Add new node
            self.graph.add_node(
                node_id,
                label=node.label,
                entity_type=node.entity_type.value,
                doc_ids=node.doc_ids,
                description=node.description,
                pending_resolution=node.pending_resolution,
                embedding=node.embedding,
            )
            
            if node.pending_resolution:
                self._pending_count += 1
            
            logger.debug(f"Added node: {node_id} ({node.label})")
        self.mark_dirty()
        
        return node_id
    
    def add_node_from_extraction(
        self,
        label: str,
        entity_type: EntityType,
        doc_id: str,
        description: Optional[str] = None,
        pending_resolution: bool = True,
    ) -> str:
        """
        Add a node from extraction results.
        
        Convenience method that generates node ID automatically.
        
        Args:
            label: Entity label.
            entity_type: Entity type.
            doc_id: Source document ID.
            description: Optional description.
            pending_resolution: Whether to mark as pending.
            
        Returns:
            Generated node ID.
        """
        node_id = _generate_node_id(label, entity_type)
        
        node = GraphNode(
            id=node_id,
            label=label,
            entity_type=entity_type,
            doc_ids=[doc_id],
            description=description,
            pending_resolution=pending_resolution,
        )
        
        return self.add_node(node)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node ID.
            
        Returns:
            GraphNode if found, None otherwise.
        """
        if not self.graph.has_node(node_id):
            return None
        
        data = self.graph.nodes[node_id]
        return GraphNode(
            id=node_id,
            label=data.get("label", ""),
            entity_type=EntityType(data.get("entity_type", "concept")),
            doc_ids=data.get("doc_ids", []),
            description=data.get("description"),
            pending_resolution=data.get("pending_resolution", False),
            embedding=data.get("embedding"),
        )
    
    def find_nodes_by_label(self, label: str, fuzzy: bool = False) -> List[str]:
        """
        Find nodes by label.
        
        Args:
            label: Label to search for.
            fuzzy: If True, uses substring matching.
            
        Returns:
            List of matching node IDs.
        """
        results = []
        label_lower = label.lower().strip()
        
        for node_id, data in self.graph.nodes(data=True):
            node_label = data.get("label", "").lower()
            
            if fuzzy:
                if label_lower in node_label or node_label in label_lower:
                    results.append(node_id)
            else:
                if label_lower == node_label:
                    results.append(node_id)
        
        return results
    
    # ===== Edge Operations =====

    def edge_id(self, source_id: str, target_id: str, relation: str) -> str:
        """Build a deterministic opaque edge identifier."""
        raw = f"{source_id}|{target_id}|{relation.strip().lower()}"
        return f"edge:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"

    def record_edge_provenance(self, edge_id: str, anchors: List[EvidenceAnchor]) -> None:
        """Store provenance anchors for one edge."""
        existing = self.edge_provenance.get(edge_id)
        self.edge_provenance[edge_id] = GraphEdgeProvenance(
            edge_id=edge_id,
            anchors=[anchor.model_copy(deep=True) for anchor in anchors],
            extraction_run_id=existing.extraction_run_id if existing else None,
            extraction_prompt_version=(
                existing.extraction_prompt_version if existing else None
            ),
            created_at=existing.created_at if existing else datetime.now(),
        )
        self.mark_dirty()

    def get_edge_provenance(self, edge_id: str) -> List[EvidenceAnchor]:
        """Return persisted provenance anchors for one edge."""
        provenance = self.edge_provenance.get(edge_id)
        if provenance is None:
            return []
        return [anchor.model_copy(deep=True) for anchor in provenance.anchors]

    def get_edge_provenance_status(
        self,
        edge_id: str,
    ) -> Literal["full", "partial", "missing"]:
        """Return aggregated provenance coverage for one edge."""
        anchors = self.get_edge_provenance(edge_id)
        if not anchors:
            return "missing"
        if any(anchor.provenance_status == "full" for anchor in anchors):
            return "full"
        if any(anchor.provenance_status == "partial" for anchor in anchors):
            return "partial"
        return "missing"

    def record_raw_candidate(self, candidate: RawGraphCandidate) -> None:
        """Persist an unverified extraction candidate outside answer-eligible graph data."""
        self.raw_candidates[candidate.candidate_id] = candidate.model_copy(deep=True)

    def get_raw_candidates_for_doc(self, doc_id: str) -> List[RawGraphCandidate]:
        """Return diagnostic candidates for one document in deterministic order."""
        return [
            candidate.model_copy(deep=True)
            for candidate in sorted(
                self.raw_candidates.values(), key=lambda item: item.candidate_id
            )
            if candidate.source_doc_id == doc_id
        ]

    def record_asset_link(self, link: GraphAssetLink) -> None:
        """Insert or replace one parsed document asset in the locator registry."""
        self.asset_links[link.asset_id] = link.model_copy(deep=True)
        self.mark_dirty()

    def get_asset_links_for_doc(self, doc_id: str) -> List[GraphAssetLink]:
        """Return deterministic copies of one document's registered assets."""
        return [
            link.model_copy(deep=True)
            for link in sorted(self.asset_links.values(), key=lambda item: item.asset_id)
            if link.doc_id == doc_id
        ]

    def has_usable_asset_links(
        self,
        doc_ids: Set[str],
        asset_types: Set[str],
    ) -> bool:
        """Check an explicit request scope against parsed assets, never feature flags."""
        if not doc_ids or not asset_types:
            return False
        return any(
            link.doc_id in doc_ids
            and link.asset_type in asset_types
            and link.asset_parse_status == "parsed"
            and bool(link.source_chunk_id or (link.text_or_markdown and link.asset_text_hash))
            for link in self.asset_links.values()
        )

    def upsert_canonical_entity(
        self,
        *,
        canonical_name: str,
        entity_type: EntityType | str,
        aliases: List[str],
        source_doc_ids: List[str],
        confidence: float = 1.0,
        claim_identity: ClaimIdentity | None = None,
    ) -> str:
        """Create or reuse a canonical node under explicit type/scope merge rules."""
        resolved_type = EntityType(entity_type)
        normalized_name = _normalize_alias(canonical_name)
        normalized_aliases = _unique_aliases([canonical_name, *aliases])
        existing_id: str | None = None

        identity_key = claim_identity.stable_key if claim_identity is not None else None
        if resolved_type in AUTO_MERGE_ENTITY_TYPES:
            existing_id = self._find_exact_canonical_name(normalized_name, resolved_type)
            if existing_id is None:
                existing_id = self.find_canonical_node(
                    canonical_name,
                    resolved_type.value,
                )
        elif resolved_type in REVIEW_REQUIRED_ENTITY_TYPES:
            existing_id = self._find_scoped_alias_match(
                normalized_aliases,
                resolved_type,
                source_doc_ids,
            )
        elif resolved_type == EntityType.CLAIM and identity_key is not None:
            existing_id = next(
                (
                    entity.canonical_id
                    for entity in self.canonical_entities.values()
                    if entity.entity_type == EntityType.CLAIM
                    and entity.identity_key == identity_key
                ),
                None,
            )

        review_status: Literal["auto", "needs_review", "reviewed"] = (
            "needs_review"
            if resolved_type in REVIEW_REQUIRED_ENTITY_TYPES
            else "auto"
        )
        if existing_id is None:
            if resolved_type in AUTO_MERGE_ENTITY_TYPES:
                existing_id = _generate_node_id(canonical_name, resolved_type)
            elif resolved_type in REVIEW_REQUIRED_ENTITY_TYPES:
                existing_id = _generate_scoped_node_id(
                    canonical_name,
                    resolved_type,
                    source_doc_ids,
                )
            elif identity_key is not None:
                existing_id = _generate_scoped_node_id(
                    identity_key,
                    resolved_type,
                    source_doc_ids,
                )
            else:
                existing_id = _next_unmerged_node_id(
                    self.graph,
                    canonical_name,
                    resolved_type,
                    source_doc_ids,
                )
            self.add_node(
                GraphNode(
                    id=existing_id,
                    label=canonical_name,
                    entity_type=resolved_type,
                    doc_ids=list(dict.fromkeys(source_doc_ids)),
                    pending_resolution=False,
                )
            )
            entity = CanonicalEntity(
                canonical_id=existing_id,
                canonical_name=canonical_name,
                entity_type=resolved_type,
                aliases=normalized_aliases,
                source_doc_ids=list(dict.fromkeys(source_doc_ids)),
                identity_key=identity_key,
                confidence=confidence,
                review_status=review_status,
            )
        else:
            existing = self.canonical_entities.get(existing_id)
            if existing is None:
                existing_node = self.get_node(existing_id)
                existing = CanonicalEntity(
                    canonical_id=existing_id,
                    canonical_name=existing_node.label if existing_node else canonical_name,
                    entity_type=resolved_type,
                    aliases=[],
                    source_doc_ids=existing_node.doc_ids if existing_node else [],
                    identity_key=identity_key,
                    review_status=review_status,
                )
            entity = existing.model_copy(
                update={
                    "aliases": _unique_aliases([*existing.aliases, *normalized_aliases]),
                    "source_doc_ids": list(
                        dict.fromkeys([*existing.source_doc_ids, *source_doc_ids])
                    ),
                    "confidence": max(existing.confidence, confidence),
                    "identity_key": existing.identity_key or identity_key,
                }
            )

        self.canonical_entities[existing_id] = entity
        self._index_canonical_entity(entity)
        self.mark_dirty()
        return existing_id

    def find_canonical_node(
        self,
        label: str,
        entity_type: str | None = None,
    ) -> str | None:
        """Resolve an exact canonical name or alias without fuzzy cross-type merging."""
        normalized_label = _normalize_alias(label)
        for node_id in self.alias_index.get(normalized_label, []):
            entity = self.canonical_entities.get(node_id)
            if entity is None:
                continue
            if entity_type is None or entity.entity_type.value == entity_type:
                return node_id
        return None

    def find_canonical_nodes_in_text(self, text: str) -> List[str]:
        """Find explicit aliases in query text without an LLM classification step."""
        normalized_text = _normalize_alias(text)
        matched: List[str] = []
        for alias, node_ids in self.alias_index.items():
            compact_alias = alias.replace(" ", "")
            if len(compact_alias) < 3:
                continue
            phrase_match = re.search(
                rf"(?<![0-9a-z]){re.escape(alias)}(?![0-9a-z])",
                normalized_text,
            )
            if phrase_match is None:
                continue
            for node_id in node_ids:
                if node_id not in matched:
                    matched.append(node_id)
        return matched

    def _find_exact_canonical_name(
        self,
        normalized_name: str,
        entity_type: EntityType,
    ) -> str | None:
        for entity in self.canonical_entities.values():
            if (
                entity.entity_type == entity_type
                and _normalize_alias(entity.canonical_name) == normalized_name
            ):
                return entity.canonical_id
        return None

    def _find_scoped_alias_match(
        self,
        aliases: List[str],
        entity_type: EntityType,
        source_doc_ids: List[str],
    ) -> str | None:
        requested_docs = set(source_doc_ids)
        for alias in aliases:
            for node_id in self.alias_index.get(alias, []):
                entity = self.canonical_entities.get(node_id)
                if (
                    entity is not None
                    and entity.entity_type == entity_type
                    and requested_docs.intersection(entity.source_doc_ids)
                ):
                    return node_id
        return None

    def _index_canonical_entity(self, entity: CanonicalEntity) -> None:
        for alias, node_ids in list(self.alias_index.items()):
            self.alias_index[alias] = [
                node_id for node_id in node_ids if node_id != entity.canonical_id
            ]
            if not self.alias_index[alias]:
                del self.alias_index[alias]
        for alias in _unique_aliases([entity.canonical_name, *entity.aliases]):
            self.alias_index.setdefault(alias, []).append(entity.canonical_id)
        self.type_index.setdefault(entity.entity_type.value, [])
        if entity.canonical_id not in self.type_index[entity.entity_type.value]:
            self.type_index[entity.entity_type.value].append(entity.canonical_id)
        for doc_id in entity.source_doc_ids:
            self.doc_index.setdefault(doc_id, [])
            if entity.canonical_id not in self.doc_index[doc_id]:
                self.doc_index[doc_id].append(entity.canonical_id)
    
    def add_edge(self, edge: GraphEdge) -> Tuple[str, str]:
        """
        Add an edge to the graph.
        
        If edge exists, increases weight and merges doc_ids.
        
        Args:
            edge: GraphEdge to add.
            
        Returns:
            Tuple of (source_id, target_id).
        """
        source = edge.source_id
        target = edge.target_id
        
        if self.graph.has_edge(source, target):
            # Update existing edge
            existing = self.graph.edges[source, target]
            existing_docs = set(existing.get("doc_ids", []))
            existing_docs.update(edge.doc_ids)
            
            self.graph.edges[source, target]["doc_ids"] = list(existing_docs)
            self.graph.edges[source, target]["weight"] = min(
                existing.get("weight", 0.5) + 0.1, 1.0
            )
            existing.setdefault("edge_id", self.edge_id(source, target, edge.relation))
            logger.debug(f"Updated edge {source} -> {target}")
        else:
            # Add new edge
            self.graph.add_edge(
                source,
                target,
                edge_id=self.edge_id(source, target, edge.relation),
                relation=edge.relation,
                description=edge.description,
                weight=edge.weight,
                doc_ids=edge.doc_ids,
            )
            logger.debug(f"Added edge: {source} --[{edge.relation}]--> {target}")
        self.mark_dirty()
        
        return source, target
    
    def add_edge_from_extraction(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        doc_id: str,
        description: Optional[str] = None,
        weight: float = 1.0,
    ) -> Tuple[str, str]:
        """
        Add an edge from extraction results.
        
        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            relation: Relationship type.
            doc_id: Source document ID.
            description: Optional description.
            weight: Edge weight.
            
        Returns:
            Tuple of (source_id, target_id).
        """
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            doc_ids=[doc_id],
            description=description,
            weight=weight,
        )
        return self.add_edge(edge)
    
    def get_edges_for_node(self, node_id: str) -> List[GraphEdge]:
        """
        Get all edges connected to a node.
        
        Args:
            node_id: Node ID.
            
        Returns:
            List of GraphEdge objects.
        """
        edges = []
        
        # Outgoing edges
        for _, target, data in self.graph.out_edges(node_id, data=True):
            edges.append(GraphEdge(
                source_id=node_id,
                target_id=target,
                relation=data.get("relation", "related"),
                description=data.get("description"),
                weight=data.get("weight", 1.0),
                doc_ids=data.get("doc_ids", []),
            ))
        
        # Incoming edges
        for source, _, data in self.graph.in_edges(node_id, data=True):
            edges.append(GraphEdge(
                source_id=source,
                target_id=node_id,
                relation=data.get("relation", "related"),
                description=data.get("description"),
                weight=data.get("weight", 1.0),
                doc_ids=data.get("doc_ids", []),
            ))
        
        return edges
    
    # ===== Document Operations =====
    
    def remove_document(self, doc_id: str) -> int:
        """
        Remove all nodes and edges associated with a document.
        
        Args:
            doc_id: Document ID to remove.
            
        Returns:
            Number of nodes removed.
        """
        nodes_to_remove = []
        
        # Find nodes to remove (only referenced by this doc)
        for node_id, data in self.graph.nodes(data=True):
            doc_ids = set(data.get("doc_ids", []))
            
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                
                if not doc_ids:
                    # Node only belonged to this doc, remove it
                    nodes_to_remove.append(node_id)
                else:
                    # Update doc_ids
                    self.graph.nodes[node_id]["doc_ids"] = list(doc_ids)

        node_ids_to_remove = set(nodes_to_remove)
        provenance_ids_to_remove: Set[str] = set()
        for source, target, data in self.graph.edges(data=True):
            if source in node_ids_to_remove or target in node_ids_to_remove:
                relation = data.get("relation", "related")
                edge_id = data.get("edge_id") or self.edge_id(source, target, relation)
                provenance_ids_to_remove.add(edge_id)
        
        # Remove nodes (edges are automatically removed)
        for node_id in nodes_to_remove:
            self.graph.remove_node(node_id)
        
        # Also update edge doc_ids
        edges_to_remove = []
        for source, target, data in self.graph.edges(data=True):
            edge_docs = set(data.get("doc_ids", []))
            
            if doc_id in edge_docs:
                edge_docs.discard(doc_id)
                
                if not edge_docs:
                    relation = data.get("relation", "related")
                    edge_id = data.get("edge_id") or self.edge_id(source, target, relation)
                    provenance_ids_to_remove.add(edge_id)
                    edges_to_remove.append((source, target))
                else:
                    self.graph.edges[source, target]["doc_ids"] = list(edge_docs)
                    relation = data.get("relation", "related")
                    edge_id = data.get("edge_id") or self.edge_id(source, target, relation)
                    provenance = self.edge_provenance.get(edge_id)
                    if provenance is not None:
                        remaining_anchors = [
                            anchor
                            for anchor in provenance.anchors
                            if anchor.doc_id != doc_id
                        ]
                        if remaining_anchors:
                            self.edge_provenance[edge_id] = provenance.model_copy(
                                update={"anchors": remaining_anchors},
                                deep=True,
                            )
                        else:
                            provenance_ids_to_remove.add(edge_id)
        
        for source, target in edges_to_remove:
            self.graph.remove_edge(source, target)

        for edge_id in provenance_ids_to_remove:
            self.edge_provenance.pop(edge_id, None)
        removed_asset_ids = [
            asset_id
            for asset_id, link in self.asset_links.items()
            if link.doc_id == doc_id
        ]
        for asset_id in removed_asset_ids:
            self.asset_links.pop(asset_id, None)
        
        logger.info(
            f"Removed doc {doc_id} from graph: "
            f"{len(nodes_to_remove)} nodes, {len(edges_to_remove)} edges removed"
        )
        if nodes_to_remove or edges_to_remove or removed_asset_ids:
            self.mark_dirty()
        
        return len(nodes_to_remove)
    
    def get_documents(self) -> Set[str]:
        """
        Get all document IDs in the graph.
        
        Returns:
            Set of document IDs.
        """
        doc_ids = set()
        
        for _, data in self.graph.nodes(data=True):
            doc_ids.update(data.get("doc_ids", []))
        
        return doc_ids

    def get_document_status(self, doc_id: str) -> Optional[GraphDocumentStatus]:
        """Return persisted GraphRAG status for one document."""
        return self.document_statuses.get(doc_id)

    def record_extraction_manifest(self, manifest: GraphExtractionRunManifest) -> None:
        """Record a successful extraction run for inclusion in the next snapshot."""
        self.extraction_manifests.append(manifest)

    def get_latest_extraction_manifest(self, doc_id: str) -> Optional[GraphExtractionRunManifest]:
        """Return the newest successful extraction manifest for one document."""
        return next(
            (manifest for manifest in reversed(self.extraction_manifests) if manifest.doc_id == doc_id),
            None,
        )

    def upsert_document_status(self, status: GraphDocumentStatus) -> None:
        """Insert or replace GraphRAG status for one document."""
        self.document_statuses[status.doc_id] = status

    def remove_document_status(self, doc_id: str) -> None:
        """Remove persisted GraphRAG status for one document if present."""
        self.document_statuses.pop(doc_id, None)

    def get_all_document_statuses(self) -> List[GraphDocumentStatus]:
        """Return all persisted GraphRAG document statuses."""
        return list(self.document_statuses.values())

    def list_eligible_document_ids(self) -> List[str]:
        """Return document ids that still have OCR markdown artifacts on disk."""
        user_dir = Path(upload_paths.get_user_upload_dir(self.user_id))
        if not user_dir.exists():
            return []

        doc_ids: List[str] = []
        for entry in sorted(user_dir.iterdir(), key=lambda item: item.name):
            if not entry.is_dir():
                continue
            if (entry / GRAPH_SOURCE_MARKDOWN_FILENAME).exists():
                doc_ids.append(entry.name)
        return doc_ids

    def set_active_job_state(self, state: Optional[str]) -> None:
        """Update the currently active graph maintenance job label."""
        self.active_job_state = state

    def get_document_status_counts(self) -> Dict[str, int]:
        """Aggregate persisted document statuses by state."""
        counts = {
            "indexed": 0,
            "failed": 0,
            "partial": 0,
            "empty": 0,
        }
        for status in self.document_statuses.values():
            if status.status in counts:
                counts[status.status] += 1
        return counts
    
    # ===== Graph Analysis =====
    
    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        max_nodes: int = 50,
    ) -> List[str]:
        """
        Get neighbor nodes within N hops.
        
        Args:
            node_id: Starting node ID.
            hops: Number of hops to traverse.
            max_nodes: Maximum number of nodes to return.
            
        Returns:
            List of neighbor node IDs.
        """
        if not self.graph.has_node(node_id):
            return []
        
        visited = {node_id}
        current_level = {node_id}
        
        for _ in range(hops):
            next_level = set()
            
            for node in current_level:
                # Add successors and predecessors
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            
            next_level -= visited
            visited.update(next_level)
            current_level = next_level
            
            if len(visited) >= max_nodes:
                break
        
        # Remove starting node from results
        result = list(visited - {node_id})
        return result[:max_nodes]
    
    def get_subgraph(self, node_ids: List[str]) -> nx.DiGraph:
        """
        Extract a subgraph containing given nodes.
        
        Args:
            node_ids: List of node IDs to include.
            
        Returns:
            NetworkX DiGraph subgraph.
        """
        return self.graph.subgraph(node_ids).copy()
    
    def get_relationship_context(
        self,
        node_id: str,
        hops: int = 1,
    ) -> str:
        """
        Get human-readable relationship context for a node.
        
        Args:
            node_id: Node ID.
            hops: Number of hops.
            
        Returns:
            Formatted string describing relationships.
        """
        if not self.graph.has_node(node_id):
            return ""
        
        node_data = self.graph.nodes[node_id]
        node_label = node_data.get("label", node_id)
        
        lines = [f"實體: {node_label}"]
        
        # Outgoing relationships
        for _, target, data in self.graph.out_edges(node_id, data=True):
            target_label = self.graph.nodes[target].get("label", target)
            relation = data.get("relation", "相關")
            desc = data.get("description", "")
            
            line = f"- {node_label} [{relation}] {target_label}"
            if desc:
                line += f" ({desc})"
            lines.append(line)
        
        # Incoming relationships
        for source, _, data in self.graph.in_edges(node_id, data=True):
            source_label = self.graph.nodes[source].get("label", source)
            relation = data.get("relation", "相關")
            desc = data.get("description", "")
            
            line = f"- {source_label} [{relation}] {node_label}"
            if desc:
                line += f" ({desc})"
            lines.append(line)
        
        return "\n".join(lines)
    
    # ===== Status & Metadata =====
    
    def get_status(self) -> GraphStatusResponse:
        """
        Get graph status information.
        
        Returns:
            GraphStatusResponse with current stats.
        """
        # Count pending nodes
        pending = sum(
            1 for _, data in self.graph.nodes(data=True)
            if data.get("pending_resolution", False)
        )
        
        has_graph = self.graph.number_of_nodes() > 0
        eligible_document_ids = self.list_eligible_document_ids()
        document_counts = self.get_document_status_counts()
        needs_optimization = (
            pending > 0
            or self.graph_dirty
            or (has_graph and len(self.communities) == 0)
            or self.index_version < GRAPH_INDEX_VERSION
        )

        return GraphStatusResponse(
            has_graph=has_graph,
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
            community_count=len(self.communities),
            pending_resolution=pending,
            needs_optimization=needs_optimization,
            last_updated=self.last_updated,
            index_version=self.index_version,
            community_level_counts=self.get_community_level_counts(),
            last_optimized_at=self.last_optimized_at,
            eligible_document_count=len(eligible_document_ids),
            indexed_document_count=document_counts["indexed"],
            failed_document_count=document_counts["failed"],
            partial_document_count=document_counts["partial"],
            empty_document_count=document_counts["empty"],
            active_job_state=self.active_job_state,
        )
    
    def mark_pending_resolution(self, node_ids: List[str]) -> int:
        """
        Mark nodes as pending entity resolution.
        
        Args:
            node_ids: List of node IDs to mark.
            
        Returns:
            Number of nodes marked.
        """
        count = 0
        for node_id in node_ids:
            if self.graph.has_node(node_id):
                if not self.graph.nodes[node_id].get("pending_resolution", False):
                    self.graph.nodes[node_id]["pending_resolution"] = True
                    count += 1
        
        self._pending_count += count
        if count:
            self.mark_dirty()
        return count
    
    def clear_pending_resolution(self) -> int:
        """
        Clear pending resolution flag from all nodes.
        
        Returns:
            Number of nodes cleared.
        """
        count = 0
        for node_id in self.graph.nodes():
            if self.graph.nodes[node_id].get("pending_resolution", False):
                self.graph.nodes[node_id]["pending_resolution"] = False
                count += 1
        
        self._pending_count = 0
        return count
    
    def get_all_nodes(self) -> List[GraphNode]:
        """
        Get all nodes in the graph.
        
        Returns:
            List of GraphNode objects.
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append(GraphNode(
                id=node_id,
                label=data.get("label", ""),
                entity_type=EntityType(data.get("entity_type", "concept")),
                doc_ids=data.get("doc_ids", []),
                description=data.get("description"),
                pending_resolution=data.get("pending_resolution", False),
                embedding=data.get("embedding"),
            ))
        return nodes
    
    def get_all_edges(self) -> List[GraphEdge]:
        """
        Get all edges in the graph.
        
        Returns:
            List of GraphEdge objects.
        """
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append(GraphEdge(
                source_id=source,
                target_id=target,
                relation=data.get("relation", "related"),
                description=data.get("description"),
                weight=data.get("weight", 1.0),
                doc_ids=data.get("doc_ids", []),
            ))
        return edges

    def get_community_level_counts(self) -> Dict[str, int]:
        """Return a count of communities grouped by level."""
        counts: Dict[str, int] = {}
        for community in self.communities:
            level_key = str(community.level)
            counts[level_key] = counts.get(level_key, 0) + 1
        return counts

    def mark_optimized(self) -> None:
        """Mark the graph metadata as optimized."""
        self.index_version = GRAPH_INDEX_VERSION
        self.last_optimized_at = datetime.now()
        self.graph_dirty = False
