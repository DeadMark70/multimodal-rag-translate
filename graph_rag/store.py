"""
GraphRAG Store Module

Provides NetworkX-based graph storage with serialization support.
Manages per-user knowledge graphs with CRUD operations.
"""

# Standard library
import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Third-party
import networkx as nx

# Local application
from core import uploads as upload_paths
from graph_rag.schemas import (
    Community,
    GraphDocumentStatus,
    EntityType,
    GraphEdge,
    GraphNode,
    GraphStatusResponse,
)

# Configure logging
logger = logging.getLogger(__name__)

GRAPH_INDEX_VERSION = 2
GRAPH_DOCUMENTS_FILENAME = "graph.documents.json"
GRAPH_SOURCE_MARKDOWN_FILENAME = "extracted.md"
NODE_VECTOR_INDEX_NAME = "node_index"
NODE_VECTOR_MAP_FILENAME = "node_index_map.json"
NODE_VECTOR_META_FILENAME = "node_index.meta.json"


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
        self._storage_dir = (
            Path(storage_dir)
            if storage_dir is not None
            else upload_paths.get_rag_index_dir_path(self.user_id)
        )
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self.graph: nx.DiGraph = nx.DiGraph()
        self.communities: List[Community] = []
        self.document_statuses: Dict[str, GraphDocumentStatus] = {}
        self.last_updated: Optional[datetime] = None
        self.last_optimized_at: Optional[datetime] = None
        self.index_version: int = GRAPH_INDEX_VERSION
        self.graph_dirty: bool = False
        self.node_vector_dirty: bool = False
        self._pending_count: int = 0
        self.active_job_state: Optional[str] = None
        
        # Try to load existing graph; if only sidecars exist, still recover metadata/status.
        if self._graph_exists():
            self.load()
        else:
            self._load_metadata()
            self._load_document_statuses()
    
    def _get_graph_path(self) -> Path:
        """
        Get path to graph pickle file.
        
        Returns:
            Path to graph.pkl file.
        """
        return self._storage_dir / "graph.pkl"
    
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
            "pending_count": self._pending_count,
            "community_level_counts": self.get_community_level_counts(),
            "active_job_state": self.active_job_state,
        }

    def _write_metadata(self) -> None:
        """Persist sidecar metadata JSON."""
        path = self._get_metadata_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._build_metadata_payload(), f, ensure_ascii=False, indent=2)

    def _write_document_statuses(self) -> None:
        """Persist GraphRAG per-document status JSON."""
        path = self._get_document_status_path()
        payload = {
            "documents": [
                status.model_dump(mode="json")
                for status in sorted(self.document_statuses.values(), key=lambda item: item.doc_id)
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

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

    def save_sidecars(self) -> None:
        """Persist metadata-only sidecars without rewriting the graph pickle."""
        self._write_metadata()
        self._write_document_statuses()

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
        self._pending_count = 0
        self.last_optimized_at = None
        self.graph_dirty = False
        self.node_vector_dirty = False
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
            logger.debug(f"Updated edge {source} -> {target}")
        else:
            # Add new edge
            self.graph.add_edge(
                source,
                target,
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
                    edges_to_remove.append((source, target))
                else:
                    self.graph.edges[source, target]["doc_ids"] = list(edge_docs)
        
        for source, target in edges_to_remove:
            self.graph.remove_edge(source, target)
        
        logger.info(
            f"Removed doc {doc_id} from graph: "
            f"{len(nodes_to_remove)} nodes, {len(edges_to_remove)} edges removed"
        )
        if nodes_to_remove or edges_to_remove:
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
