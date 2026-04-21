"""
GraphRAG Schemas Module

Provides Pydantic models for knowledge graph nodes, edges, communities,
and API response schemas.
"""

# Standard library
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

# Third-party
from pydantic import BaseModel, ConfigDict, Field


class EntityType(str, Enum):
    """
    Entity types for knowledge graph nodes.
    
    Defines the categories of entities that can be extracted
    from academic papers and research documents.
    """
    CONCEPT = "concept"       # Concepts/theories (e.g., "Attention Mechanism")
    METHOD = "method"         # Methods/techniques (e.g., "BERT", "Transformer")
    METRIC = "metric"         # Metrics/measures (e.g., "F1 Score", "Accuracy")
    RESULT = "result"         # Results/findings (e.g., "State-of-the-art")
    AUTHOR = "author"         # Authors (e.g., "Vaswani et al.")


class RelationType(str, Enum):
    """
    Relation types for knowledge graph edges.
    
    Defines the semantic relationships between entities.
    """
    USES = "uses"                   # Method uses another method/concept
    OUTPERFORMS = "outperforms"     # Method outperforms another
    PROPOSES = "proposes"           # Author proposes method/concept
    EVALUATES_WITH = "evaluates_with"  # Method evaluated with metric
    CITES = "cites"                 # Reference citation
    EXTENDS = "extends"             # Extends/builds upon
    PART_OF = "part_of"             # Is a component of
    APPLIES_TO = "applies_to"       # Applied to domain/task


class GraphNode(BaseModel):
    """
    Knowledge graph node representing an entity.
    
    Attributes:
        id: Unique identifier for the node.
        label: Display name of the entity.
        entity_type: Category of the entity.
        doc_ids: List of document IDs where this entity appears.
        description: Optional description of the entity.
        pending_resolution: Whether this node needs entity resolution.
        embedding: Optional embedding vector for similarity matching.
    """
    id: str = Field(..., description="唯一識別碼")
    label: str = Field(..., description="實體顯示名稱")
    entity_type: EntityType = Field(..., description="實體類別")
    doc_ids: List[str] = Field(default_factory=list, description="來源文件 ID 列表")
    description: Optional[str] = Field(default=None, description="實體描述")
    pending_resolution: bool = Field(default=False, description="是否待融合")
    embedding: Optional[List[float]] = Field(default=None, description="向量嵌入")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "node_transformer_001",
                "label": "Transformer",
                "entity_type": "method",
                "doc_ids": ["doc-uuid-1", "doc-uuid-2"],
                "description": "一種基於自注意力機制的神經網路架構",
                "pending_resolution": False,
            }
        }
    )


class GraphEdge(BaseModel):
    """
    Knowledge graph edge representing a relationship.
    
    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        relation: Type of relationship.
        description: Optional description of the relationship.
        weight: Relationship strength (0.0 to 1.0).
        doc_ids: List of document IDs where this relationship is mentioned.
    """
    source_id: str = Field(..., description="來源節點 ID")
    target_id: str = Field(..., description="目標節點 ID")
    relation: str = Field(..., description="關係類型")
    description: Optional[str] = Field(default=None, description="關係描述")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="關係強度")
    doc_ids: List[str] = Field(default_factory=list, description="來源文件 ID 列表")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_id": "node_bert_001",
                "target_id": "node_lstm_001",
                "relation": "outperforms",
                "description": "在 GLUE benchmark 上表現更優",
                "weight": 0.9,
                "doc_ids": ["doc-uuid-1"],
            }
        }
    )


class Community(BaseModel):
    """
    A community (cluster) of related nodes.
    
    Communities are detected using the Leiden algorithm and
    represent thematically related groups of entities.
    
    Attributes:
        id: Unique community identifier.
        node_ids: List of node IDs in this community.
        summary: LLM-generated summary of the community.
        title: Short title for the community.
        level: Hierarchy level (Leiden supports multi-level).
    """
    id: int = Field(..., description="社群 ID")
    node_ids: List[str] = Field(default_factory=list, description="成員節點 ID 列表")
    summary: Optional[str] = Field(default=None, description="社群摘要 (LLM 生成)")
    title: Optional[str] = Field(default=None, description="社群標題")
    level: int = Field(default=0, description="層級 (Leiden 支援多層)")
    parent_id: Optional[int] = Field(default=None, description="父社群 ID")
    child_ids: List[int] = Field(default_factory=list, description="子社群 ID 列表")
    ranking_text: Optional[str] = Field(default=None, description="檢索排序文字")
    summary_version: int = Field(default=1, description="社群摘要版本")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "node_ids": [
                    "node_transformer_001",
                    "node_bert_001",
                    "node_attention_001",
                ],
                "summary": "此社群包含與 Transformer 架構相關的方法...",
                "title": "Transformer 與注意力機制",
                "level": 0,
                "parent_id": None,
                "child_ids": [],
                "ranking_text": "Transformer 與注意力機制 此社群包含與 Transformer 架構相關的方法...",
                "summary_version": 1,
            }
        }
    )


class GraphStatusResponse(BaseModel):
    """
    Response model for graph status API endpoint.
    
    Provides overview information about the user's knowledge graph.
    """
    has_graph: bool = Field(..., description="是否有圖譜")
    node_count: int = Field(default=0, description="節點數量")
    edge_count: int = Field(default=0, description="邊數量")
    community_count: int = Field(default=0, description="社群數量")
    pending_resolution: int = Field(default=0, description="待融合實體數量")
    needs_optimization: bool = Field(default=False, description="是否需要優化")
    last_updated: Optional[datetime] = Field(default=None, description="最後更新時間")
    index_version: int = Field(default=1, description="圖譜索引版本")
    community_level_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="各層級社群數量",
    )
    last_optimized_at: Optional[datetime] = Field(default=None, description="最後優化時間")
    eligible_document_count: int = Field(default=0, description="可用於 GraphRAG 重建的文件數量")
    indexed_document_count: int = Field(default=0, description="GraphRAG 成功建入的文件數量")
    failed_document_count: int = Field(default=0, description="GraphRAG 失敗文件數量")
    partial_document_count: int = Field(default=0, description="GraphRAG 部分成功文件數量")
    empty_document_count: int = Field(default=0, description="GraphRAG 成功執行但未抽出實體的文件數量")
    active_job_state: Optional[str] = Field(default=None, description="目前進行中的圖譜工作狀態")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "has_graph": True,
                "node_count": 1250,
                "edge_count": 3400,
                "community_count": 12,
                "pending_resolution": 45,
                "needs_optimization": True,
                "index_version": 2,
                "community_level_counts": {"0": 12},
                "last_updated": "2025-12-21T10:30:00Z",
                "last_optimized_at": "2025-12-21T10:35:00Z",
                "eligible_document_count": 15,
                "indexed_document_count": 12,
                "failed_document_count": 1,
                "partial_document_count": 1,
                "empty_document_count": 1,
                "active_job_state": None,
            }
        }
    )


GraphDocumentExtractionState = Literal["indexed", "partial", "empty", "failed", "running", "skipped"]


class GraphDocumentStatus(BaseModel):
    """Persisted GraphRAG extraction status for one source document."""

    doc_id: str = Field(..., description="來源文件 ID")
    status: GraphDocumentExtractionState = Field(..., description="GraphRAG 抽取狀態")
    chunk_count: int = Field(default=0, ge=0, description="文件被切成的有效 chunk 數")
    chunks_succeeded: int = Field(default=0, ge=0, description="成功抽取的 chunk 數")
    chunks_failed: int = Field(default=0, ge=0, description="失敗的 chunk 數")
    entities_added: int = Field(default=0, ge=0, description="新增節點數")
    edges_added: int = Field(default=0, ge=0, description="新增邊數")
    last_error: Optional[str] = Field(default=None, description="最後一次錯誤訊息")
    last_attempted_at: Optional[datetime] = Field(default=None, description="最後一次嘗試時間")
    last_succeeded_at: Optional[datetime] = Field(default=None, description="最後一次成功時間")


class GraphDocumentStatusItem(GraphDocumentStatus):
    """Graph document status row returned to the frontend."""

    file_name: Optional[str] = Field(default=None, description="文件名稱")
    is_eligible: bool = Field(default=True, description="是否仍具備 OCR artifact 可重建")


class GraphDocumentStatusListResponse(BaseModel):
    """Response for GraphRAG per-document status listing."""

    documents: List[GraphDocumentStatusItem] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)


class GraphExtractionRunResult(BaseModel):
    """Internal summary of one document-level GraphRAG extraction run."""

    doc_id: str
    status: GraphDocumentExtractionState
    chunk_count: int = 0
    chunks_succeeded: int = 0
    chunks_failed: int = 0
    entities_added: int = 0
    edges_added: int = 0
    last_error: Optional[str] = None


NodeVectorSyncState = Literal["idle", "running", "completed", "failed"]


class NodeVectorSyncStatusResponse(BaseModel):
    """Response for node-vector manual sync status."""

    state: NodeVectorSyncState = Field(default="idle", description="同步狀態")
    processed: int = Field(default=0, ge=0, description="已處理節點數")
    total: int = Field(default=0, ge=0, description="總節點數")
    changed: int = Field(default=0, ge=0, description="本次重新嵌入節點數")
    reused: int = Field(default=0, ge=0, description="沿用舊向量節點數")
    removed: int = Field(default=0, ge=0, description="移除節點數")
    index_state: Optional[str] = Field(default=None, description="索引狀態")
    autosync_duration_ms: Optional[int] = Field(default=None, description="同步耗時毫秒")
    last_error: Optional[str] = Field(default=None, description="最後錯誤")
    started_at: Optional[datetime] = Field(default=None, description="開始時間")
    updated_at: Optional[datetime] = Field(default=None, description="最後更新時間")
    finished_at: Optional[datetime] = Field(default=None, description="完成時間")


class ExtractedEntity(BaseModel):
    """
    Entity extracted from text by LLM.
    
    Used as intermediate representation before adding to graph.
    """
    label: str = Field(..., description="實體名稱")
    entity_type: EntityType = Field(..., description="實體類別")
    description: Optional[str] = Field(default=None, description="實體描述")


class ExtractedRelation(BaseModel):
    """
    Relation extracted from text by LLM.
    
    Used as intermediate representation before adding to graph.
    """
    entity1: str = Field(..., description="來源實體名稱")
    entity1_type: EntityType = Field(..., description="來源實體類別")
    relation: str = Field(..., description="關係類型")
    entity2: str = Field(..., description="目標實體名稱")
    entity2_type: EntityType = Field(..., description="目標實體類別")
    description: Optional[str] = Field(default=None, description="關係描述")


class ExtractionResult(BaseModel):
    """
    Complete extraction result from a document chunk.
    """
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)
    doc_id: str = Field(..., description="來源文件 ID")
    chunk_index: int = Field(default=0, description="來源區塊索引")
