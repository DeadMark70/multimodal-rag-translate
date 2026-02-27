"""
GraphRAG Schemas Module

Provides Pydantic models for knowledge graph nodes, edges, communities,
and API response schemas.
"""

# Standard library
from datetime import datetime
from enum import Enum
from typing import List, Optional

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
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "has_graph": True,
                "node_count": 1250,
                "edge_count": 3400,
                "community_count": 12,
                "pending_resolution": 45,
                "needs_optimization": True,
                "last_updated": "2025-12-21T10:30:00Z",
            }
        }
    )


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
