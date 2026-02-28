"""
GraphRAG Extractor Module

Provides LLM-based entity and relationship extraction from text.
Uses Gemini Flash for fast, cost-effective extraction.
"""

# Standard library
import json
import logging
import re
from typing import List, Tuple

# Third-party
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

# Local application
from core.providers import get_llm
from graph_rag.schemas import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from graph_rag.store import GraphStore

# Configure logging
logger = logging.getLogger(__name__)


# ===== Extraction Prompts =====

_ENTITY_EXTRACTION_PROMPT = """你是一個學術論文實體抽取專家。請從以下文本中識別重要的學術實體。

實體類別 (entity_type):
- concept: 概念/理論 (例如: "注意力機制", "深度學習")
- method: 方法/技術/模型 (例如: "BERT", "Transformer", "CNN")
- metric: 指標/度量 (例如: "F1 分數", "準確率", "BLEU")
- result: 結果/發現 (例如: "最先進性能", "顯著改進")
- author: 作者/研究者 (例如: "Vaswani et al.")

請嚴格以 JSON 格式輸出，不要添加其他文字：
```json
[
  {{"label": "實體名稱", "entity_type": "method", "description": "簡短描述"}},
  ...
]
```

文本：
{text}

JSON 輸出："""

_RELATION_EXTRACTION_PROMPT = """你是一個學術論文關係抽取專家。請從以下文本中識別實體之間的關係。

已識別的實體：
{entities}

關係類型 (relation):
- uses: 使用 (A 使用 B)
- outperforms: 優於 (A 的表現優於 B)
- proposes: 提出 (作者提出方法)
- evaluates_with: 用...評估 (用指標評估方法)
- cites: 引用 (引用其他工作)
- extends: 擴展 (基於...擴展)
- part_of: 是...的一部分
- applies_to: 應用於

請嚴格以 JSON 格式輸出，不要添加其他文字：
```json
[
  {{"entity1": "來源實體", "entity1_type": "method", "relation": "uses", "entity2": "目標實體", "entity2_type": "concept", "description": "關係描述"}},
  ...
]
```

文本：
{text}

JSON 輸出："""


def _parse_json_from_response(response: str) -> List[dict]:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        response: Raw LLM response text.
        
    Returns:
        Parsed list of dictionaries.
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find raw JSON array
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response.strip()
    
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return []


def _normalize_entity_type(type_str: str) -> EntityType:
    """
    Normalize entity type string to EntityType enum.
    
    Args:
        type_str: Raw entity type string.
        
    Returns:
        Normalized EntityType.
    """
    type_map = {
        "concept": EntityType.CONCEPT,
        "method": EntityType.METHOD,
        "metric": EntityType.METRIC,
        "result": EntityType.RESULT,
        "author": EntityType.AUTHOR,
        "技術": EntityType.METHOD,
        "方法": EntityType.METHOD,
        "模型": EntityType.METHOD,
        "概念": EntityType.CONCEPT,
        "理論": EntityType.CONCEPT,
        "指標": EntityType.METRIC,
        "結果": EntityType.RESULT,
        "作者": EntityType.AUTHOR,
    }
    return type_map.get(type_str.lower().strip(), EntityType.CONCEPT)


class EntityRelationExtractor:
    """
    Extracts entities and relationships from text using LLM.
    
    Uses Gemini Flash for fast, cost-effective extraction.
    Implements two-stage extraction: entities first, then relations.
    
    Attributes:
        min_text_length: Minimum text length to process.
        max_entities_per_chunk: Maximum entities to extract per chunk.
    """
    
    def __init__(
        self,
        min_text_length: int = 50,
        max_entities_per_chunk: int = 20,
    ) -> None:
        """
        Initialize the extractor.
        
        Args:
            min_text_length: Skip text shorter than this.
            max_entities_per_chunk: Limit entities per chunk.
        """
        self.min_text_length = min_text_length
        self.max_entities_per_chunk = max_entities_per_chunk
    
    async def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text using LLM.
        
        Args:
            text: Input text to extract from.
            
        Returns:
            List of extracted entities.
        """
        if len(text.strip()) < self.min_text_length:
            return []
        
        try:
            llm = get_llm("graph_extraction")
            prompt = _ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])  # Limit input
            
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw_entities = _parse_json_from_response(response.content)
            
            entities = []
            for item in raw_entities[:self.max_entities_per_chunk]:
                try:
                    entity = ExtractedEntity(
                        label=item.get("label", "").strip(),
                        entity_type=_normalize_entity_type(item.get("entity_type", "concept")),
                        description=item.get("description"),
                    )
                    if entity.label:  # Skip empty labels
                        entities.append(entity)
                except (ValidationError, KeyError) as e:
                    logger.debug(f"Skipping invalid entity: {e}")
                    continue
            
            logger.info(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def extract_relations(
        self,
        text: str,
        entities: List[ExtractedEntity],
    ) -> List[ExtractedRelation]:
        """
        Extract relationships between entities using LLM.
        
        Args:
            text: Input text.
            entities: Previously extracted entities.
            
        Returns:
            List of extracted relations.
        """
        if not entities or len(text.strip()) < self.min_text_length:
            return []
        
        try:
            llm = get_llm("graph_extraction")
            
            # Format entities for prompt
            entity_str = "\n".join([
                f"- {e.label} ({e.entity_type.value})"
                for e in entities
            ])
            
            prompt = _RELATION_EXTRACTION_PROMPT.format(
                entities=entity_str,
                text=text[:4000],
            )
            
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw_relations = _parse_json_from_response(response.content)
            
            # Build a set of valid entity labels for validation
            valid_labels = {e.label.lower() for e in entities}
            
            relations = []
            for item in raw_relations:
                try:
                    entity1 = item.get("entity1", "").strip()
                    entity2 = item.get("entity2", "").strip()
                    
                    # Validate that entities were actually extracted
                    if entity1.lower() not in valid_labels:
                        logger.debug(f"Skipping relation: entity1 '{entity1}' not in entities")
                        continue
                    if entity2.lower() not in valid_labels:
                        logger.debug(f"Skipping relation: entity2 '{entity2}' not in entities")
                        continue
                    
                    relation = ExtractedRelation(
                        entity1=entity1,
                        entity1_type=_normalize_entity_type(item.get("entity1_type", "concept")),
                        relation=item.get("relation", "related").lower(),
                        entity2=entity2,
                        entity2_type=_normalize_entity_type(item.get("entity2_type", "concept")),
                        description=item.get("description"),
                    )
                    relations.append(relation)
                    
                except (ValidationError, KeyError) as e:
                    logger.debug(f"Skipping invalid relation: {e}")
                    continue
            
            logger.info(f"Extracted {len(relations)} relations from text")
            return relations
            
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []
    
    async def extract(
        self,
        text: str,
        doc_id: str,
        chunk_index: int = 0,
    ) -> ExtractionResult:
        """
        Perform full extraction (entities + relations) on text.
        
        Args:
            text: Input text.
            doc_id: Source document ID.
            chunk_index: Index of chunk within document.
            
        Returns:
            ExtractionResult containing entities and relations.
        """
        # Stage 1: Extract entities
        entities = await self.extract_entities(text)
        
        # Stage 2: Extract relations (only if we have entities)
        relations = []
        if entities:
            relations = await self.extract_relations(text, entities)
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            doc_id=doc_id,
            chunk_index=chunk_index,
        )


async def extract_from_chunk(
    text: str,
    doc_id: str,
    chunk_index: int = 0,
) -> ExtractionResult:
    """
    Convenience function to extract from a single chunk.
    
    Args:
        text: Text to extract from.
        doc_id: Document ID.
        chunk_index: Chunk index.
        
    Returns:
        ExtractionResult.
    """
    extractor = EntityRelationExtractor()
    return await extractor.extract(text, doc_id, chunk_index)


async def add_extraction_to_graph(
    store: GraphStore,
    result: ExtractionResult,
) -> Tuple[int, int]:
    """
    Add extraction results to a graph store.
    
    Args:
        store: GraphStore instance.
        result: Extraction result to add.
        
    Returns:
        Tuple of (nodes_added, edges_added).
    """
    nodes_added = 0
    edges_added = 0
    
    # Map from entity label to node ID
    label_to_node_id = {}
    
    # Add entities as nodes
    for entity in result.entities:
        node_id = store.add_node_from_extraction(
            label=entity.label,
            entity_type=entity.entity_type,
            doc_id=result.doc_id,
            description=entity.description,
            pending_resolution=True,
        )
        label_to_node_id[entity.label.lower()] = node_id
        nodes_added += 1
    
    # Add relations as edges
    for relation in result.relations:
        source_id = label_to_node_id.get(relation.entity1.lower())
        target_id = label_to_node_id.get(relation.entity2.lower())
        
        if source_id and target_id:
            store.add_edge_from_extraction(
                source_id=source_id,
                target_id=target_id,
                relation=relation.relation,
                doc_id=result.doc_id,
                description=relation.description,
            )
            edges_added += 1
    
    return nodes_added, edges_added


async def extract_and_add_to_graph(
    text: str,
    doc_id: str,
    store: GraphStore,
    chunk_index: int = 0,
) -> Tuple[int, int]:
    """
    Extract from text and add directly to graph.
    
    Convenience function combining extraction and graph update.
    
    Args:
        text: Text to extract from.
        doc_id: Document ID.
        store: GraphStore to update.
        chunk_index: Chunk index.
        
    Returns:
        Tuple of (nodes_added, edges_added).
    """
    result = await extract_from_chunk(text, doc_id, chunk_index)
    return await add_extraction_to_graph(store, result)
