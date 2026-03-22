"""
GraphRAG Extractor Module

Provides LLM-based entity and relationship extraction from text.
Uses Gemini Flash for fast, cost-effective extraction.
"""

# Standard library
import json
import logging
import re
from typing import Any, Dict, List, Sequence, Tuple

# Third-party
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Local application
from core.llm_factory import get_llm_usage_metrics, llm_runtime_override
from core.providers import get_llm
from graph_rag.schemas import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from graph_rag.llm_response import response_content_to_text
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

輸出格式規則:
- 只輸出一個 JSON 陣列，不能有任何前言、結語、註解或 Markdown code fence
- 第一個字元必須是 `[`，最後一個字元必須是 `]`
- 若沒有可抽取的實體，回傳 `[]`
- 不要重複輸出同一批資料，也不要在 JSON 後面補充說明

輸出範例:
[{{"label":"實體名稱","entity_type":"method","description":"簡短描述"}}]

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

輸出格式規則:
- 只輸出一個 JSON 陣列，不能有任何前言、結語、註解或 Markdown code fence
- 第一個字元必須是 `[`，最後一個字元必須是 `]`
- 若沒有可抽取的關係，回傳 `[]`
- 僅可使用上方已識別實體中的名稱，不要自行創造新實體
- 不要重複輸出同一批資料，也不要在 JSON 後面補充說明

輸出範例:
[{{"entity1":"來源實體","entity1_type":"method","relation":"uses","entity2":"目標實體","entity2_type":"concept","description":"關係描述"}}]

文本：
{text}

JSON 輸出："""

_ONE_PASS_EXTRACTION_PROMPT = """你是一個學術論文知識圖譜抽取專家。請從以下文本中一次完成實體與關係抽取。

實體類別 (entity_type):
- concept: 概念/理論 (例如: "注意力機制", "深度學習")
- method: 方法/技術/模型 (例如: "BERT", "Transformer", "CNN")
- metric: 指標/度量 (例如: "F1 分數", "準確率", "BLEU")
- result: 結果/發現 (例如: "最先進性能", "顯著改進")
- author: 作者/研究者 (例如: "Vaswani et al.")

關係類型 (relation):
- uses: 使用 (A 使用 B)
- outperforms: 優於 (A 的表現優於 B)
- proposes: 提出 (作者提出方法)
- evaluates_with: 用...評估 (用指標評估方法)
- cites: 引用 (引用其他工作)
- extends: 擴展 (基於...擴展)
- part_of: 是...的一部分
- applies_to: 應用於

抽取規則:
- 只抽取文本中明確提到的重要學術實體與關係
- 每個 entity 都要有唯一 `id`，relation 必須引用這些 entity `id`
- 不要為了補齊圖譜而猜測文本中不存在的關係
- 如果同一個實體重複出現，可以重用同一個實體概念
- 若無法同時提供 `source_entity_id`、`target_entity_id`、`relation` 三個欄位，該 relation 就不要輸出
- 每個 relation 都必須引用 `entities` 陣列中已出現的 `id`
- 只輸出單一 JSON 物件，不能有任何前言、結語、註解或 Markdown code fence
- JSON 物件必須符合此形狀：`{{"entities":[...],"relations":[...]}}`
- 若沒有可抽取內容，回傳 `{{"entities":[],"relations":[]}}`
- 不要重複輸出同一個 JSON 物件，也不要在 JSON 前後補充說明

文本：
{text}
"""


class _StructuredEntity(BaseModel):
    """Private schema for one-pass Gemini structured output."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique entity id referenced by relations")
    label: str = Field(..., description="Entity label from the text")
    entity_type: str = Field(default="concept", description="Academic entity type")
    description: str | None = Field(default=None, description="Short entity description")


class _StructuredRelation(BaseModel):
    """Private schema for one-pass Gemini structured output."""

    model_config = ConfigDict(extra="ignore")

    source_entity_id: str = Field(..., description="Source entity id")
    target_entity_id: str = Field(..., description="Target entity id")
    relation: str = Field(..., description="Relation type between the entities")
    description: str | None = Field(default=None, description="Short relation description")


class _StructuredExtractionPayload(BaseModel):
    """Private schema for one-pass Gemini structured output."""

    model_config = ConfigDict(extra="ignore")

    entities: List[_StructuredEntity] = Field(default_factory=list)
    relations: List[_StructuredRelation] = Field(default_factory=list)


def _parse_json_from_response(
    response: Any,
    *,
    list_keys: Sequence[str] | None = None,
) -> List[dict]:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        response: Raw LLM response text.
        
    Returns:
        Parsed list of dictionaries.
    """
    response_text = response_content_to_text(response)

    if not response_text:
        logger.warning("Failed to parse JSON: empty response text")
        return []

    response_text = response_text.strip().lstrip("\ufeff")

    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find raw JSON array
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            object_match = re.search(r"\{[\s\S]*\}", response_text)
            json_str = object_match.group(0) if object_match else response_text

    try:
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(json_str.lstrip())
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in list_keys or ():
                value = result.get(key)
                if isinstance(value, list):
                    return value
        return []
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON: %s", e)
        return []


def _log_thinking_usage(call_name: str, response: Any) -> None:
    """Log reasoning-token usage for GraphRAG calls when available."""
    usage = get_llm_usage_metrics(response)
    logger.info(
        "%s usage: input_tokens=%s output_tokens=%s total_tokens=%s reasoning_tokens=%s",
        call_name,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["total_tokens"],
        usage["reasoning_tokens"],
    )


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


def _normalize_label(label: str) -> str:
    """Normalize entity labels for deduplication and validation."""
    return label.strip().lower()


def _coerce_structured_payload(raw_payload: Any) -> _StructuredExtractionPayload:
    """Validate raw structured output into the private extraction payload."""
    if isinstance(raw_payload, _StructuredExtractionPayload):
        return raw_payload
    if isinstance(raw_payload, BaseModel):
        raw_payload = raw_payload.model_dump()
    if isinstance(raw_payload, dict):
        if "parsed" in raw_payload or "raw" in raw_payload:
            parsed_payload = raw_payload.get("parsed")
            if parsed_payload is not None:
                return _coerce_structured_payload(parsed_payload)

            parsing_error = raw_payload.get("parsing_error")
            raw_message = raw_payload.get("raw")
            raw_preview = response_content_to_text(
                getattr(raw_message, "content", raw_message)
            )
            raw_preview = raw_preview[:200] if raw_preview else ""
            detail_parts = []
            if parsing_error:
                detail_parts.append(f"parsing_error={parsing_error}")
            if raw_preview:
                detail_parts.append(f"raw_preview={raw_preview!r}")
            detail = ", ".join(detail_parts) or "parsed payload was None"
            raise RuntimeError(f"Structured output returned no parsed payload ({detail})")
        return _StructuredExtractionPayload.model_validate(raw_payload)
    raise TypeError(f"Unexpected structured payload type: {type(raw_payload)!r}")


class EntityRelationExtractor:
    """
    Extracts entities and relationships from text using LLM.
    
    Uses Gemini Flash for fast, cost-effective extraction.
    Defaults to one-pass structured extraction with hidden two-stage fallback.
    
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

    async def _extract_one_pass_structured(
        self,
        text: str,
    ) -> tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Extract entities and relations in a single structured-output call.

        Raises:
            Exception: Any model setup / invocation / validation failure.
        """
        with llm_runtime_override(thinking_budget=-1, include_thoughts=False):
            llm = get_llm("graph_extraction")
            if not hasattr(llm, "with_structured_output"):
                raise RuntimeError("graph_extraction model does not support structured output")

            try:
                structured_llm = llm.with_structured_output(
                    _StructuredExtractionPayload,
                    include_raw=True,
                )
            except TypeError:
                structured_llm = llm.with_structured_output(
                    schema=_StructuredExtractionPayload,
                    include_raw=True,
                )
            prompt = _ONE_PASS_EXTRACTION_PROMPT.format(text=text[:4000])
            raw_payload = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        if isinstance(raw_payload, dict):
            raw_response = raw_payload.get("raw")
            if raw_response is not None:
                _log_thinking_usage("Structured graph extraction", raw_response)
        payload = _coerce_structured_payload(raw_payload)
        return self._build_extraction_from_payload(payload)

    def _build_extraction_from_payload(
        self,
        payload: _StructuredExtractionPayload,
    ) -> tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """Convert structured payload into existing extraction result types."""
        entities: List[ExtractedEntity] = []
        canonical_entities: Dict[str, ExtractedEntity] = {}
        canonical_ids_by_label: Dict[str, str] = {}
        entity_aliases: Dict[str, str] = {}

        for item in payload.entities:
            normalized_label = _normalize_label(item.label)
            if not normalized_label:
                continue

            existing_canonical_id = canonical_ids_by_label.get(normalized_label)
            if existing_canonical_id:
                entity_aliases[item.id] = existing_canonical_id
                continue

            if len(entities) >= self.max_entities_per_chunk:
                continue

            entity = ExtractedEntity(
                label=item.label.strip(),
                entity_type=_normalize_entity_type(item.entity_type),
                description=item.description,
            )
            entities.append(entity)
            canonical_entities[item.id] = entity
            canonical_ids_by_label[normalized_label] = item.id
            entity_aliases[item.id] = item.id

        relations: List[ExtractedRelation] = []
        for item in payload.relations:
            source_id = entity_aliases.get(item.source_entity_id)
            target_id = entity_aliases.get(item.target_entity_id)
            if not source_id or not target_id:
                logger.debug(
                    "Skipping structured relation: missing entity id(s) %s -> %s",
                    item.source_entity_id,
                    item.target_entity_id,
                )
                continue

            source_entity = canonical_entities.get(source_id)
            target_entity = canonical_entities.get(target_id)
            if not source_entity or not target_entity:
                continue

            try:
                relation = ExtractedRelation(
                    entity1=source_entity.label,
                    entity1_type=source_entity.entity_type,
                    relation=item.relation.strip().lower() or "related",
                    entity2=target_entity.label,
                    entity2_type=target_entity.entity_type,
                    description=item.description,
                )
                relations.append(relation)
            except ValidationError as e:
                logger.debug(f"Skipping invalid structured relation: {e}")
                continue

        logger.info(
            "Structured GraphRAG extraction produced %s entities and %s relations",
            len(entities),
            len(relations),
        )
        return entities, relations
    
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
            with llm_runtime_override(thinking_budget=-1, include_thoughts=False):
                llm = get_llm("graph_extraction")
                prompt = _ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])  # Limit input
                response = await llm.ainvoke([HumanMessage(content=prompt)])
            _log_thinking_usage("Legacy entity extraction", response)
            raw_entities = _parse_json_from_response(
                response.content,
                list_keys=("entities", "items", "data"),
            )
            
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
            with llm_runtime_override(thinking_budget=-1, include_thoughts=False):
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
            _log_thinking_usage("Legacy relation extraction", response)
            raw_relations = _parse_json_from_response(
                response.content,
                list_keys=("relations", "items", "data"),
            )
            
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
        if len(text.strip()) < self.min_text_length:
            entities: List[ExtractedEntity] = []
            relations: List[ExtractedRelation] = []
        else:
            try:
                entities, relations = await self._extract_one_pass_structured(text)
            except Exception as e:
                logger.warning(
                    "Structured GraphRAG extraction failed; falling back to legacy two-pass flow: %s",
                    e,
                )
                entities = await self.extract_entities(text)
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
