"""
GraphRAG Extractor Module

Provides LLM-based entity and relationship extraction from text.
Uses Gemini Flash for fast, cost-effective extraction.
"""

# Standard library
import json
import logging
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

# Third-party
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Local application
from core.llm_factory import get_llm_usage_metrics, graph_rag_llm_runtime_override
from core.providers import get_llm
from core.prompt_loader import format_graph_rag_prompt
from graph_rag.schemas import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    EvidenceAnchor,
    ExtractionResult,
    GRAPH_EDGE_TYPES_V1,
    GRAPH_NODE_TYPES_V1,
    RawGraphCandidate,
)
from graph_rag.llm_response import response_content_to_text
from graph_rag.store import GraphStore

# Configure logging
logger = logging.getLogger(__name__)


# ===== Extraction Prompts =====


class _StructuredEntity(BaseModel):
    """Private schema for one-pass Gemini structured output."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique entity id referenced by relations")
    label: str = Field(..., description="Entity label from the text")
    entity_type: str = Field(default="concept", description="Academic entity type")
    description: str | None = Field(default=None, description="Short entity description")
    canonical_name: str | None = Field(default=None, description="Canonical entity name")
    aliases: List[str] = Field(default_factory=list, description="Known entity aliases")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_quote: str | None = Field(default=None, description="Exact source quote")


class _StructuredRelation(BaseModel):
    """Private schema for one-pass Gemini structured output."""

    model_config = ConfigDict(extra="ignore")

    source_entity_id: str = Field(..., description="Source entity id")
    target_entity_id: str = Field(..., description="Target entity id")
    relation: str = Field(..., description="Relation type between the entities")
    description: str | None = Field(default=None, description="Short relation description")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_quote: str | None = Field(default=None, description="Exact source quote")


class _StructuredExtractionPayload(BaseModel):
    """Private schema for one-pass Gemini structured output."""

    model_config = ConfigDict(extra="ignore")

    entities: List[_StructuredEntity] = Field(default_factory=list)
    relations: List[_StructuredRelation] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RelationSchemaDecision:
    allowed: bool
    normalized_relation: str


@dataclass(frozen=True, slots=True)
class NodeSchemaDecision:
    allowed: bool
    entity_type: EntityType | None


_NODE_TYPE_ALIASES = {
    "architecture": "architecture_component",
    "component": "architecture_component",
    "claimscope": "claim_scope",
    "training setting": "training_setting",
    "benchmark setting": "benchmark_setting",
    "prompt type": "prompt_type",
    "evidence span": "evidence_span",
}


def _normalize_schema_value(value: str) -> str:
    return re.sub(r"_+", "_", value.strip().lower().replace("-", " ").replace(" ", "_"))


def classify_relation_for_answer_graph(relation: str) -> RelationSchemaDecision:
    """Allow only declared relation values to enter the answer graph."""
    normalized = _normalize_schema_value(relation)
    if normalized in GRAPH_EDGE_TYPES_V1:
        return RelationSchemaDecision(True, normalized)
    return RelationSchemaDecision(False, "unknown_relation")


def _classify_node_for_answer_graph(entity_type: str) -> NodeSchemaDecision:
    normalized = _normalize_schema_value(entity_type)
    normalized = _NODE_TYPE_ALIASES.get(normalized, normalized)
    if normalized in GRAPH_NODE_TYPES_V1:
        return NodeSchemaDecision(True, EntityType(normalized))
    return NodeSchemaDecision(False, None)


def _candidate_id(
    *,
    candidate_type: str,
    payload: Dict[str, object],
    doc_id: str,
    chunk_index: int,
) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(
        f"{candidate_type}|{doc_id}|{chunk_index}|{raw}".encode("utf-8")
    ).hexdigest()[:20]
    return f"raw:{digest}"


def _raw_candidate(
    *,
    candidate_type: str,
    payload: Dict[str, object],
    doc_id: str,
    chunk_index: int,
    confidence: float,
) -> RawGraphCandidate:
    return RawGraphCandidate(
        candidate_id=_candidate_id(
            candidate_type=candidate_type,
            payload=payload,
            doc_id=doc_id,
            chunk_index=chunk_index,
        ),
        candidate_type=candidate_type,
        payload=payload,
        source_doc_id=doc_id,
        source_chunk_index=chunk_index,
        confidence=confidence,
        needs_review=True,
    )


def _verified_text_anchor(
    *,
    text: str,
    doc_id: str,
    chunk_index: int,
    quote: str,
    confidence: float,
) -> EvidenceAnchor | None:
    normalized_quote = quote.strip()
    start = text.find(normalized_quote)
    if not normalized_quote or start < 0:
        return None
    chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    quote_hash = hashlib.sha256(normalized_quote.encode("utf-8")).hexdigest()
    return EvidenceAnchor(
        doc_id=doc_id,
        chunk_id=f"graph:{doc_id}:chunk:{chunk_index}",
        chunk_index=chunk_index,
        quote=normalized_quote,
        quote_hash=quote_hash,
        chunk_hash=chunk_hash,
        source_text_hash=chunk_hash,
        markdown_char_start=start,
        markdown_char_end=start + len(normalized_quote),
        confidence=confidence,
        extraction_prompt_version="graph-extract-v2",
        verification_status="quote_match",
    )


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
        return _StructuredExtractionPayload.model_validate(raw_payload)
    raise TypeError(f"Unexpected structured payload type: {type(raw_payload)!r}")


def _extract_json_object_text(raw_text: str) -> str:
    """Extract the first JSON object from a model response string."""
    response_text = raw_text.strip().lstrip("\ufeff")
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if json_match:
        candidate = json_match.group(1).strip()
    else:
        object_match = re.search(r"\{[\s\S]*\}", response_text)
        if not object_match:
            preview = response_text[:200]
            raise RuntimeError(
                f"Structured output returned no JSON object (raw_preview={preview!r})"
            )
        candidate = object_match.group(0)
    return candidate


def _coerce_structured_payload_from_response(response: Any) -> _StructuredExtractionPayload:
    """Parse and validate a native JSON-schema response payload."""
    response_text = response_content_to_text(getattr(response, "content", response))
    if not response_text:
        raise RuntimeError("Structured output returned empty response text")

    try:
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(_extract_json_object_text(response_text).lstrip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Structured output returned invalid JSON ({exc})") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Structured output returned non-object payload ({type(payload)!r})"
        )

    return _coerce_structured_payload(payload)


def _coerce_structured_payload_from_raw_payload(raw_payload: Any) -> _StructuredExtractionPayload:
    """Recover structured payload from LangChain's include_raw response shape."""
    if not isinstance(raw_payload, dict):
        raise TypeError(f"Unexpected structured raw payload type: {type(raw_payload)!r}")

    parsed_payload = raw_payload.get("parsed")
    if parsed_payload is not None:
        return _coerce_structured_payload(parsed_payload)

    raw_message = raw_payload.get("raw")
    if raw_message is not None:
        return _coerce_structured_payload_from_response(raw_message)

    parsing_error = raw_payload.get("parsing_error")
    detail = f"parsing_error={parsing_error}" if parsing_error else "parsed payload was None"
    raise RuntimeError(f"Structured output returned no parsed payload ({detail})")


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
        doc_id: str,
        chunk_index: int,
    ) -> tuple[List[ExtractedEntity], List[ExtractedRelation], List[RawGraphCandidate]]:
        """
        Extract entities and relations in a single structured-output call.

        Raises:
            Exception: Any model setup / invocation / validation failure.
        """
        prompt = format_graph_rag_prompt("one_pass_extraction_schema_v1", text=text[:4000])
        with graph_rag_llm_runtime_override("graph_extraction"):
            llm = get_llm("graph_extraction")
            try:
                structured_llm = llm.bind(
                    response_mime_type="application/json",
                    response_json_schema=_StructuredExtractionPayload.model_json_schema(),
                )
                response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
                _log_thinking_usage("Structured graph extraction", response)
                payload = _coerce_structured_payload_from_response(response)
            except TypeError as exc:
                if "response_mime_type" not in str(exc):
                    raise
                logger.info(
                    "Native JSON-schema binding unsupported for current Gemini client; "
                    "falling back to include_raw structured output: %s",
                    exc,
                )
                if not hasattr(llm, "with_structured_output"):
                    raise RuntimeError(
                        "graph_extraction model does not support structured output"
                    ) from exc
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
                raw_payload = await structured_llm.ainvoke([HumanMessage(content=prompt)])
                if isinstance(raw_payload, dict):
                    raw_response = raw_payload.get("raw")
                    if raw_response is not None:
                        _log_thinking_usage("Structured graph extraction", raw_response)
                payload = _coerce_structured_payload_from_raw_payload(raw_payload)
        return self._build_extraction_from_payload(
            payload,
            text=text,
            doc_id=doc_id,
            chunk_index=chunk_index,
        )

    def _build_extraction_from_payload(
        self,
        payload: _StructuredExtractionPayload,
        *,
        text: str,
        doc_id: str,
        chunk_index: int,
    ) -> tuple[List[ExtractedEntity], List[ExtractedRelation], List[RawGraphCandidate]]:
        """Convert structured payload into existing extraction result types."""
        entities: List[ExtractedEntity] = []
        canonical_entities: Dict[str, ExtractedEntity] = {}
        canonical_ids_by_label: Dict[str, str] = {}
        entity_aliases: Dict[str, str] = {}
        raw_candidates: List[RawGraphCandidate] = []

        for item in payload.entities:
            normalized_label = _normalize_label(item.label)
            if not normalized_label:
                raw_candidates.append(
                    _raw_candidate(
                        candidate_type="missing_entity_label",
                        payload=item.model_dump(mode="json"),
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        confidence=item.confidence,
                    )
                )
                continue

            node_decision = _classify_node_for_answer_graph(item.entity_type)
            if not node_decision.allowed or node_decision.entity_type is None:
                raw_candidates.append(
                    _raw_candidate(
                        candidate_type="unknown_node_type",
                        payload=item.model_dump(mode="json"),
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        confidence=item.confidence,
                    )
                )
                continue

            anchors: List[EvidenceAnchor] = []
            if item.evidence_quote:
                anchor = _verified_text_anchor(
                    text=text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    quote=item.evidence_quote,
                    confidence=item.confidence,
                )
                if anchor is None:
                    raw_candidates.append(
                        _raw_candidate(
                            candidate_type="quote_mismatch",
                            payload=item.model_dump(mode="json"),
                            doc_id=doc_id,
                            chunk_index=chunk_index,
                            confidence=item.confidence,
                        )
                    )
                    continue
                anchors.append(anchor)

            existing_canonical_id = canonical_ids_by_label.get(normalized_label)
            if existing_canonical_id:
                entity_aliases[item.id] = existing_canonical_id
                continue

            if len(entities) >= self.max_entities_per_chunk:
                continue

            entity = ExtractedEntity(
                label=item.label.strip(),
                entity_type=node_decision.entity_type,
                description=item.description,
                canonical_name=(item.canonical_name or item.label).strip(),
                aliases=[alias.strip() for alias in item.aliases if alias.strip()],
                confidence=item.confidence,
                anchors=anchors,
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
                raw_candidates.append(
                    _raw_candidate(
                        candidate_type="unknown_entity_reference",
                        payload=item.model_dump(mode="json"),
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        confidence=item.confidence,
                    )
                )
                continue

            source_entity = canonical_entities.get(source_id)
            target_entity = canonical_entities.get(target_id)
            if not source_entity or not target_entity:
                continue

            relation_decision = classify_relation_for_answer_graph(item.relation)
            if not relation_decision.allowed:
                raw_candidates.append(
                    _raw_candidate(
                        candidate_type="unknown_relation",
                        payload=item.model_dump(mode="json"),
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        confidence=item.confidence,
                    )
                )
                continue

            anchors: List[EvidenceAnchor] = []
            if item.evidence_quote:
                anchor = _verified_text_anchor(
                    text=text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    quote=item.evidence_quote,
                    confidence=item.confidence,
                )
                if anchor is None:
                    raw_candidates.append(
                        _raw_candidate(
                            candidate_type="quote_mismatch",
                            payload=item.model_dump(mode="json"),
                            doc_id=doc_id,
                            chunk_index=chunk_index,
                            confidence=item.confidence,
                        )
                    )
                    continue
                anchors.append(anchor)

            try:
                relation = ExtractedRelation(
                    entity1=source_entity.label,
                    entity1_type=source_entity.entity_type,
                    relation=relation_decision.normalized_relation,
                    entity2=target_entity.label,
                    entity2_type=target_entity.entity_type,
                    description=item.description,
                    confidence=item.confidence,
                    anchors=anchors,
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
        return entities, relations, raw_candidates
    
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
            with graph_rag_llm_runtime_override("graph_extraction"):
                llm = get_llm("graph_extraction")
                prompt = format_graph_rag_prompt("entity_extraction", text=text[:4000])  # Limit input
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
            with graph_rag_llm_runtime_override("graph_extraction"):
                llm = get_llm("graph_extraction")

                # Format entities for prompt
                entity_str = "\n".join([
                    f"- {e.label} ({e.entity_type.value})"
                    for e in entities
                ])

                prompt = format_graph_rag_prompt(
                    "relation_extraction",
                    text=text[:4000],
                    entities=entity_str,
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
            raw_candidates: List[RawGraphCandidate] = []
        else:
            try:
                entities, relations, raw_candidates = await self._extract_one_pass_structured(
                    text,
                    doc_id,
                    chunk_index,
                )
            except Exception as e:
                logger.warning(
                    "Structured GraphRAG extraction failed; falling back to legacy two-pass flow: %s",
                    e,
                )
                entities = await self.extract_entities(text)
                relations = []
                if entities:
                    relations = await self.extract_relations(text, entities)
                raw_candidates = []
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            doc_id=doc_id,
            chunk_index=chunk_index,
            raw_candidates=raw_candidates,
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

    for candidate in result.raw_candidates:
        store.record_raw_candidate(candidate)

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
            if relation.anchors:
                store.record_edge_provenance(
                    store.edge_id(source_id, target_id, relation.relation),
                    relation.anchors,
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
