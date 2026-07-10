from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from graph_rag.extractor import (
    EntityRelationExtractor,
    add_extraction_to_graph,
    classify_relation_for_answer_graph,
)
from graph_rag.schemas import (
    EntityType,
    EvidenceAnchor,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    GRAPH_EDGE_TYPES_V1,
    GRAPH_NODE_TYPES_V1,
    RawGraphCandidate,
)
from graph_rag.store import GraphStore


def _workspace_upload_root(test_name: str) -> Path:
    root = Path("output") / "test_tmp" / f"{test_name}_{uuid4().hex}" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _llm_with_payload(payload: dict[str, object]) -> Mock:
    response = Mock(content=json.dumps(payload), usage_metadata={})
    bound_llm = Mock()
    bound_llm.ainvoke = AsyncMock(return_value=response)
    llm = Mock()
    llm.bind = Mock(return_value=bound_llm)
    return llm


def test_schema_v1_exposes_controlled_academic_node_and_edge_types() -> None:
    assert "method" in GRAPH_NODE_TYPES_V1
    assert "claim" in GRAPH_NODE_TYPES_V1
    assert "method_evaluated_on_dataset" in GRAPH_EDGE_TYPES_V1


def test_unknown_relation_is_classified_as_non_answer_graph_candidate() -> None:
    decision = classify_relation_for_answer_graph("loosely related to")

    assert decision.allowed is False
    assert decision.normalized_relation == "unknown_relation"


def test_raw_graph_candidate_can_never_be_used_as_final_evidence() -> None:
    candidate = RawGraphCandidate(
        candidate_id="raw-1",
        candidate_type="unknown_relation",
        payload={"relation": "loosely_related_to"},
        source_doc_id="doc-1",
        source_chunk_index=0,
        confidence=0.5,
        needs_review=True,
    )

    assert candidate.usable_as_final_evidence is False


@pytest.mark.asyncio
async def test_schema_first_extraction_accepts_only_verified_relation_anchors() -> None:
    text = "MedSAM-2 uses memory attention for medical image segmentation."
    extractor = EntityRelationExtractor()
    llm = _llm_with_payload(
        {
            "entities": [
                {
                    "id": "method",
                    "label": "MedSAM-2",
                    "canonical_name": "MedSAM-2",
                    "aliases": ["MedSAM2"],
                    "entity_type": "method",
                    "confidence": 0.9,
                    "evidence_quote": "MedSAM-2 uses memory attention",
                },
                {
                    "id": "component",
                    "label": "memory attention",
                    "canonical_name": "memory attention",
                    "entity_type": "architecture_component",
                    "confidence": 0.8,
                    "evidence_quote": "uses memory attention",
                },
            ],
            "relations": [
                {
                    "source_entity_id": "method",
                    "target_entity_id": "component",
                    "relation": "method_uses_component",
                    "confidence": 0.95,
                    "evidence_quote": "MedSAM-2 uses memory attention",
                }
            ],
        }
    )

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract(text, "doc-1", 4)

    assert [relation.relation for relation in result.relations] == [
        "method_uses_component"
    ]
    assert result.relations[0].anchors[0].doc_id == "doc-1"
    assert result.relations[0].anchors[0].chunk_index == 4
    assert result.relations[0].anchors[0].verification_status == "quote_match"
    assert result.raw_candidates == []


@pytest.mark.asyncio
async def test_schema_first_extraction_buffers_unknown_or_unverified_relations() -> None:
    extractor = EntityRelationExtractor()
    llm = _llm_with_payload(
        {
            "entities": [
                {
                    "id": "method",
                    "label": "MedSAM-2",
                    "entity_type": "method",
                    "confidence": 0.9,
                    "evidence_quote": "MedSAM-2 uses memory attention",
                },
                {
                    "id": "dataset",
                    "label": "Synapse",
                    "entity_type": "dataset",
                    "confidence": 0.8,
                    "evidence_quote": "MedSAM-2 uses memory attention",
                },
            ],
            "relations": [
                {
                    "source_entity_id": "method",
                    "target_entity_id": "dataset",
                    "relation": "loosely_related_to",
                    "confidence": 0.6,
                    "evidence_quote": "MedSAM-2 uses memory attention",
                },
                {
                    "source_entity_id": "method",
                    "target_entity_id": "dataset",
                    "relation": "method_evaluated_on_dataset",
                    "confidence": 0.6,
                    "evidence_quote": "This quote is not in the source chunk",
                },
            ],
        }
    )

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract(
            "MedSAM-2 uses memory attention for medical image segmentation.",
            "doc-1",
            0,
        )

    assert result.relations == []
    assert {candidate.candidate_type for candidate in result.raw_candidates} == {
        "unknown_relation",
        "quote_mismatch",
    }


def test_raw_candidate_buffer_persists_without_becoming_graph_evidence() -> None:
    upload_root = _workspace_upload_root("raw_candidate_buffer")
    candidate = RawGraphCandidate(
        candidate_id="raw-1",
        candidate_type="unknown_relation",
        payload={"relation": "loosely_related_to"},
        source_doc_id="doc-1",
        source_chunk_index=0,
        confidence=0.5,
        needs_review=True,
    )

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("user-1")
        store.record_raw_candidate(candidate)
        store.save_sidecars()

        reloaded = GraphStore("user-1")

    assert reloaded.get_raw_candidates_for_doc("doc-1") == [candidate]
    assert reloaded.get_status().edge_count == 0


@pytest.mark.asyncio
async def test_graph_write_persists_verified_anchor_and_raw_candidate_separately() -> None:
    upload_root = _workspace_upload_root("schema_graph_write")
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="graph:doc-1:chunk:0",
        chunk_index=0,
        quote="MedSAM-2 uses memory attention",
        quote_hash="quote-hash",
        chunk_hash="chunk-hash",
        confidence=0.9,
        verification_status="quote_match",
    )
    result = ExtractionResult(
        doc_id="doc-1",
        chunk_index=0,
        entities=[
            ExtractedEntity(label="MedSAM-2", entity_type=EntityType.METHOD),
            ExtractedEntity(
                label="memory attention",
                entity_type=EntityType.ARCHITECTURE_COMPONENT,
            ),
        ],
        relations=[
            ExtractedRelation(
                entity1="MedSAM-2",
                entity1_type=EntityType.METHOD,
                relation="method_uses_component",
                entity2="memory attention",
                entity2_type=EntityType.ARCHITECTURE_COMPONENT,
                anchors=[anchor],
            )
        ],
        raw_candidates=[
            RawGraphCandidate(
                candidate_id="raw-unknown",
                candidate_type="unknown_relation",
                payload={"relation": "loosely_related_to"},
                source_doc_id="doc-1",
                source_chunk_index=0,
                confidence=0.4,
            )
        ],
    )

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("user-1")
        await add_extraction_to_graph(store, result)

    edge_id = next(iter(store.edge_provenance))
    assert store.get_edge_provenance(edge_id) == [anchor]
    assert store.get_raw_candidates_for_doc("doc-1")[0].candidate_id == "raw-unknown"


@pytest.mark.asyncio
async def test_structured_extraction_failure_is_buffered_not_sent_to_legacy_graph_writer() -> None:
    extractor = EntityRelationExtractor()
    llm = Mock()
    llm.bind.side_effect = RuntimeError("schema service unavailable")

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract("x" * 120, "doc-1", 0)

    assert result.entities == []
    assert result.relations == []
    assert [candidate.candidate_type for candidate in result.raw_candidates] == [
        "structured_extraction_failed"
    ]


@pytest.mark.asyncio
async def test_v1_schema_rejects_legacy_node_and_relation_labels() -> None:
    extractor = EntityRelationExtractor()
    llm = _llm_with_payload(
        {
            "entities": [
                {
                    "id": "method",
                    "label": "MedSAM-2",
                    "entity_type": "method",
                    "confidence": 0.9,
                    "evidence_quote": "MedSAM-2 uses memory attention",
                },
                {
                    "id": "author",
                    "label": "Author A",
                    "entity_type": "author",
                    "confidence": 0.9,
                    "evidence_quote": "MedSAM-2 uses memory attention",
                },
            ],
            "relations": [
                {
                    "source_entity_id": "method",
                    "target_entity_id": "method",
                    "relation": "uses",
                    "confidence": 0.9,
                    "evidence_quote": "MedSAM-2 uses memory attention",
                }
            ],
        }
    )

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract(
            "MedSAM-2 uses memory attention for medical image segmentation.",
            "doc-1",
            0,
        )

    assert result.relations == []
    assert {candidate.candidate_type for candidate in result.raw_candidates} == {
        "unknown_node_type",
        "unknown_relation",
    }
