from unittest.mock import AsyncMock, Mock, patch

import pytest

from graph_rag.extractor import EntityRelationExtractor
from graph_rag.schemas import EntityType, ExtractedEntity, ExtractedRelation
from pdfserviceMD.router import _run_graph_extraction


def _make_llm_with_structured_payload(payload: dict) -> tuple[Mock, Mock]:
    structured_llm = Mock()
    structured_llm.ainvoke = AsyncMock(return_value=payload)
    llm = Mock()
    llm.with_structured_output = Mock(return_value=structured_llm)
    return llm, structured_llm


@pytest.mark.asyncio
async def test_structured_one_pass_returns_entities_and_relations() -> None:
    extractor = EntityRelationExtractor()
    llm, structured_llm = _make_llm_with_structured_payload(
        {
            "entities": [
                {
                    "id": "e1",
                    "label": "Transformer",
                    "entity_type": "method",
                    "description": "A neural architecture",
                },
                {
                    "id": "e2",
                    "label": "Attention",
                    "entity_type": "concept",
                    "description": "A mechanism",
                },
            ],
            "relations": [
                {
                    "source_entity_id": "e1",
                    "target_entity_id": "e2",
                    "relation": "uses",
                    "description": "Transformer uses attention",
                }
            ],
        }
    )

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract("x" * 120, "doc-1", 3)

    llm.with_structured_output.assert_called_once()
    structured_llm.ainvoke.assert_awaited_once()
    assert result.doc_id == "doc-1"
    assert result.chunk_index == 3
    assert [entity.label for entity in result.entities] == ["Transformer", "Attention"]
    assert result.relations[0].entity1 == "Transformer"
    assert result.relations[0].entity1_type == EntityType.METHOD
    assert result.relations[0].entity2 == "Attention"
    assert result.relations[0].entity2_type == EntityType.CONCEPT


@pytest.mark.asyncio
async def test_structured_one_pass_drops_relations_with_unknown_entity_ids() -> None:
    extractor = EntityRelationExtractor()
    llm, _ = _make_llm_with_structured_payload(
        {
            "entities": [
                {
                    "id": "e1",
                    "label": "BERT",
                    "entity_type": "method",
                }
            ],
            "relations": [
                {
                    "source_entity_id": "e1",
                    "target_entity_id": "missing",
                    "relation": "uses",
                }
            ],
        }
    )

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract("x" * 120, "doc-1")

    assert [entity.label for entity in result.entities] == ["BERT"]
    assert result.relations == []


@pytest.mark.asyncio
async def test_structured_one_pass_deduplicates_labels_and_keeps_valid_relations() -> None:
    extractor = EntityRelationExtractor()
    llm, _ = _make_llm_with_structured_payload(
        {
            "entities": [
                {
                    "id": "e1",
                    "label": "Transformer",
                    "entity_type": "method",
                },
                {
                    "id": "e2",
                    "label": "Transformer",
                    "entity_type": "method",
                },
                {
                    "id": "e3",
                    "label": "Machine Translation",
                    "entity_type": "concept",
                },
            ],
            "relations": [
                {
                    "source_entity_id": "e2",
                    "target_entity_id": "e3",
                    "relation": "applies_to",
                    "description": "duplicate entity id should alias to the kept entity",
                }
            ],
        }
    )

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract("x" * 120, "doc-1")

    assert [entity.label for entity in result.entities] == [
        "Transformer",
        "Machine Translation",
    ]
    assert len(result.relations) == 1
    assert result.relations[0].entity1 == "Transformer"
    assert result.relations[0].entity2 == "Machine Translation"


@pytest.mark.asyncio
async def test_structured_one_pass_falls_back_to_legacy_two_pass() -> None:
    extractor = EntityRelationExtractor()
    llm = Mock()
    llm.with_structured_output = Mock(side_effect=RuntimeError("schema init failed"))
    legacy_entities = [
        ExtractedEntity(
            label="LeNet",
            entity_type=EntityType.METHOD,
            description="Legacy entity",
        )
    ]
    legacy_relations = [
        ExtractedRelation(
            entity1="LeNet",
            entity1_type=EntityType.METHOD,
            relation="uses",
            entity2="CNN",
            entity2_type=EntityType.CONCEPT,
            description="Legacy relation",
        )
    ]

    with (
        patch("graph_rag.extractor.get_llm", return_value=llm),
        patch.object(
            extractor,
            "extract_entities",
            new=AsyncMock(return_value=legacy_entities),
        ) as mock_entities,
        patch.object(
            extractor,
            "extract_relations",
            new=AsyncMock(return_value=legacy_relations),
        ) as mock_relations,
    ):
        result = await extractor.extract("x" * 120, "doc-1")

    mock_entities.assert_awaited_once()
    mock_relations.assert_awaited_once_with("x" * 120, legacy_entities)
    assert result.entities == legacy_entities
    assert result.relations == legacy_relations


@pytest.mark.asyncio
async def test_structured_one_pass_enforces_max_entities_per_chunk() -> None:
    extractor = EntityRelationExtractor(max_entities_per_chunk=2)
    llm, _ = _make_llm_with_structured_payload(
        {
            "entities": [
                {"id": "e1", "label": "Entity 1", "entity_type": "concept"},
                {"id": "e2", "label": "Entity 2", "entity_type": "concept"},
                {"id": "e3", "label": "Entity 3", "entity_type": "concept"},
            ],
            "relations": [
                {
                    "source_entity_id": "e3",
                    "target_entity_id": "e1",
                    "relation": "cites",
                }
            ],
        }
    )

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract("x" * 120, "doc-1")

    assert [entity.label for entity in result.entities] == ["Entity 1", "Entity 2"]
    assert result.relations == []


@pytest.mark.asyncio
async def test_run_graph_extraction_invokes_one_extraction_call_per_valid_chunk() -> None:
    markdown_text = "A" * 16050
    mock_store = Mock()

    with (
        patch("pdfserviceMD.router.GraphStore", return_value=mock_store),
        patch(
            "pdfserviceMD.router.extract_and_add_to_graph",
            new=AsyncMock(side_effect=[(1, 0), (2, 1)]),
        ) as mock_extract,
    ):
        await _run_graph_extraction(
            user_id="user-1",
            doc_id="doc-1",
            markdown_text=markdown_text,
            batch_size=2,
        )

    assert mock_extract.await_count == 2
    assert [call.kwargs["chunk_index"] for call in mock_extract.await_args_list] == [0, 1]
    mock_store.save.assert_called_once()
