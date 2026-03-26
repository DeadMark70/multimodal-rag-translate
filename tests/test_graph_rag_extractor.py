import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from graph_rag.extractor import (
    EntityRelationExtractor,
    _StructuredExtractionPayload,
    _parse_json_from_response,
)
from graph_rag.schemas import EntityType, ExtractedEntity, ExtractedRelation
from graph_rag.service import run_graph_extraction


def _make_llm_with_structured_payload(
    payload: dict | str,
    *,
    content_as_list: bool = False,
) -> tuple[Mock, Mock]:
    bound_llm = Mock()
    content = payload if isinstance(payload, str) else json.dumps(payload)
    response = Mock(
        content=[{"type": "text", "text": content}] if content_as_list else content,
        usage_metadata={},
    )
    bound_llm.ainvoke = AsyncMock(return_value=response)
    llm = Mock()
    llm.bind = Mock(return_value=bound_llm)
    return llm, bound_llm


@pytest.mark.asyncio
async def test_structured_one_pass_returns_entities_and_relations() -> None:
    extractor = EntityRelationExtractor()
    llm, bound_llm = _make_llm_with_structured_payload(
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

    llm.bind.assert_called_once()
    bound_llm.ainvoke.assert_awaited_once()
    assert result.doc_id == "doc-1"
    assert result.chunk_index == 3
    assert [entity.label for entity in result.entities] == ["Transformer", "Attention"]
    assert result.relations[0].entity1 == "Transformer"
    assert result.relations[0].entity1_type == EntityType.METHOD
    assert result.relations[0].entity2 == "Attention"
    assert result.relations[0].entity2_type == EntityType.CONCEPT


@pytest.mark.asyncio
async def test_structured_one_pass_binds_native_json_schema() -> None:
    extractor = EntityRelationExtractor()
    bound_llm = Mock()
    bound_llm.ainvoke = AsyncMock(
        return_value=Mock(
            content='{"entities":[{"id":"e1","label":"Transformer","entity_type":"method"}],"relations":[]}',
            usage_metadata={},
        )
    )
    llm = Mock()
    llm.bind = Mock(return_value=bound_llm)

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract("x" * 120, "doc-1")

    llm.bind.assert_called_once()
    kwargs = llm.bind.call_args.kwargs
    assert kwargs["response_mime_type"] == "application/json"
    assert kwargs["response_json_schema"] == _StructuredExtractionPayload.model_json_schema()
    assert [entity.label for entity in result.entities] == ["Transformer"]
    bound_llm.ainvoke.assert_awaited_once()


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
    llm.bind = Mock(side_effect=RuntimeError("schema init failed"))
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
async def test_legacy_entity_parsing_handles_content_blocks_and_wrapped_dict() -> None:
    extractor = EntityRelationExtractor()
    llm = Mock()
    llm.bind = Mock(side_effect=RuntimeError("force fallback"))
    llm.ainvoke = AsyncMock(
        return_value=Mock(
            content=[
                {
                    "text": """```json
{"entities":[{"label":"BERT","entity_type":"method","description":"encoder"}]}
```"""
                }
            ]
        )
    )

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        entities = await extractor.extract_entities("x" * 120)

    assert len(entities) == 1
    assert entities[0].label == "BERT"
    assert entities[0].entity_type == EntityType.METHOD


@pytest.mark.asyncio
async def test_structured_one_pass_accepts_json_from_list_content_blocks() -> None:
    extractor = EntityRelationExtractor()
    llm, _ = _make_llm_with_structured_payload(
        {
            "entities": [{"id": "e1", "label": "raw only", "entity_type": "concept"}],
            "relations": [],
        },
        content_as_list=True,
    )

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract("x" * 120, "doc-1")

    assert [entity.label for entity in result.entities] == ["raw only"]
    assert result.relations == []


@pytest.mark.asyncio
async def test_structured_one_pass_falls_back_on_invalid_json_response() -> None:
    extractor = EntityRelationExtractor()
    llm, _ = _make_llm_with_structured_payload('{"entities": [', content_as_list=True)
    legacy_entities = [
        ExtractedEntity(label="Fallback Entity", entity_type=EntityType.CONCEPT, description=None)
    ]

    with (
        patch("graph_rag.extractor.get_llm", return_value=llm),
        patch.object(extractor, "extract_entities", new=AsyncMock(return_value=legacy_entities)) as mock_entities,
        patch.object(extractor, "extract_relations", new=AsyncMock(return_value=[])) as mock_relations,
    ):
        result = await extractor.extract("x" * 120, "doc-1")

    mock_entities.assert_awaited_once()
    mock_relations.assert_awaited_once_with("x" * 120, legacy_entities)
    assert [entity.label for entity in result.entities] == ["Fallback Entity"]


@pytest.mark.asyncio
async def test_structured_one_pass_recovers_via_include_raw_when_native_schema_bind_is_unsupported() -> None:
    extractor = EntityRelationExtractor()
    structured_llm = Mock()
    structured_llm.ainvoke = AsyncMock(
        return_value={
            "parsed": None,
            "raw": Mock(
                content='{"entities":[{"id":"e1","label":"Recovered","entity_type":"method"}],"relations":[]}',
                usage_metadata={},
            ),
            "parsing_error": TypeError("schema parse failed"),
        }
    )
    llm = Mock()
    llm.bind = Mock(
        side_effect=TypeError(
            "GenerativeServiceAsyncClient.generate_content() got an unexpected keyword argument 'response_mime_type'"
        )
    )
    llm.with_structured_output = Mock(return_value=structured_llm)

    with patch("graph_rag.extractor.get_llm", return_value=llm):
        result = await extractor.extract("x" * 120, "doc-1")

    llm.bind.assert_called_once()
    llm.with_structured_output.assert_called_once()
    structured_llm.ainvoke.assert_awaited_once()
    assert [entity.label for entity in result.entities] == ["Recovered"]


def test_parse_json_from_response_ignores_trailing_text_after_array() -> None:
    payload = '[{"label":"BERT","entity_type":"method"}]\n補充說明不要解析'
    result = _parse_json_from_response(payload)

    assert result == [{"label": "BERT", "entity_type": "method"}]


def test_parse_json_from_response_ignores_trailing_text_after_wrapped_object() -> None:
    payload = '{"entities":[{"label":"BLEU","entity_type":"metric"}]}\n\n額外說明'
    result = _parse_json_from_response(payload, list_keys=("entities",))

    assert result == [{"label": "BLEU", "entity_type": "metric"}]


@pytest.mark.asyncio
async def test_run_graph_extraction_invokes_one_extraction_call_per_valid_chunk() -> None:
    markdown_text = "A" * 16050
    mock_store = Mock()

    with (
        patch("graph_rag.service.GraphStore", return_value=mock_store),
        patch(
            "graph_rag.service.extract_and_add_to_graph",
            new=AsyncMock(side_effect=[(1, 0), (2, 1)]),
        ) as mock_extract,
    ):
        await run_graph_extraction(
            user_id="user-1",
            doc_id="doc-1",
            markdown_text=markdown_text,
            batch_size=2,
        )

    assert mock_extract.await_count == 2
    assert [call.kwargs["chunk_index"] for call in mock_extract.await_args_list] == [0, 1]
    mock_store.save.assert_called_once()


@pytest.mark.asyncio
async def test_run_graph_extraction_marks_empty_when_no_valid_chunks() -> None:
    mock_store = Mock()

    with patch("graph_rag.service.GraphStore", return_value=mock_store):
        result = await run_graph_extraction(
            user_id="user-1",
            doc_id="doc-empty",
            markdown_text="too short",
        )

    assert result.status == "empty"
    mock_store.upsert_document_status.assert_called_once()
    saved_status = mock_store.upsert_document_status.call_args.args[0]
    assert saved_status.status == "empty"


@pytest.mark.asyncio
async def test_run_graph_extraction_marks_partial_when_some_chunks_fail() -> None:
    markdown_text = "A" * 16050
    mock_store = Mock()

    with (
        patch("graph_rag.service.GraphStore", return_value=mock_store),
        patch(
            "graph_rag.service.extract_and_add_to_graph",
            new=AsyncMock(side_effect=[(1, 0), RuntimeError("quota exceeded")]),
        ),
    ):
        result = await run_graph_extraction(
            user_id="user-1",
            doc_id="doc-partial",
            markdown_text=markdown_text,
            batch_size=2,
        )

    assert result.status == "partial"
    saved_status = mock_store.upsert_document_status.call_args.args[0]
    assert saved_status.status == "partial"
    assert "quota exceeded" in (saved_status.last_error or "")
