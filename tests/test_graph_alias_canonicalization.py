from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from graph_rag.generic_mode import link_query_entities
from graph_rag.local_search import local_search
from graph_rag.extractor import add_extraction_to_graph
from graph_rag.schemas import ClaimIdentity, EntityType, ExtractedEntity, ExtractionResult
from graph_rag.store import GraphStore


def _workspace_upload_root(test_name: str) -> Path:
    root = Path("output") / "test_tmp" / f"{test_name}_{uuid4().hex}" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_method_alias_resolves_to_persisted_canonical_node() -> None:
    upload_root = _workspace_upload_root("canonical_alias")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("user-1")
        canonical_id = store.upsert_canonical_entity(
            canonical_name="MedSAM-2",
            entity_type="method",
            aliases=["MedSAM2", "MedSAM 2"],
            source_doc_ids=["doc-1"],
        )
        store.save_sidecars()

        reloaded = GraphStore("user-1")

    assert reloaded.find_canonical_node("MedSAM2", "method") == canonical_id
    assert reloaded.find_canonical_node("medsam 2", None) == canonical_id


def test_claim_identity_keeps_scope_and_source_in_stable_key() -> None:
    scoped = ClaimIdentity(
        claim_type="first_claim",
        subject="Weak-Mamba-UNet",
        scope="scribble-based weakly supervised segmentation",
        condition="scribble supervision",
        source_doc="weak-mamba.pdf",
    )
    different_scope = scoped.model_copy(update={"scope": "fully supervised segmentation"})
    different_source = scoped.model_copy(update={"source_doc": "other-paper.pdf"})

    assert scoped.stable_key != different_scope.stable_key
    assert scoped.stable_key != different_source.stable_key


def test_claim_identity_is_persisted_as_the_explicit_claim_identity_key() -> None:
    upload_root = _workspace_upload_root("claim_identity_key")
    claim = ClaimIdentity(
        claim_type="first_claim",
        subject="Weak-Mamba-UNet",
        scope="scribble-based weakly supervised segmentation",
        source_doc="weak-mamba.pdf",
    )

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("user-1")
        claim_id = store.upsert_canonical_entity(
            canonical_name="first claim",
            entity_type="claim",
            aliases=[],
            source_doc_ids=["doc-1"],
            claim_identity=claim,
        )

    assert store.canonical_entities[claim_id].identity_key == claim.stable_key


def test_review_required_model_alias_does_not_cross_merge_document_scope() -> None:
    upload_root = _workspace_upload_root("review_required_alias")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("user-1")
        first_id = store.upsert_canonical_entity(
            canonical_name="U-Mamba",
            entity_type="model",
            aliases=["U Mamba"],
            source_doc_ids=["doc-1"],
        )
        second_id = store.upsert_canonical_entity(
            canonical_name="U-Mamba_Enc",
            entity_type="model",
            aliases=["U Mamba"],
            source_doc_ids=["doc-2"],
        )

    assert first_id != second_id
    assert store.find_canonical_node("U Mamba", "model") == first_id


def test_review_required_and_never_merge_types_do_not_reuse_cross_document_names() -> None:
    upload_root = _workspace_upload_root("canonical_scope_boundaries")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("user-1")
        model_first = store.upsert_canonical_entity(
            canonical_name="U-Mamba",
            entity_type="model",
            aliases=[],
            source_doc_ids=["doc-1"],
        )
        model_second = store.upsert_canonical_entity(
            canonical_name="U-Mamba",
            entity_type="model",
            aliases=[],
            source_doc_ids=["doc-2"],
        )
        claim_first = store.upsert_canonical_entity(
            canonical_name="first claim",
            entity_type="claim",
            aliases=[],
            source_doc_ids=["doc-1"],
        )
        claim_second = store.upsert_canonical_entity(
            canonical_name="first claim",
            entity_type="claim",
            aliases=[],
            source_doc_ids=["doc-1"],
        )

    assert model_first != model_second
    assert claim_first != claim_second


def test_query_linking_prefers_alias_index_before_search_fallback() -> None:
    upload_root = _workspace_upload_root("query_alias_linking")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("user-1")
        canonical_id = store.upsert_canonical_entity(
            canonical_name="MedSAM-2",
            entity_type="method",
            aliases=["MedSAM2"],
            source_doc_ids=["doc-1"],
        )

    assert link_query_entities(store, ["MedSAM2", "missing"]) == [canonical_id]


@pytest.mark.asyncio
async def test_schema_first_entity_write_registers_canonical_aliases() -> None:
    upload_root = _workspace_upload_root("schema_alias_write")
    result = ExtractionResult(
        doc_id="doc-1",
        chunk_index=0,
        entities=[
            ExtractedEntity(
                label="MedSAM-2",
                canonical_name="MedSAM-2",
                aliases=["MedSAM2"],
                entity_type=EntityType.METHOD,
            )
        ],
    )

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("user-1")
        await add_extraction_to_graph(store, result)

    assert store.find_canonical_node("MedSAM2", "method") is not None


@pytest.mark.asyncio
async def test_local_search_uses_alias_seed_before_vector_and_fuzzy_fallback() -> None:
    upload_root = _workspace_upload_root("local_search_alias_seed")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("user-1")
        canonical_id = store.upsert_canonical_entity(
            canonical_name="MedSAM-2",
            entity_type="method",
            aliases=["MedSAM2"],
            source_doc_ids=["doc-1"],
        )
        with (
            patch("graph_rag.local_search.node_vector_search_enabled", return_value=False),
            patch(
                "graph_rag.local_search.identify_query_entities",
                new=AsyncMock(return_value=["MedSAM2"]),
            ),
            patch("graph_rag.local_search.find_matching_nodes") as fuzzy_match,
        ):
            _, matched = await local_search(store, "How does MedSAM2 work?")

    assert matched == [canonical_id]
    fuzzy_match.assert_not_called()
