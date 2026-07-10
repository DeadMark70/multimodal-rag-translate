from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from graph_rag.generic_mode import link_query_entities
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
