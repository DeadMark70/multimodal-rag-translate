from typing import get_args

import pytest
from langchain_core.documents import Document

from graph_rag.anchor_resolver import ChunkAnchorResolver
from graph_rag.feature_flags import get_graph_feature_flags
from graph_rag.schemas import EvidenceAnchor, GraphEvidenceMode


@pytest.fixture
def fake_lookup():
    class FakeLookup:
        def by_chunk_id(self, user_id: str, chunk_id: str) -> Document | None:
            if user_id == "user-1" and chunk_id == "chunk-1":
                return Document(
                    page_content="Current chunk text",
                    metadata={"doc_id": "doc-1", "chunk_id": "chunk-1", "chunk_hash": "new-hash"},
                )
            if user_id == "user-1" and chunk_id == "chunk-wrong-doc":
                return Document(
                    page_content="Wrong document chunk text",
                    metadata={"doc_id": "doc-2", "chunk_id": "chunk-wrong-doc", "chunk_hash": "wrong-doc-hash"},
                )
            return None

        def by_doc_and_index(self, user_id: str, doc_id: str, chunk_index: int) -> Document | None:
            if user_id == "user-1" and doc_id == "doc-1" and chunk_index == 3:
                return Document(
                    page_content="Indexed chunk text",
                    metadata={"doc_id": "doc-1", "chunk_index": 3, "chunk_hash": "indexed-hash"},
                )
            return None

        def by_chunk_hash(self, user_id: str, doc_id: str, chunk_hash: str) -> Document | None:
            if user_id == "user-1" and doc_id == "doc-1" and chunk_hash == "indexed-hash":
                return Document(
                    page_content="Indexed chunk text",
                    metadata={"doc_id": "doc-1", "chunk_index": 3, "chunk_hash": "indexed-hash"},
                )
            return None

        def fuzzy_by_quote(self, user_id: str, doc_id: str, quote: str) -> Document | None:
            if (
                user_id == "user-1"
                and doc_id == "doc-1"
                and quote == "MedSAM-2 uses memory attention."
            ):
                return Document(
                    page_content="MedSAM-2 uses memory attention in the decoder.",
                    metadata={"doc_id": "doc-1", "chunk_id": "chunk-fuzzy", "chunk_hash": "fuzzy-hash"},
                )
            if user_id == "user-1" and doc_id == "doc-1" and quote == "Quote from another document.":
                return Document(
                    page_content="Quote from another document.",
                    metadata={"doc_id": "doc-2", "chunk_id": "chunk-other-doc", "chunk_hash": "other-doc-hash"},
                )
            return None

    return FakeLookup()


def test_graph_evidence_mode_contract_contains_rollout_modes() -> None:
    modes = get_args(GraphEvidenceMode)

    assert "raw_current" in modes
    assert "provenance_gated" in modes
    assert "locator_to_chunk" in modes
    assert "router_auto" in modes
    assert "locator_only" in modes


def test_evidence_anchor_minimum_text_contract() -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=3,
        page=7,
        quote="MedSAM-2 uses memory attention.",
        quote_hash="qhash",
        chunk_hash="chash",
        source_text_hash="sourcehash",
        confidence=0.91,
        extraction_model="gemini-test",
        extraction_prompt_version="graph-extract-v2",
    )

    assert anchor.doc_id == "doc-1"
    assert anchor.anchor_type == "text"
    assert anchor.provenance_status == "full"


def test_evidence_anchor_missing_quote_is_partial() -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=3,
        confidence=0.72,
    )

    assert anchor.provenance_status == "partial"


def test_graph_feature_flags_default_to_legacy_safe_path() -> None:
    flags = get_graph_feature_flags({})

    assert flags.graph_raw_current_enabled is True
    assert flags.graph_evidence_locator_enabled is False
    assert flags.graph_provenance_gate_enabled is False
    assert flags.graph_to_chunk_enabled is False


def test_evidence_anchor_serializes_computed_provenance_status() -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        quote="MedSAM-2 uses memory attention.",
        quote_hash="qhash",
        chunk_hash="chash",
        confidence=0.91,
    )

    payload = anchor.model_dump(mode="json")

    assert payload["provenance_status"] == "full"


def test_graph_feature_flags_are_serializable_for_run_snapshot() -> None:
    flags = get_graph_feature_flags({"graph_to_chunk_enabled": "true"})

    assert flags.to_snapshot()["graph_to_chunk_enabled"] is True


def test_anchor_resolver_detects_hash_mismatch(fake_lookup) -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_hash="old-hash",
        confidence=0.8,
    )

    result = ChunkAnchorResolver(fake_lookup).resolve("user-1", anchor)

    assert result.resolution_status == "stale"
    assert result.verification_status == "hash_mismatch"


def test_anchor_resolver_fuzzy_quote_match(fake_lookup) -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        quote="MedSAM-2 uses memory attention.",
        quote_hash="qhash",
        confidence=0.8,
    )

    result = ChunkAnchorResolver(fake_lookup).resolve("user-1", anchor)

    assert result.resolution_status == "fuzzy_resolved"
    assert result.verification_status == "quote_match"


def test_anchor_resolver_rejects_chunk_id_from_wrong_document(fake_lookup) -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-wrong-doc",
        confidence=0.8,
    )

    result = ChunkAnchorResolver(fake_lookup).resolve("user-1", anchor)

    assert result.resolution_status == "unresolved"
    assert result.verification_status == "not_checked"
    assert result.reason == "doc_id_mismatch"


def test_anchor_resolver_rejects_fuzzy_quote_from_wrong_document(fake_lookup) -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        quote="Quote from another document.",
        confidence=0.8,
    )

    result = ChunkAnchorResolver(fake_lookup).resolve("user-1", anchor)

    assert result.resolution_status == "unresolved"
    assert result.verification_status == "not_checked"
    assert result.reason == "doc_id_mismatch"
