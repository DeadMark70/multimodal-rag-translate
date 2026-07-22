"""Focused contracts for the provenance-safe Agentic v9 evidence pool."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from data_base.agentic_v9.evidence_pool import EvidencePool
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    SourceLocator,
)


def _packet(
    evidence_id: str,
    *,
    doc_id: str = "doc-1",
    chunk_id: str | None = "chunk-1",
    parent_id: str | None = "parent-1",
    page: int | None = 2,
    asset_id: str | None = None,
    statement: str = "The measured score is 0.91.",
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["slot-1"],
        statement=statement,
        support_type="direct",
        source=EvidenceSource(
            doc_id=doc_id,
            chunk_id=chunk_id,
            parent_id=parent_id,
            asset_id=asset_id,
        ),
        scope=EvidenceScope(metric="Dice"),
        locator=SourceLocator(pdf_page_index=page, section="Results"),
    )


def test_pool_identity_uses_packet_source_and_page_not_aggregate_source_ids() -> None:
    pool = EvidencePool()
    first = _packet("wrong-list", doc_id="document-a", chunk_id="chunk-a", page=4)
    second = _packet("document-b", doc_id="document-b", chunk_id="chunk-a", page=4)

    pool.add(first, metadata={"source_doc_ids": ["document-b", "document-a"]})
    pool.add(second)

    assert pool.retrieved_ids == ("document-b", "wrong-list")
    assert pool.get("wrong-list").identity.doc_id == "document-a"
    assert pool.get("document-b").identity.doc_id == "document-b"


def test_pool_uses_normalized_content_hash_only_when_source_location_is_absent() -> None:
    pool = EvidencePool()
    first = _packet(
        "first",
        chunk_id=None,
        parent_id=None,
        page=None,
        asset_id=None,
        statement="  Score   is  0.91. ",
    )
    second = _packet(
        "second",
        chunk_id=None,
        parent_id=None,
        page=None,
        asset_id=None,
        statement="score is 0.91.",
    )

    pool.add(first)
    pool.add(second)

    assert pool.retrieved_ids == ("first",)
    assert pool.get("first").identity.normalized_hash is not None


def test_pool_preserves_metadata_and_retrieval_scores_for_each_item() -> None:
    pool = EvidencePool()
    packet = _packet("score", asset_id="figure-3")

    entry = pool.add(
        packet,
        metadata={"retriever": "hybrid", "source_doc_ids": ["unrelated", "doc-1"]},
        retrieval_scores={"vector": 0.91, "reranker": 0.83},
    )

    assert entry.metadata == {
        "retriever": "hybrid",
        "source_doc_ids": ["unrelated", "doc-1"],
    }
    assert entry.retrieval_scores == {"vector": 0.91, "reranker": 0.83}
    assert entry.identity.asset_id == "figure-3"
    assert entry.identity.doc_id == "doc-1"


def test_pool_tracks_each_lifecycle_set_without_implicit_answer_state() -> None:
    pool = EvidencePool()
    accepted = _packet("accepted")
    rejected = _packet("rejected", chunk_id="chunk-2")
    pool.add_many([accepted, rejected])

    pool.mark_accepted("accepted")
    pool.mark_packed("accepted")
    pool.mark_used("accepted")
    pool.mark_rejected("rejected", reason="out_of_scope")

    assert pool.retrieved_ids == ("accepted", "rejected")
    assert pool.accepted_ids == ("accepted",)
    assert pool.packed_ids == ("accepted",)
    assert pool.used_ids == ("accepted",)
    assert pool.rejected_ids == ("rejected",)
    assert pool.rejection_reason("rejected") == "out_of_scope"


def test_concurrent_duplicate_adds_are_deterministic_and_idempotent() -> None:
    pool = EvidencePool()
    packet = _packet("same")

    with ThreadPoolExecutor(max_workers=8) as executor:
        entries = list(
            executor.map(
                lambda _: pool.add(
                    packet,
                    metadata={"retriever": "vector"},
                    retrieval_scores={"vector": 0.9},
                ),
                range(32),
            )
        )

    assert pool.retrieved_ids == ("same",)
    assert all(entry is entries[0] for entry in entries)
    assert entries[0].metadata == {"retriever": "vector"}
    assert entries[0].retrieval_scores == {"vector": 0.9}
