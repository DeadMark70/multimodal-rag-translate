import pytest
from pydantic import ValidationError

from graph_rag.generic_mode import merge_graph_evidence_bundle
from graph_rag.schemas import EvidenceAnchor, GraphEvidenceBundle, GraphEvidenceItem, GraphHint


def test_community_summary_is_hint_not_final_evidence() -> None:
    hint = GraphHint(
        hint_id="community:1",
        hint_type="community_summary",
        text="This community discusses SAM variants.",
        confidence=0.7,
        source_ids=["community:1"],
    )

    bundle = merge_graph_evidence_bundle(hints=[hint], evidence_items=[], token_budget=800)

    assert bundle.hints[0].usable_as_final_evidence is False
    assert bundle.final_context_items == []


def test_edge_without_provenance_is_excluded_from_final_context() -> None:
    item = GraphEvidenceItem(
        item_id="edge:1",
        graph_mode="local",
        source="edge",
        node_ids=["a", "b"],
        edge_ids=["edge:1"],
        source_chunk_ids=[],
        source_doc_ids=["doc-1"],
        pages=[],
        relation_type="extends",
        evidence_quote=None,
        summary="A extends B",
        confidence=0.8,
        provenance_status="missing",
        usable_as_context=False,
        use_reason="missing provenance",
    )

    bundle = merge_graph_evidence_bundle(hints=[], evidence_items=[item], token_budget=800)

    assert bundle.final_context_items == []


def test_full_provenance_edge_can_enter_locator_bundle() -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=1,
        quote="A extends B.",
        quote_hash="q",
        chunk_hash="c",
        confidence=0.9,
    )
    item = GraphEvidenceItem.from_anchor(
        item_id="edge:1",
        graph_mode="local",
        source="edge",
        edge_ids=["edge:1"],
        node_ids=["a", "b"],
        relation_type="extends",
        summary="A extends B",
        anchor=anchor,
        resolution_status="resolved",
    )

    bundle = merge_graph_evidence_bundle(hints=[], evidence_items=[item], token_budget=800)

    assert bundle.final_context_items[0].source_chunk_ids == ["chunk-1"]


def test_unresolved_anchor_is_not_final_context() -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=1,
        quote="A extends B.",
        quote_hash="q",
        chunk_hash="c",
        confidence=0.9,
    )
    item = GraphEvidenceItem.from_anchor(
        item_id="edge:1",
        graph_mode="local",
        source="edge",
        edge_ids=["edge:1"],
        node_ids=["a", "b"],
        relation_type="extends",
        summary="A extends B",
        anchor=anchor,
    )

    bundle = merge_graph_evidence_bundle(hints=[], evidence_items=[item], token_budget=800)

    assert item.resolution_status == "unresolved"
    assert item.usable_as_context is False
    assert bundle.final_context_items == []


def test_community_hint_never_becomes_final_context_even_with_high_confidence() -> None:
    hint = GraphHint(
        hint_id="community:important",
        hint_type="community_summary",
        text="A highly relevant community summary.",
        confidence=0.99,
        source_ids=["community:important"],
    )

    bundle = merge_graph_evidence_bundle(hints=[hint], evidence_items=[], token_budget=800)

    assert bundle.final_context_items == []
    assert bundle.hints[0].usable_as_final_evidence is False


def _resolved_item(
    *,
    item_id: str = "edge:resolved",
    confidence: float = 0.9,
    quote: str = "Verified source quote.",
) -> GraphEvidenceItem:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id=f"chunk-{item_id}",
        quote=quote,
        quote_hash="quote-hash",
        chunk_hash="chunk-hash",
        confidence=confidence,
    )
    return GraphEvidenceItem.from_anchor(
        item_id=item_id,
        graph_mode="local",
        source="edge",
        edge_ids=[item_id],
        node_ids=["a", "b"],
        relation_type="supports",
        summary="Inferred graph summary that must not control eligibility.",
        anchor=anchor,
        resolution_status="resolved",
    )


def test_graph_hint_cannot_be_constructed_as_final_evidence() -> None:
    with pytest.raises(ValidationError):
        GraphHint(
            hint_id="community:1",
            hint_type="community_summary",
            text="Community output.",
            confidence=0.8,
            usable_as_final_evidence=True,
        )


def test_bundle_rejects_ineligible_final_context_item() -> None:
    item = _resolved_item()
    item.usable_as_context = False

    with pytest.raises(ValidationError):
        GraphEvidenceBundle(
            query="q",
            route="local-first",
            final_context_items=[item],
        )


@pytest.mark.parametrize(
    ("resolution_status", "verification_status"),
    [
        ("unresolved", "not_checked"),
        ("stale", "hash_mismatch"),
        ("resolved", "quote_mismatch"),
    ],
)
def test_merge_rejects_statuses_inconsistent_with_final_context(
    resolution_status: str,
    verification_status: str,
) -> None:
    item = _resolved_item()
    item.resolution_status = resolution_status
    item.verification_status = verification_status
    item.usable_as_context = True

    bundle = merge_graph_evidence_bundle(
        hints=[],
        evidence_items=[item],
        token_budget=800,
    )

    assert bundle.final_context_items == []


def test_merge_is_deterministic_deduplicated_and_never_exceeds_first_item_budget() -> None:
    first = _resolved_item(item_id="edge:b", confidence=0.8, quote="B" * 40)
    duplicate = _resolved_item(item_id="edge:b", confidence=0.8, quote="B" * 40)
    second = _resolved_item(item_id="edge:a", confidence=0.8, quote="A" * 4)

    within_budget = merge_graph_evidence_bundle(
        hints=[],
        evidence_items=[first, duplicate, second],
        token_budget=10,
    )
    no_budget = merge_graph_evidence_bundle(
        hints=[],
        evidence_items=[first],
        token_budget=9,
    )

    assert [item.item_id for item in within_budget.final_context_items] == ["edge:a"]
    assert no_budget.final_context_items == []
