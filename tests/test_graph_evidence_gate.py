from graph_rag.generic_mode import merge_graph_evidence_bundle
from graph_rag.schemas import EvidenceAnchor, GraphEvidenceItem, GraphHint


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
    )

    bundle = merge_graph_evidence_bundle(hints=[], evidence_items=[item], token_budget=800)

    assert bundle.final_context_items[0].source_chunk_ids == ["chunk-1"]


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
