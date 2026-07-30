"""Deterministic final-evidence balancing for comparison subjects."""

from __future__ import annotations

from data_base.agentic_v9.comparison_context import (
    comparison_final_limit,
    select_balanced_comparison_packets,
)
from data_base.agentic_v9.schemas import (
    ComparisonPlan,
    ComparisonSubject,
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    SourceLocator,
)


def _subject(subject_id: str, display_name: str) -> ComparisonSubject:
    return ComparisonSubject(
        subject_id=subject_id,
        display_name=display_name,
        retrieval_query=f"{display_name} evidence",
    )


def _plan(*subjects: ComparisonSubject) -> ComparisonPlan:
    return ComparisonPlan(subjects=list(subjects), dimensions=["latency"])


def _packet(
    evidence_id: str,
    subject_id: str,
    *,
    doc_id: str | None = None,
    chunk_id: str | None = None,
    task_id: str | None = None,
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id=task_id or f"task:{subject_id}",
        round_id="round-1",
        query_id="query-1",
        slot_ids=[f"comparison-subject:{subject_id}"],
        statement=f"Evidence for {subject_id}: {evidence_id}",
        support_type="direct",
        source=EvidenceSource(
            doc_id=doc_id or f"doc:{subject_id}",
            chunk_id=chunk_id or evidence_id,
        ),
        scope=EvidenceScope(metric="latency"),
        locator=SourceLocator(pdf_page_index=0),
    )


def test_comparison_final_limit_is_four_or_six() -> None:
    assert comparison_final_limit(2) == 4
    assert comparison_final_limit(3) == 6
    assert comparison_final_limit(4) == 6


def test_two_subjects_keep_at_most_two_packets_each() -> None:
    plan = _plan(_subject("a", "Model A"), _subject("b", "Model B"))
    packets = [
        _packet("a-low", "a"),
        _packet("a-high", "a"),
        _packet("a-mid", "a"),
        _packet("b-high", "b"),
        _packet("b-low", "b"),
        _packet("b-third", "b"),
    ]

    selected = select_balanced_comparison_packets(
        packets,
        plan=plan,
        quality_by_evidence_id={
            "a-low": 0.1,
            "a-high": 0.9,
            "a-mid": 0.5,
            "b-high": 0.8,
            "b-low": 0.2,
            "b-third": 0.0,
        },
    )

    assert [packet.evidence_id for packet in selected] == [
        "a-high",
        "a-mid",
        "b-high",
        "b-low",
    ]


def test_three_subjects_reserve_one_each_then_fill_by_quality() -> None:
    plan = _plan(
        _subject("a", "Model A"),
        _subject("b", "Model B"),
        _subject("c", "Model C"),
    )
    packets = [
        _packet("a1", "a"),
        _packet("a2", "a"),
        _packet("b1", "b"),
        _packet("b2", "b"),
        _packet("c1", "c"),
        _packet("c2", "c"),
        _packet("c3", "c"),
    ]

    selected = select_balanced_comparison_packets(
        packets,
        plan=plan,
        quality_by_evidence_id={
            "a1": 0.9,
            "a2": 0.8,
            "b1": 0.7,
            "b2": 0.1,
            "c1": -1.0,
            "c2": -2.0,
            "c3": 1.0,
        },
    )

    assert len(selected) == 6
    assert {packet.slot_ids[0] for packet in selected} == {
        "comparison-subject:a",
        "comparison-subject:b",
        "comparison-subject:c",
    }
    assert "c3" in [packet.evidence_id for packet in selected]
    assert "c2" not in [packet.evidence_id for packet in selected]


def test_four_subjects_preserve_low_scoring_only_evidence() -> None:
    plan = _plan(
        _subject("a", "Model A"),
        _subject("b", "Model B"),
        _subject("c", "Model C"),
        _subject("d", "Model D"),
    )
    packets = [
        _packet("a1", "a"),
        _packet("a2", "a"),
        _packet("b1", "b"),
        _packet("b2", "b"),
        _packet("c1", "c"),
        _packet("d-only", "d"),
    ]

    selected = select_balanced_comparison_packets(
        packets,
        plan=plan,
        quality_by_evidence_id={
            "a1": 1.0,
            "a2": 0.9,
            "b1": 0.8,
            "b2": 0.7,
            "c1": 0.6,
            "d-only": -100.0,
        },
    )

    assert len(selected) == 6
    assert "d-only" in [packet.evidence_id for packet in selected]


def test_duplicate_source_identity_keeps_the_better_packet_once() -> None:
    plan = _plan(_subject("a", "Model A"), _subject("b", "Model B"))
    duplicate_low = _packet(
        "duplicate-low", "a", doc_id="doc-a", chunk_id="same-chunk"
    )
    duplicate_high = _packet(
        "duplicate-high", "a", doc_id="doc-a", chunk_id="same-chunk"
    )
    packets = [
        duplicate_low,
        duplicate_high,
        _packet("a2", "a"),
        _packet("b1", "b"),
    ]

    selected = select_balanced_comparison_packets(
        packets,
        plan=plan,
        quality_by_evidence_id={
            "duplicate-low": 0.1,
            "duplicate-high": 0.9,
            "a2": 0.5,
            "b1": 0.2,
        },
    )

    selected_ids = [packet.evidence_id for packet in selected]
    assert "duplicate-high" in selected_ids
    assert "duplicate-low" not in selected_ids


def test_duplicate_source_identity_across_subjects_is_selected_only_once() -> None:
    plan = _plan(_subject("a", "Model A"), _subject("b", "Model B"))
    packets = [
        _packet("shared-a", "a", doc_id="shared-doc", chunk_id="shared-chunk"),
        _packet("shared-b", "b", doc_id="shared-doc", chunk_id="shared-chunk"),
        _packet("a-only", "a"),
        _packet("b-only", "b"),
    ]

    selected = select_balanced_comparison_packets(
        packets,
        plan=plan,
        quality_by_evidence_id={
            "shared-a": 0.8,
            "shared-b": 0.9,
            "a-only": 0.5,
            "b-only": 0.4,
        },
    )

    selected_ids = [packet.evidence_id for packet in selected]
    assert "shared-b" in selected_ids
    assert "shared-a" not in selected_ids
    assert len(
        [
            packet
            for packet in selected
            if packet.source.doc_id == "shared-doc"
            and packet.source.chunk_id == "shared-chunk"
        ]
    ) == 1


def test_undeclared_or_multi_subject_packets_are_not_specialized_evidence() -> None:
    plan = _plan(_subject("a", "Model A"), _subject("b", "Model B"))
    undeclared = _packet("unknown", "unknown")
    ambiguous = _packet("ambiguous", "a")
    ambiguous.slot_ids.append("comparison-subject:b")
    original = [undeclared, ambiguous, _packet("a1", "a"), _packet("b1", "b")]

    selected = select_balanced_comparison_packets(
        original,
        plan=plan,
        quality_by_evidence_id={},
    )

    assert [packet.evidence_id for packet in selected] == ["a1", "b1"]
    assert [packet.evidence_id for packet in original] == [
        "unknown",
        "ambiguous",
        "a1",
        "b1",
    ]
