"""Deterministic subject-balanced evidence selection for Agentic v9."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from data_base.agentic_v9.schemas import ComparisonPlan, EvidencePacket


@dataclass(frozen=True, slots=True)
class _Candidate:
    packet: EvidencePacket
    subject_id: str
    quality: float
    index: int


def comparison_final_limit(subject_count: int) -> int:
    """Return the approved bounded final-evidence limit."""
    if subject_count == 2:
        return 4
    if 3 <= subject_count <= 4:
        return 6
    raise ValueError("comparison subject count must be between 2 and 4")


def select_balanced_comparison_packets(
    packets: Sequence[EvidencePacket],
    *,
    plan: ComparisonPlan,
    quality_by_evidence_id: Mapping[str, float],
) -> tuple[EvidencePacket, ...]:
    """Retain subject coverage without imposing a fixed score threshold."""
    subject_ids = [subject.subject_id for subject in plan.subjects]
    grouped: dict[str, list[_Candidate]] = {subject_id: [] for subject_id in subject_ids}
    for index, packet in enumerate(packets):
        matches = [
            subject_id
            for subject_id in subject_ids
            if f"comparison-subject:{subject_id}" in packet.slot_ids
        ]
        if len(matches) != 1:
            continue
        subject_id = matches[0]
        grouped[subject_id].append(
            _Candidate(
                packet=packet,
                subject_id=subject_id,
                quality=_quality(quality_by_evidence_id.get(packet.evidence_id)),
                index=index,
            )
        )

    ranked = {
        subject_id: _deduplicated_ranked(candidates)
        for subject_id, candidates in grouped.items()
    }
    limit = comparison_final_limit(len(subject_ids))
    if len(subject_ids) == 2:
        selected = [
            candidate
            for subject_id in subject_ids
            for candidate in ranked[subject_id][:2]
        ]
        return tuple(candidate.packet for candidate in selected[:limit])

    selected = [
        candidates[0]
        for subject_id in subject_ids
        if (candidates := ranked[subject_id])
    ]
    selected_ids = {candidate.packet.evidence_id for candidate in selected}
    remaining = sorted(
        (
            candidate
            for subject_id in subject_ids
            for candidate in ranked[subject_id][1:2]
            if candidate.packet.evidence_id not in selected_ids
        ),
        key=_quality_order,
    )
    selected.extend(remaining[: max(limit - len(selected), 0)])
    return tuple(candidate.packet for candidate in selected[:limit])


def _deduplicated_ranked(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    winners: dict[tuple[str, ...], _Candidate] = {}
    for candidate in candidates:
        identity = _source_identity(candidate.packet)
        previous = winners.get(identity)
        if previous is None or _quality_order(candidate) < _quality_order(previous):
            winners[identity] = candidate
    return sorted(winners.values(), key=_quality_order)


def _source_identity(packet: EvidencePacket) -> tuple[str, ...]:
    source = packet.source
    if source.chunk_id:
        return ("chunk", source.doc_id, source.chunk_id)
    if source.source_span_hash:
        return ("span", source.doc_id, source.source_span_hash)
    return ("packet", packet.evidence_id)


def _quality(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if isfinite(float(value)) else 0.0


def _quality_order(candidate: _Candidate) -> tuple[float, int, str]:
    return (-candidate.quality, candidate.index, candidate.packet.evidence_id)


__all__ = [
    "comparison_final_limit",
    "select_balanced_comparison_packets",
]
