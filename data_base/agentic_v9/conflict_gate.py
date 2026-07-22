"""Deterministic, scope-aware conflict candidates for Agentic v9 evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from data_base.agentic_v9.schemas import (
    ConflictCandidate,
    EvidencePacket,
    EvidenceScope,
    LlmInvoker,
    ScopeMatch,
)


_SCOPE_FIELDS = (
    "metric",
    "dataset",
    "split",
    "model_variant",
    "training_protocol",
    "prompt_setting",
)
_USABLE_VALIDATION_STATUSES = {"deterministic_valid", "quote_bound"}


def match_evidence_scope(left: EvidenceScope, right: EvidenceScope) -> ScopeMatch:
    """Classify experimental-scope equivalence without inventing missing scope.

    A scope is the same only when every comparison-critical dimension is known
    and equal.  A known mismatch makes it different; otherwise a missing
    dimension leaves the relationship unknown.
    """
    unknown = False
    for field_name in _SCOPE_FIELDS:
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        if left_value is None or right_value is None:
            unknown = True
        elif left_value != right_value:
            return "different"
    return "unknown" if unknown else "same"


def detect_conflict_candidates(
    evidence_packets: Iterable[EvidencePacket],
) -> list[ConflictCandidate]:
    """Return unresolved candidates for incompatible values in a shared slot.

    Packets with a known scope difference are distinct measurements, not a
    conflict.  Incompatible values with incomplete scope remain explicit
    ``scope_ambiguous`` candidates so a later stage can arbitrate or qualify
    them; they are never converted into a silent no-conflict outcome.
    """
    packets = _validated_usable_packets(evidence_packets)
    candidates: list[ConflictCandidate] = []
    for index, left in enumerate(packets):
        for right in packets[index + 1 :]:
            if not _values_are_incompatible(left, right):
                continue
            scope_match = match_evidence_scope(left.scope, right.scope)
            if scope_match == "different":
                continue
            for slot_id in sorted(set(left.slot_ids).intersection(right.slot_ids)):
                evidence_ids = sorted((left.evidence_id, right.evidence_id))
                candidates.append(
                    ConflictCandidate(
                        candidate_id=_candidate_id(slot_id, evidence_ids),
                        slot_id=slot_id,
                        evidence_ids=evidence_ids,
                        scope_match=scope_match,
                        reason=(
                            "same_scope_incompatible_values"
                            if scope_match == "same"
                            else "scope_ambiguous: incomplete comparison scope; "
                            "arbitrate or qualify before use."
                        ),
                    )
                )
    return candidates


async def arbitrate_persisted_unresolved_candidates(
    *,
    persisted_candidates: Iterable[ConflictCandidate],
    persisted_candidate_ids: Iterable[str],
    evidence_packets: Iterable[EvidencePacket],
    llm_invoker: LlmInvoker,
) -> Any | None:
    """Arbitrate persisted unresolved candidates in one injected v9 call.

    The caller supplies the persistence acknowledgement separately so candidates
    freshly detected in memory cannot accidentally invoke a provider.  At most
    one call is made for all eligible candidates; the payload contains only
    candidate metadata and provenance-bound evidence packets, never subtask
    answers or legacy synthesizer output.
    """
    persisted_ids = set(persisted_candidate_ids)
    eligible = [
        candidate
        for candidate in persisted_candidates
        if candidate.unresolved and candidate.candidate_id in persisted_ids
    ]
    if not eligible:
        return None

    packets_by_id = {
        packet.evidence_id: packet
        for packet in _validated_usable_packets(evidence_packets)
    }
    required_evidence_ids = {
        evidence_id for candidate in eligible for evidence_id in candidate.evidence_ids
    }
    missing_evidence_ids = required_evidence_ids.difference(packets_by_id)
    if missing_evidence_ids:
        missing = ", ".join(sorted(missing_evidence_ids))
        raise ValueError(
            f"persisted conflict candidate references unknown evidence: {missing}"
        )

    return await llm_invoker.invoke(
        phase="conflict_arbitration",
        purpose="conflict_arbitration",
        messages=[
            {
                "role": "user",
                "content": _arbitration_payload(
                    candidates=eligible,
                    packets=[
                        packets_by_id[evidence_id]
                        for evidence_id in sorted(required_evidence_ids)
                    ],
                ),
            }
        ],
    )


def _validated_usable_packets(
    evidence_packets: Iterable[EvidencePacket],
) -> list[EvidencePacket]:
    packets_by_id: dict[str, EvidencePacket] = {}
    for packet in evidence_packets:
        if packet.evidence_id in packets_by_id:
            raise ValueError(f"duplicate evidence ID: {packet.evidence_id}")
        if (
            packet.validation_status in _USABLE_VALIDATION_STATUSES
            and packet.support_type != "contradictory"
        ):
            packets_by_id[packet.evidence_id] = packet
    return list(packets_by_id.values())


def _values_are_incompatible(left: EvidencePacket, right: EvidencePacket) -> bool:
    left_value = _comparison_value(left)
    right_value = _comparison_value(right)
    return (
        left_value is not None and right_value is not None and left_value != right_value
    )


def _comparison_value(packet: EvidencePacket) -> object | None:
    """Prefer normalized numeric values; raw values are the bounded fallback."""
    return (
        packet.normalized_value
        if packet.normalized_value is not None
        else packet.raw_value
    )


def _candidate_id(slot_id: str, evidence_ids: Sequence[str]) -> str:
    return f"conflict:{slot_id}:{':'.join(evidence_ids)}"


def _arbitration_payload(
    *, candidates: Sequence[ConflictCandidate], packets: Sequence[EvidencePacket]
) -> str:
    """Serialize a bounded arbitration input containing evidence, not answers."""
    return json.dumps(
        {
            "candidates": [
                candidate.model_dump(mode="json") for candidate in candidates
            ],
            "evidence_packets": [packet.model_dump(mode="json") for packet in packets],
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


__all__ = [
    "arbitrate_persisted_unresolved_candidates",
    "detect_conflict_candidates",
    "match_evidence_scope",
]
