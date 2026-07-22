"""Deterministic, persisted required-slot sufficiency for Agentic v9."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from data_base.agentic_v9.schemas import (
    EvidencePacket,
    QueryContract,
    SlotResolution,
    SufficiencyReport,
)


_USABLE_VALIDATION_STATUSES = {"deterministic_valid", "quote_bound"}


class SufficiencyEvaluation(BaseModel):
    """Serializable gate result for persistence and the bounded repair stage."""

    slot_resolutions: tuple[SlotResolution, ...]
    report: SufficiencyReport
    repairable_slot_ids: tuple[str, ...] = Field(default_factory=tuple)
    repair_stopped_slot_ids: tuple[str, ...] = Field(default_factory=tuple)


def evaluate_sufficiency(
    contract: QueryContract,
    evidence_packets: Iterable[EvidencePacket],
    persisted_resolutions: Iterable[SlotResolution] = (),
) -> SufficiencyEvaluation:
    """Resolve every declared slot without turning absence into positive evidence.

    Existing conflict and explicit-unavailability records are authoritative. A
    ``not_found`` record is provisional and becomes supported when newly supplied,
    validated evidence covers its slot. Required-slot state alone determines the
    response status; optional slots are still persisted but never downgrade a
    complete required-slot projection.
    """
    declared_slots = tuple(contract.required_slots)
    _require_unique_slot_ids(declared_slots)
    declared_slot_ids = {slot.slot_id for slot in declared_slots}
    packets_by_id = _packets_by_id(evidence_packets)
    resolutions_by_slot = _resolutions_by_slot(
        persisted_resolutions, declared_slot_ids, packets_by_id
    )

    slot_resolutions = tuple(
        _resolve_slot(
            slot_id=slot.slot_id,
            persisted=resolutions_by_slot.get(slot.slot_id),
            evidence_ids=_usable_evidence_ids_for_slot(slot.slot_id, packets_by_id),
        )
        for slot in declared_slots
    )

    required_slot_ids = {slot.slot_id for slot in declared_slots if slot.required}
    required_resolutions = tuple(
        resolution
        for resolution in slot_resolutions
        if resolution.slot_id in required_slot_ids
    )
    supported = _slot_ids_with_status(required_resolutions, "supported")
    conflicted = _slot_ids_with_status(required_resolutions, "conflicted")
    unavailable = _slot_ids_with_status(
        required_resolutions, "explicitly_unavailable"
    )
    missing = _slot_ids_with_status(required_resolutions, "not_found")

    evidence_complete = bool(required_slot_ids) and len(supported) == len(
        required_slot_ids
    )
    answerable = bool(supported)
    response_status = (
        "complete"
        if evidence_complete
        else "qualified_partial"
        if answerable
        else "insufficient"
    )
    report = SufficiencyReport(
        evidence_complete=evidence_complete,
        answerable=answerable,
        response_status=response_status,
        supported_slot_ids=list(supported),
        conflicted_slot_ids=list(conflicted),
        explicitly_unavailable_slot_ids=list(unavailable),
        not_found_slot_ids=list(missing),
        stop_reason=_stop_reason(
            has_required_slots=bool(required_slot_ids),
            explicitly_unavailable_slot_ids=unavailable,
        ),
    )
    return SufficiencyEvaluation(
        slot_resolutions=slot_resolutions,
        report=report,
        repairable_slot_ids=missing,
        repair_stopped_slot_ids=unavailable,
    )


def _require_unique_slot_ids(slots: tuple[object, ...]) -> None:
    slot_ids = [slot.slot_id for slot in slots]  # type: ignore[attr-defined]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("query contract contains duplicate slot IDs")


def _packets_by_id(
    evidence_packets: Iterable[EvidencePacket],
) -> dict[str, EvidencePacket]:
    packets_by_id: dict[str, EvidencePacket] = {}
    for packet in evidence_packets:
        if packet.evidence_id in packets_by_id:
            raise ValueError(f"duplicate evidence ID: {packet.evidence_id}")
        packets_by_id[packet.evidence_id] = packet
    return packets_by_id


def _resolutions_by_slot(
    persisted_resolutions: Iterable[SlotResolution],
    declared_slot_ids: set[str],
    packets_by_id: dict[str, EvidencePacket],
) -> dict[str, SlotResolution]:
    resolutions_by_slot: dict[str, SlotResolution] = {}
    for resolution in persisted_resolutions:
        if resolution.slot_id not in declared_slot_ids:
            raise ValueError(
                f"persisted resolution references undeclared slot: {resolution.slot_id}"
            )
        if resolution.slot_id in resolutions_by_slot:
            raise ValueError(f"duplicate persisted resolution: {resolution.slot_id}")
        _validate_resolution_evidence(resolution, packets_by_id)
        resolutions_by_slot[resolution.slot_id] = resolution
    return resolutions_by_slot


def _validate_resolution_evidence(
    resolution: SlotResolution, packets_by_id: dict[str, EvidencePacket]
) -> None:
    for evidence_id in resolution.evidence_ids:
        packet = packets_by_id.get(evidence_id)
        if packet is None or not _is_usable_packet(packet):
            raise ValueError(
                f"persisted resolution references unknown or invalid evidence: {evidence_id}"
            )
        if resolution.slot_id not in packet.slot_ids:
            raise ValueError(
                f"persisted resolution evidence does not cover slot: {resolution.slot_id}"
            )


def _resolve_slot(
    *,
    slot_id: str,
    persisted: SlotResolution | None,
    evidence_ids: tuple[str, ...],
) -> SlotResolution:
    if persisted is not None and persisted.status in {
        "conflicted",
        "explicitly_unavailable",
    }:
        if evidence_ids and persisted.status == "explicitly_unavailable":
            raise ValueError(
                f"explicitly unavailable slot has valid evidence: {slot_id}"
            )
        return persisted
    if persisted is not None and persisted.status == "supported":
        return persisted
    if evidence_ids:
        return SlotResolution(
            slot_id=slot_id,
            status="supported",
            evidence_ids=list(evidence_ids),
            resolution_stage="sufficiency_gate",
        )
    return SlotResolution(
        slot_id=slot_id,
        status="not_found",
        reason="No valid evidence or persisted resolution is available.",
        resolution_stage="sufficiency_gate",
    )


def _usable_evidence_ids_for_slot(
    slot_id: str, packets_by_id: dict[str, EvidencePacket]
) -> tuple[str, ...]:
    return tuple(
        evidence_id
        for evidence_id, packet in packets_by_id.items()
        if slot_id in packet.slot_ids and _is_usable_packet(packet)
    )


def _is_usable_packet(packet: EvidencePacket) -> bool:
    return (
        packet.validation_status in _USABLE_VALIDATION_STATUSES
        and packet.support_type != "contradictory"
    )


def _slot_ids_with_status(
    resolutions: tuple[SlotResolution, ...], status: str
) -> tuple[str, ...]:
    return tuple(resolution.slot_id for resolution in resolutions if resolution.status == status)


def _stop_reason(
    *, has_required_slots: bool, explicitly_unavailable_slot_ids: tuple[str, ...]
) -> str | None:
    if not has_required_slots:
        return "no_required_slots"
    if explicitly_unavailable_slot_ids:
        return "explicitly_unavailable"
    return None
