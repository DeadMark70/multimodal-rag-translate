"""Compact final synthesis context projection for Agentic v9."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_base.agentic_v9.schemas import (
    ConflictCandidate,
    EvidencePacket,
    EvidenceSupportType,
    QueryContract,
    ResponseConstraint,
    SlotResolution,
    SourceLocator,
    SufficiencyReport,
    SynthesisObligation,
    UnresolvedRequirement,
)


class FinalSynthesisSlot(BaseModel):
    """Compact projection of a required direct evidence slot."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    expected_answer_type: str = "text"


class FinalSynthesisEvidence(BaseModel):
    """Compact projection of a packed, qualified evidence packet."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    slot_ids: list[str] = Field(min_length=1)
    statement: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    locator: SourceLocator
    support_type: EvidenceSupportType
    premise_evidence_ids: list[str] = Field(default_factory=list)


class FinalSynthesisContext(BaseModel):
    """Compact input context strictly required for final answer synthesis."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    required_slots: list[FinalSynthesisSlot] = Field(default_factory=list)
    slot_resolutions: list[SlotResolution] = Field(default_factory=list)
    synthesis_obligations: list[SynthesisObligation] = Field(default_factory=list)
    response_constraints: list[ResponseConstraint] = Field(default_factory=list)
    unresolved_requirements: list[UnresolvedRequirement] = Field(default_factory=list)
    packed_evidence: list[FinalSynthesisEvidence] = Field(default_factory=list)
    arbitration: Any | None = None


def build_final_synthesis_context(
    *,
    question: str,
    contract: QueryContract,
    packets: Sequence[EvidencePacket],
    slot_resolutions: Sequence[SlotResolution],
    sufficiency_report: SufficiencyReport | None = None,
    arbitration: Any | None = None,
) -> FinalSynthesisContext:
    """Build a compact, fail-closed synthesis context without diagnostic bloat."""
    del sufficiency_report

    slots = [
        FinalSynthesisSlot(
            slot_id=slot.slot_id,
            description=slot.description,
            expected_answer_type=getattr(slot, "expected_answer_type", "text") or "text",
        )
        for slot in contract.required_slots
    ]

    evidence_aliases = {
        packet.evidence_id: f"E{index}"
        for index, packet in enumerate(packets, start=1)
    }
    projected_evidence = [
        FinalSynthesisEvidence(
            evidence_id=evidence_aliases[packet.evidence_id],
            slot_ids=list(packet.slot_ids),
            statement=packet.statement,
            doc_id=packet.source.doc_id,
            locator=packet.locator,
            support_type=packet.support_type,
            premise_evidence_ids=[
                evidence_aliases[premise_id]
                for premise_id in packet.premise_evidence_ids
                if premise_id in evidence_aliases
            ],
        )
        for packet in packets
    ]

    unresolved: list[UnresolvedRequirement] = []
    resolution_by_slot = {r.slot_id: r for r in slot_resolutions}
    for slot in contract.required_slots:
        res = resolution_by_slot.get(slot.slot_id)
        if res is not None and res.status in {"explicitly_unavailable", "not_found"}:
            unresolved.append(
                UnresolvedRequirement(
                    slot_id=slot.slot_id,
                    reason=res.reason or f"Evidence unavailable for slot {slot.slot_id}.",
                )
            )

    serialized_arbitration = _serialize_arbitration(arbitration)

    return FinalSynthesisContext(
        question=question,
        required_slots=slots,
        slot_resolutions=[
            resolution.model_copy(
                update={
                    "evidence_ids": [
                        evidence_aliases[evidence_id]
                        for evidence_id in resolution.evidence_ids
                        if evidence_id in evidence_aliases
                    ]
                }
            )
            for resolution in slot_resolutions
        ],
        synthesis_obligations=list(contract.synthesis_obligations),
        response_constraints=list(contract.response_constraints),
        unresolved_requirements=unresolved,
        packed_evidence=projected_evidence,
        arbitration=serialized_arbitration,
    )


def _serialize_arbitration(arbitration: Any | None) -> Any | None:
    if arbitration is None:
        return None
    if isinstance(arbitration, ConflictCandidate):
        return arbitration.model_dump(mode="json")
    if isinstance(arbitration, Sequence) and not isinstance(arbitration, (str, bytes)):
        return [
            value.model_dump(mode="json")
            if isinstance(value, ConflictCandidate)
            else value
            for value in arbitration
        ]
    return arbitration


__all__ = [
    "FinalSynthesisContext",
    "FinalSynthesisEvidence",
    "FinalSynthesisSlot",
    "build_final_synthesis_context",
]
