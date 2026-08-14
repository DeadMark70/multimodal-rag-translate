"""One isolated, verified final generation for Agentic v9."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from typing import Any, Protocol

from data_base.agentic_v9.citation_renderer import render_verified_answer
from data_base.agentic_v9.claim_verifier import (
    ClaimVerifier,
    qualify_failed_claim,
    requires_prose_verification,
    verify_claim_deterministically,
)
from data_base.agentic_v9.schemas import (
    ConflictCandidate,
    EvidencePacket,
    FinalAnswerDraft,
    FinalAnswerResult,
    FinalClaim,
    LlmInvoker,
    QueryContract,
    ResponseStatus,
    SlotResolution,
    SufficiencyReport,
    UnresolvedRequirement,
)


_FINAL_GENERATION_UNAVAILABLE_ANSWER = (
    "Final generation was unavailable; evidence is returned as a qualified partial."
)


class PackedEvidenceProjection(Protocol):
    """The typed packet projection exposed by ``PackedEvidenceContext``."""

    packets: Sequence[EvidencePacket]


class FinalAnswerRenderer:
    """Generate once, verify every claim, and never regenerate on a failure."""

    def __init__(
        self, llm_invoker: LlmInvoker, *, citation_format_version: str = "1"
    ) -> None:
        self._invoker = llm_invoker
        self._citation_format_version = citation_format_version

    async def render(
        self,
        *,
        question: str,
        contract: QueryContract,
        packed_packets: Iterable[EvidencePacket] | PackedEvidenceProjection,
        slot_resolutions: Sequence[SlotResolution],
        sufficiency_report: SufficiencyReport | None = None,
        arbitration: Any | None = None,
    ) -> FinalAnswerResult:
        """Use only packed evidence, with one final call and at most one verifier call."""
        packets = _coerce_packed_packets(packed_packets)
        packets_by_id = _packets_by_id(packets)
        try:
            response = await self._invoker.invoke(
                phase="final_answer",
                purpose="final_answer",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Use only supplied evidence. Return JSON with exactly "
                            "supported_findings and unresolved_requirements. Every finding "
                            "must name one required slot and list only packed evidence_ids "
                            "or premise_evidence_ids. Do not return prose outside the JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _final_payload(
                            question=question,
                            contract=contract,
                            packets=packets,
                            slot_resolutions=slot_resolutions,
                            sufficiency_report=sufficiency_report,
                            arbitration=arbitration,
                        ),
                    },
                ],
            )
            if _is_fixed_no_claim_fallback(response):
                return response
            draft = FinalAnswerDraft.model_validate(_response_content(response))
        except Exception:
            return FinalAnswerResult(
                response_status="insufficient",
                answer="Final generation was unavailable; no verified answer was produced.",
                final_generation_count=0,
            )

        accepted: list[FinalClaim] = []
        pending_verification: list[FinalClaim] = []
        for claim in _claims_from_findings(
            draft, contract=contract, packets_by_id=packets_by_id
        ):
            verdict = verify_claim_deterministically(claim, packets_by_id)
            if verdict.reason in {
                "claim_has_no_evidence_ids",
                "claim_references_unpacked_or_unknown_evidence",
                "invalid_evidence",
                "missing_premise_closure",
            }:
                continue
            if not verdict.supported:
                accepted.append(qualify_failed_claim(claim, verdict))
            elif requires_prose_verification(claim):
                pending_verification.append(claim)
            else:
                accepted.append(claim)

        verifier_verdicts = await ClaimVerifier(self._invoker).verify(
            pending_verification, packets_by_id
        )
        for claim in pending_verification:
            verdict = verifier_verdicts[claim.claim_id]
            accepted.append(
                claim if verdict.supported else qualify_failed_claim(claim, verdict)
            )

        supported_claims = [
            claim for claim in accepted if claim.qualified_reason is None
        ]
        used_evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for claim in supported_claims
                for evidence_id in [*claim.evidence_ids, *claim.premise_evidence_ids]
            )
        )
        unresolved_requirements = _unresolved_requirements(
            draft=draft,
            contract=contract,
            slot_resolutions=slot_resolutions,
            supported_claims=supported_claims,
        )
        response_status = _response_status(
            supported_claims, contract, slot_resolutions
        )
        return FinalAnswerResult(
            response_status=response_status,
            answer=render_verified_answer(
                supported_claims,
                packets,
                unresolved_requirements=unresolved_requirements,
                citation_format_version=self._citation_format_version,
            ),
            claims=accepted,
            used_evidence_ids=used_evidence_ids,
            final_generation_count=1,
        )


async def generate_final_answer(
    *,
    question: str,
    contract: QueryContract,
    packed_packets: Iterable[EvidencePacket] | PackedEvidenceProjection,
    slot_resolutions: Sequence[SlotResolution],
    llm_invoker: LlmInvoker,
    sufficiency_report: SufficiencyReport | None = None,
    arbitration: Any | None = None,
    citation_format_version: str = "1",
) -> FinalAnswerResult:
    """Functional entry point for the v9 execution core."""
    return await FinalAnswerRenderer(
        llm_invoker, citation_format_version=citation_format_version
    ).render(
        question=question,
        contract=contract,
        packed_packets=packed_packets,
        slot_resolutions=slot_resolutions,
        sufficiency_report=sufficiency_report,
        arbitration=arbitration,
    )


def _final_payload(
    *,
    question: str,
    contract: QueryContract,
    packets: Sequence[EvidencePacket],
    slot_resolutions: Sequence[SlotResolution],
    sufficiency_report: SufficiencyReport | None,
    arbitration: Any | None,
) -> str:
    return json.dumps(
        {
            "question": question,
            "contract": contract.model_dump(mode="json"),
            "packed_evidence_packets": [
                packet.model_dump(mode="json") for packet in packets
            ],
            "slot_resolutions": [
                resolution.model_dump(mode="json") for resolution in slot_resolutions
            ],
            "sufficiency_report": (
                sufficiency_report.model_dump(mode="json")
                if sufficiency_report is not None
                else None
            ),
            "arbitration": _serialize_arbitration(arbitration),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
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


def _claims_from_findings(
    draft: FinalAnswerDraft,
    *,
    contract: QueryContract,
    packets_by_id: Mapping[str, EvidencePacket],
) -> list[FinalClaim]:
    valid_slots = {slot.slot_id for slot in contract.required_slots}
    claims: list[FinalClaim] = []
    for index, finding in enumerate(draft.supported_findings, start=1):
        evidence_ids = list(
            dict.fromkeys([*finding.evidence_ids, *finding.premise_evidence_ids])
        )
        packets = [packets_by_id.get(evidence_id) for evidence_id in evidence_ids]
        if (
            finding.slot_id not in valid_slots
            or not evidence_ids
            or any(packet is None for packet in packets)
            or any(
                finding.slot_id not in packet.slot_ids
                for packet in packets
                if packet is not None
            )
        ):
            continue
        claims.append(
            FinalClaim(
                claim_id=f"claim-{index}",
                slot_id=finding.slot_id,
                statement=finding.statement,
                support_type=finding.support_type,
                evidence_ids=finding.evidence_ids,
                premise_evidence_ids=finding.premise_evidence_ids,
            )
        )
    return claims


def _unresolved_requirements(
    *,
    draft: FinalAnswerDraft,
    contract: QueryContract,
    slot_resolutions: Sequence[SlotResolution],
    supported_claims: Sequence[FinalClaim],
) -> tuple[UnresolvedRequirement, ...]:
    supported_slots = _supported_claim_slot_ids(supported_claims)
    resolution_by_slot = {item.slot_id: item for item in slot_resolutions}
    provider_reasons = {
        item.slot_id: item.reason for item in draft.unresolved_requirements
    }
    unresolved: list[UnresolvedRequirement] = []
    for slot in contract.required_slots:
        resolution = resolution_by_slot.get(slot.slot_id)
        if (
            slot.slot_id in supported_slots
            and resolution is not None
            and resolution.status == "supported"
        ):
            continue
        reason = provider_reasons.get(slot.slot_id)
        if not reason and resolution is not None:
            reason = resolution.reason
        if not reason:
            reason = (
                "No accepted final finding covered this required slot."
                if resolution is not None and resolution.status == "supported"
                else "No qualified evidence was found for this required slot."
            )
        unresolved.append(
            UnresolvedRequirement(slot_id=slot.slot_id, reason=reason)
        )
    return tuple(unresolved)


def _packets_by_id(packets: Iterable[EvidencePacket]) -> dict[str, EvidencePacket]:
    result: dict[str, EvidencePacket] = {}
    for packet in packets:
        if packet.evidence_id in result:
            raise ValueError(f"duplicate packed evidence ID: {packet.evidence_id}")
        result[packet.evidence_id] = packet
    return result


def _coerce_packed_packets(
    packed_packets: Iterable[EvidencePacket] | PackedEvidenceProjection,
) -> tuple[EvidencePacket, ...]:
    packets = getattr(packed_packets, "packets", packed_packets)
    return tuple(packets)


def _response_content(response: Any) -> Any:
    content = getattr(response, "content", response)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return json.loads(content) if isinstance(content, str) else content


def _is_fixed_no_claim_fallback(response: Any) -> bool:
    """Allow only the budgeted provider's fixed, claim-free terminal fallback."""
    return (
        isinstance(response, FinalAnswerResult)
        and response.response_status == "qualified_partial"
        and response.answer == _FINAL_GENERATION_UNAVAILABLE_ANSWER
        and not response.claims
        and not response.used_evidence_ids
        and response.final_generation_count == 0
    )


def _supported_claim_slot_ids(claims: Sequence[FinalClaim]) -> set[str]:
    return {
        claim.slot_id
        for claim in claims
        if claim.slot_id is not None and claim.qualified_reason is None
    }


def _response_status(
    claims: Sequence[FinalClaim],
    contract: QueryContract,
    slot_resolutions: Sequence[SlotResolution],
) -> ResponseStatus:
    supported_claim_slots = _supported_claim_slot_ids(claims)
    if not supported_claim_slots:
        return "insufficient"
    required_slots = {slot.slot_id for slot in contract.required_slots}
    resolution_by_slot = {item.slot_id: item for item in slot_resolutions}
    if required_slots and required_slots.issubset(supported_claim_slots) and all(
        resolution_by_slot.get(slot_id) is not None
        and resolution_by_slot[slot_id].status == "supported"
        for slot_id in required_slots
    ):
        return "complete"
    return "qualified_partial"


__all__ = ["FinalAnswerDraft", "FinalAnswerRenderer", "generate_final_answer"]
