"""One isolated, verified final generation for Agentic v9."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from typing import Any, Protocol

from data_base.agentic_v9.citation_renderer import render_verified_answer
from data_base.agentic_v9.claim_verifier import (
    ClaimVerifier,
    gate_as_verdict,
    gate_claim_deterministically,
    qualify_failed_claim,
)
from data_base.agentic_v9.final_synthesis_context import (
    build_final_synthesis_context,
)
from data_base.agentic_v9.schemas import (
    ClaimSupportType,
    EvidencePacket,
    FinalAnswerDraft,
    FinalAnswerResult,
    FinalClaim,
    LlmInvoker,
    QueryContract,
    ResponseStatus,
    SlotResolution,
    SufficiencyReport,
    UnresolvedObligation,
    UnresolvedRequirement,
)


_FINAL_GENERATION_UNAVAILABLE_ANSWER = (
    "Final generation was unavailable; evidence is returned as a qualified partial."
)

_OBLIGATION_SUPPORT_TYPE: dict[str, ClaimSupportType] = {
    "aggregation": "calculated",
    "comparison": "comparative_inference",
    "selection": "comparative_inference",
    "causal": "comparative_inference",
    "qualification": "qualified",
}


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
                            "supported_findings, synthesized_findings, unresolved_requirements, "
                            "and unresolved_obligations. Every supported finding must name one "
                            "required slot (S#) and list only packed evidence_ids. Every synthesized "
                            "finding must name one obligation (O#) and list all required premise_evidence_ids. "
                            "Do not return prose outside the JSON."
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
            gate = gate_claim_deterministically(claim, packets_by_id)
            if gate.status == "accepted":
                accepted.append(claim)
            elif gate.status == "verify":
                pending_verification.append(claim)
            else:
                accepted.append(
                    qualify_failed_claim(
                        claim,
                        gate_as_verdict(gate),
                    )
                )

        verifier_verdicts = await ClaimVerifier(self._invoker).verify(
            pending_verification,
            packets_by_id,
            contract=contract,
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
        unresolved_requirements, unresolved_obligations = _unresolved_items(
            draft=draft,
            contract=contract,
            slot_resolutions=slot_resolutions,
            supported_claims=supported_claims,
        )
        response_status = _response_status(
            supported_claims,
            contract,
            slot_resolutions,
            unresolved_requirements=unresolved_requirements,
            unresolved_obligations=unresolved_obligations,
        )
        all_unresolved = list(unresolved_requirements) + list(unresolved_obligations)
        return FinalAnswerResult(
            response_status=response_status,
            answer=render_verified_answer(
                supported_claims,
                packets,
                unresolved_requirements=all_unresolved,
                citation_format_version=self._citation_format_version,
            ),
            claims=accepted,
            used_evidence_ids=used_evidence_ids,
            final_generation_count=1,
            unresolved_requirements=list(unresolved_requirements),
            unresolved_obligations=list(unresolved_obligations),
            claim_verifier_call_count=1 if pending_verification else 0,
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
    context = build_final_synthesis_context(
        question=question,
        contract=contract,
        packets=packets,
        slot_resolutions=slot_resolutions,
        sufficiency_report=sufficiency_report,
        arbitration=arbitration,
    )
    return context.model_dump_json()


def _claims_from_findings(
    draft: FinalAnswerDraft,
    *,
    contract: QueryContract,
    packets_by_id: Mapping[str, EvidencePacket],
) -> list[FinalClaim]:
    valid_slots = {slot.slot_id for slot in contract.required_slots}
    obligation_by_id = {
        obligation.obligation_id: obligation
        for obligation in contract.synthesis_obligations
    }
    claims: list[FinalClaim] = []
    claim_counter = 1

    # 1. Direct findings
    for finding in draft.supported_findings:
        evidence_ids = list(
            dict.fromkeys([*finding.evidence_ids, *finding.premise_evidence_ids])
        )
        packets = [packets_by_id.get(evidence_id) for evidence_id in evidence_ids]
        if (
            finding.slot_id not in valid_slots
            or any(
                finding.slot_id not in packet.slot_ids
                for packet in packets
                if packet is not None
            )
        ):
            continue
        claims.append(
            FinalClaim(
                claim_id=f"claim-{claim_counter}",
                slot_id=finding.slot_id,
                obligation_id=None,
                statement=finding.statement,
                support_type="direct",
                evidence_ids=finding.evidence_ids,
                premise_evidence_ids=finding.premise_evidence_ids,
            )
        )
        claim_counter += 1

    # 2. Synthesized findings
    for finding in draft.synthesized_findings:
        obligation = obligation_by_id.get(finding.obligation_id)
        if obligation is None:
            continue
        premise_ids = list(dict.fromkeys(finding.premise_evidence_ids))
        premise_packets = [packets_by_id.get(eid) for eid in premise_ids]
        typed_premise_packets = [p for p in premise_packets if p is not None]

        # Validate that every dependency slot of this obligation contributes at least one premise evidence ID
        has_full_dependency_closure = True
        for dep_slot_id in obligation.depends_on_slot_ids:
            if not any(dep_slot_id in p.slot_ids for p in typed_premise_packets):
                # Leave unknown packet IDs for the deterministic gate so the
                # rejected candidate keeps its provenance in the final result.
                if not any(p is None for p in premise_packets):
                    has_full_dependency_closure = False
                break
        if not has_full_dependency_closure:
            continue

        derived_support_type = _OBLIGATION_SUPPORT_TYPE.get(
            obligation.kind, "comparative_inference"
        )
        claims.append(
            FinalClaim(
                claim_id=f"claim-{claim_counter}",
                slot_id=None,
                obligation_id=finding.obligation_id,
                statement=finding.statement,
                support_type=derived_support_type,
                evidence_ids=[],
                premise_evidence_ids=premise_ids,
            )
        )
        claim_counter += 1

    return claims


def _unresolved_items(
    *,
    draft: FinalAnswerDraft,
    contract: QueryContract,
    slot_resolutions: Sequence[SlotResolution],
    supported_claims: Sequence[FinalClaim],
) -> tuple[tuple[UnresolvedRequirement, ...], tuple[UnresolvedObligation, ...]]:
    supported_slots = _supported_claim_slot_ids(supported_claims)
    supported_obs = _supported_claim_obligation_ids(supported_claims)
    resolution_by_slot = {item.slot_id: item for item in slot_resolutions}
    provider_slot_reasons = {
        item.slot_id: item.reason for item in draft.unresolved_requirements
    }
    provider_ob_reasons = {
        item.obligation_id: item.reason for item in draft.unresolved_obligations
    }

    unresolved_slots: list[UnresolvedRequirement] = []
    for slot in contract.required_slots:
        resolution = resolution_by_slot.get(slot.slot_id)
        if (
            slot.slot_id in supported_slots
            and resolution is not None
            and resolution.status == "supported"
        ):
            continue
        reason = provider_slot_reasons.get(slot.slot_id)
        if not reason and resolution is not None:
            reason = resolution.reason
        if not reason:
            reason = (
                "No accepted final finding covered this required slot."
                if resolution is not None and resolution.status == "supported"
                else "No qualified evidence was found for this required slot."
            )
        unresolved_slots.append(
            UnresolvedRequirement(slot_id=slot.slot_id, reason=reason)
        )

    unresolved_obs: list[UnresolvedObligation] = []
    for obligation in contract.synthesis_obligations:
        if obligation.obligation_id in supported_obs:
            continue
        reason = provider_ob_reasons.get(obligation.obligation_id)
        if not reason:
            reason = f"Obligation {obligation.obligation_id} ({obligation.description}) was not synthesized from verified premises."
        unresolved_obs.append(
            UnresolvedObligation(
                obligation_id=obligation.obligation_id, reason=reason
            )
        )

    return tuple(unresolved_slots), tuple(unresolved_obs)


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


def _supported_claim_obligation_ids(claims: Sequence[FinalClaim]) -> set[str]:
    return {
        claim.obligation_id
        for claim in claims
        if claim.obligation_id is not None and claim.qualified_reason is None
    }


def reduce_terminal_status(
    *,
    contract: QueryContract,
    slot_resolutions: Sequence[SlotResolution],
    accepted_claims: Sequence[FinalClaim],
    unresolved_requirements: Sequence[UnresolvedRequirement] = (),
    unresolved_obligations: Sequence[UnresolvedObligation] = (),
) -> ResponseStatus:
    """Return complete, qualified_partial, or insufficient from verified output."""
    supported_claim_slots = _supported_claim_slot_ids(accepted_claims)
    supported_claim_obs = _supported_claim_obligation_ids(accepted_claims)

    if not supported_claim_slots and not supported_claim_obs:
        return "insufficient"

    if unresolved_requirements or unresolved_obligations:
        return "qualified_partial"

    required_slots = {slot.slot_id for slot in contract.required_slots}
    resolution_by_slot = {item.slot_id: item for item in slot_resolutions}
    all_slots_supported = (
        required_slots
        and required_slots.issubset(supported_claim_slots)
        and all(
            resolution_by_slot.get(slot_id) is not None
            and resolution_by_slot[slot_id].status == "supported"
            for slot_id in required_slots
        )
    )
    required_obligations = {ob.obligation_id for ob in contract.synthesis_obligations}
    all_obs_supported = (
        not required_obligations or required_obligations.issubset(supported_claim_obs)
    )

    if all_slots_supported and all_obs_supported:
        return "complete"
    return "qualified_partial"


def _response_status(
    claims: Sequence[FinalClaim],
    contract: QueryContract,
    slot_resolutions: Sequence[SlotResolution],
    *,
    unresolved_requirements: Sequence[UnresolvedRequirement] = (),
    unresolved_obligations: Sequence[UnresolvedObligation] = (),
) -> ResponseStatus:
    return reduce_terminal_status(
        contract=contract,
        slot_resolutions=slot_resolutions,
        accepted_claims=claims,
        unresolved_requirements=unresolved_requirements,
        unresolved_obligations=unresolved_obligations,
    )


__all__ = [
    "FinalAnswerDraft",
    "FinalAnswerRenderer",
    "generate_final_answer",
    "reduce_terminal_status",
]
