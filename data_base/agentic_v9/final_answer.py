"""One isolated, verified final generation for Agentic v9."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Protocol

from core.prompt_loader import PromptRegistry
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
    SlotResolution,
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
                            _final_answer_prompt()
                        ),
                    },
                    {
                        "role": "user",
                        "content": _final_payload(
                            question=question,
                            contract=contract,
                            packets=packets,
                            slot_resolutions=slot_resolutions,
                            arbitration=arbitration,
                        ),
                    },
                ],
            )
        except Exception:
            unresolved_requirements = _required_unresolved_requirements(
                contract, slot_resolutions
            )
            return FinalAnswerResult(
                response_status="qualified_partial" if packets else "insufficient",
                answer=render_verified_answer(
                    (),
                    packets,
                    unresolved_requirements=unresolved_requirements,
                    citation_format_version=self._citation_format_version,
                ),
                final_generation_count=0,
            )
        if _is_fixed_no_claim_fallback(response):
            return response
        if isinstance(response, FinalAnswerResult):
            response = {
                "supported_findings": [
                    {
                        "slot_id": claim.slot_id
                        or _single_supported_slot_id(slot_resolutions),
                        "statement": claim.statement,
                        "evidence_ids": [
                            *claim.evidence_ids,
                            *claim.premise_evidence_ids,
                        ],
                    }
                    for claim in response.claims
                ],
                "unresolved_requirements": [],
            }
        try:
            draft = FinalAnswerDraft.model_validate(_response_content(response))
        except Exception:
            draft = FinalAnswerDraft()

        accepted: list[FinalClaim] = []
        unresolved: list[FinalClaim] = []
        contract_slot_ids = {slot.slot_id for slot in contract.required_slots}
        resolutions_by_slot = {
            resolution.slot_id: resolution for resolution in slot_resolutions
        }
        for index, finding in enumerate(draft.supported_findings, start=1):
            resolution = resolutions_by_slot.get(finding.slot_id)
            if (
                finding.slot_id not in contract_slot_ids
                or resolution is None
                or resolution.status != "supported"
                or not finding.evidence_ids
                or any(
                    evidence_id not in packets_by_id
                    or finding.slot_id not in packets_by_id[evidence_id].slot_ids
                    or evidence_id not in resolution.evidence_ids
                    for evidence_id in finding.evidence_ids
                )
            ):
                continue
            claim = FinalClaim(
                claim_id=f"claim-{index}",
                slot_id=finding.slot_id,
                statement=finding.statement,
                support_type="direct",
                evidence_ids=finding.evidence_ids,
            )
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
                unresolved.append(claim)
            else:
                accepted.append(claim)

        verifier_verdicts = await ClaimVerifier(self._invoker).verify(
            unresolved, packets_by_id
        )
        for claim in unresolved:
            verdict = verifier_verdicts[claim.claim_id]
            accepted.append(
                claim if verdict.supported else qualify_failed_claim(claim, verdict)
            )

        used_evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for claim in accepted
                for evidence_id in [*claim.evidence_ids, *claim.premise_evidence_ids]
            )
        )
        response_status = _response_status(accepted, slot_resolutions)
        unresolved_requirements = _required_unresolved_requirements(
            contract, slot_resolutions
        )
        return FinalAnswerResult(
            response_status=response_status,
            answer=render_verified_answer(
                accepted,
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
        arbitration=arbitration,
    )


def _final_payload(
    *,
    question: str,
    contract: QueryContract,
    packets: Sequence[EvidencePacket],
    slot_resolutions: Sequence[SlotResolution],
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


def _response_status(
    claims: Sequence[FinalClaim], slot_resolutions: Sequence[SlotResolution]
) -> str:
    if not claims:
        return "insufficient"
    if all(resolution.status == "supported" for resolution in slot_resolutions) and all(
        claim.qualified_reason is None for claim in claims
    ):
        return "complete"
    return "qualified_partial"


def _required_unresolved_requirements(
    contract: QueryContract,
    slot_resolutions: Sequence[SlotResolution],
) -> list[UnresolvedRequirement]:
    resolutions_by_slot = {
        resolution.slot_id: resolution for resolution in slot_resolutions
    }
    unresolved: list[UnresolvedRequirement] = []
    for slot in contract.required_slots:
        if not slot.required:
            continue
        resolution = resolutions_by_slot.get(slot.slot_id)
        if resolution is not None and resolution.status == "supported":
            continue
        status = resolution.status if resolution is not None else "not_found"
        reason = (
            resolution.reason
            if resolution is not None and resolution.reason
            else {
                "conflicted": "Conflicting source-bound evidence remains unresolved.",
                "explicitly_unavailable": "Required evidence is explicitly unavailable.",
                "not_found": "Required source-bound evidence was not found.",
            }[status]
        )
        unresolved.append(
            UnresolvedRequirement(slot_id=slot.slot_id, reason=reason)
        )
    return unresolved


@lru_cache(maxsize=1)
def _final_answer_prompt() -> str:
    registry = PromptRegistry(
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "agentic_v9_final_answer.json"
    )
    return registry.format("final_answer")


def _single_supported_slot_id(
    slot_resolutions: Sequence[SlotResolution],
) -> str:
    supported = [
        resolution.slot_id
        for resolution in slot_resolutions
        if resolution.status == "supported"
    ]
    return supported[0] if len(supported) == 1 else "legacy-unbound"


__all__ = ["FinalAnswerDraft", "FinalAnswerRenderer", "generate_final_answer"]
