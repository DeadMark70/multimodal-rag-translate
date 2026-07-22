"""One isolated, verified final generation for Agentic v9."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

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
    FinalAnswerResult,
    FinalClaim,
    LlmInvoker,
    QueryContract,
    SlotResolution,
)


class FinalAnswerDraft(BaseModel):
    """Strict, typed provider output before deterministic claim verification."""

    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    claims: list[FinalClaim] = Field(default_factory=list)


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
                            "Answer only from supplied evidence. Return JSON with exactly "
                            "answer and claims. Every claim must list its evidence_ids or "
                            "premise_evidence_ids; do not cite any other ID."
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
            if isinstance(response, FinalAnswerResult):
                return response
            draft = FinalAnswerDraft.model_validate(_response_content(response))
        except Exception:
            return FinalAnswerResult(
                response_status="qualified_partial" if packets else "insufficient",
                answer="Final generation was unavailable; no verified answer was produced.",
                final_generation_count=0,
            )

        accepted: list[FinalClaim] = []
        unresolved: list[FinalClaim] = []
        for claim in draft.claims:
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
        return FinalAnswerResult(
            response_status=response_status,
            answer=render_verified_answer(
                accepted,
                packets,
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


__all__ = ["FinalAnswerDraft", "FinalAnswerRenderer", "generate_final_answer"]
